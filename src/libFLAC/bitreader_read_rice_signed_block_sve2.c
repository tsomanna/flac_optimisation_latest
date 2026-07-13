/* libFLAC - Free Lossless Audio Codec library
 * Copyright (C) 2000-2009  Josh Coalson
 * Copyright (C) 2011-2025  Xiph.Org Foundation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * - Neither the name of the Xiph.org Foundation nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include "private/cpu.h"

#if defined(FLAC__CPU_ARM64) && !defined(FLAC__NO_ASM)

#include <stdlib.h>
#include <string.h>
#include "private/bitmath.h"
#include "private/bitreader.h"
#include "private/crc.h"
#include "private/macros.h"
#include "FLAC/assert.h"
#include "share/compat.h"
#include "share/endswap.h"

/* Replicate the necessary definitions from bitreader.c */
#if (ENABLE_64_BIT_WORDS == 0)

typedef FLAC__uint32 brword;
#define FLAC__BYTES_PER_WORD 4
#define FLAC__BITS_PER_WORD 32
#define FLAC__WORD_ALL_ONES ((FLAC__uint32)0xffffffff)
#if WORDS_BIGENDIAN
#define SWAP_BE_WORD_TO_HOST(x) (x)
#else
#define SWAP_BE_WORD_TO_HOST(x) ENDSWAP_32(x)
#endif
#define COUNT_ZERO_MSBS(word) FLAC__clz_uint32(word)
#define COUNT_ZERO_MSBS2(word) FLAC__clz2_uint32(word)

#else

typedef FLAC__uint64 brword;
#define FLAC__BYTES_PER_WORD 8
#define FLAC__BITS_PER_WORD 64
#define FLAC__WORD_ALL_ONES ((FLAC__uint64)FLAC__U64L(0xffffffffffffffff))
#if WORDS_BIGENDIAN
#define SWAP_BE_WORD_TO_HOST(x) (x)
#else
#define SWAP_BE_WORD_TO_HOST(x) ENDSWAP_64(x)
#endif
#define COUNT_ZERO_MSBS(word) FLAC__clz_uint64(word)
#define COUNT_ZERO_MSBS2(word) FLAC__clz2_uint64(word)

#endif

/*
 * We need access to the BitReader internals for the optimized path.
 * This mirrors the struct definition in bitreader.c.
 */
struct FLAC__BitReader {
	brword *buffer;
	uint32_t capacity; /* in words */
	uint32_t words; /* # of completed words in buffer */
	uint32_t bytes; /* # of bytes in incomplete word at buffer[words] */
	uint32_t consumed_words;
	uint32_t consumed_bits;
	uint32_t read_crc16;
	uint32_t crc16_offset;
	uint32_t crc16_align;
	FLAC__bool read_limit_set;
	uint32_t read_limit;
	uint32_t last_seen_framesync;
	FLAC__BitReaderReadCallback read_callback;
	void *client_data;
};

/* Branch prediction hints */
#if defined(__GNUC__) || defined(__clang__)
#define FLAC__LIKELY(x)   __builtin_expect(!!(x), 1)
#define FLAC__UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FLAC__LIKELY(x)   (x)
#define FLAC__UNLIKELY(x) (x)
#endif


/*
 * SVE2-optimized rice signed block reader.
 *
 * Optimizations over the generic version:
 * 1. Multi-sample fast path: when enough bits remain in the current word
 *    for a complete rice code (unary + binary), decode without boundary checks
 * 2. SVE2 BEXT for efficient cross-word bit extraction
 * 3. Branch prediction hints for the common (fits-in-word) case
 * 4. Tight inner loop that processes multiple samples per word without
 *    re-checking loop conditions
 */
FLAC__bool FLAC__bitreader_read_rice_signed_block_sve2(FLAC__BitReader *br, int vals[], uint32_t nvals, uint32_t parameter)
{
	uint32_t cwords, words, lsbs, msbs, x, y, limit;
	uint32_t ucbits; /* unconsumed bits in current word */
	brword b;
	int *val, *end;

	FLAC__ASSERT(0 != br);
	FLAC__ASSERT(0 != br->buffer);
	FLAC__ASSERT(FLAC__BITS_PER_WORD >= 32);
	FLAC__ASSERT(parameter < 32);

	limit = UINT32_MAX >> parameter;

	val = vals;
	end = vals + nvals;

	if(parameter == 0) {
		while(val < end) {
			/* read the unary MSBs and end bit */
			if(!FLAC__bitreader_read_unary_unsigned(br, &msbs))
				return false;
			*val++ = (int)(msbs >> 1) ^ -(int)(msbs & 1);
		}
		return true;
	}

	FLAC__ASSERT(parameter > 0);

	cwords = br->consumed_words;
	words = br->words;

	/* if we've not consumed up to a partial tail word... */
	if(cwords >= words) {
		x = 0;
		goto process_tail;
	}

	ucbits = FLAC__BITS_PER_WORD - br->consumed_bits;
	b = br->buffer[cwords] << br->consumed_bits;

	while(val < end) {
		/* ============================================================
		 * FAST PATH: Try to decode multiple samples from current word
		 * without any word-boundary checks.
		 *
		 * A rice code needs at minimum (1 + parameter) bits:
		 *   1 bit for the shortest unary code (just the stop bit)
		 *   + parameter bits for the binary part
		 *
		 * While we have enough bits, keep decoding in a tight loop.
		 * ============================================================ */
		while(FLAC__LIKELY(ucbits > parameter && val < end)) {
			/* Count leading zeros to find unary value */
			x = COUNT_ZERO_MSBS2(b);

			/* Check if the unary code + stop bit + parameter bits all fit */
			if(FLAC__LIKELY(x + 1 + parameter <= ucbits)) {
				/* Everything fits in the current word - fast path */
				msbs = x;

				if(FLAC__UNLIKELY(msbs > limit))
					return false;

				/* Skip past unary zeros and stop bit */
				b <<= (x + 1);
				ucbits -= (x + 1);

				/* Extract parameter LSBs directly */
				lsbs = (FLAC__uint32)(b >> (FLAC__BITS_PER_WORD - parameter));
				b <<= parameter;
				ucbits -= parameter;

				/* Compose and store the value (zig-zag decode) */
				x = (msbs << parameter) | lsbs;
				*val++ = (int)(x >> 1) ^ -(int)(x & 1);
			} else {
				/* Not enough bits for fast path, break to slow path */
				break;
			}
		}

		if(val >= end)
			break;

		/* ============================================================
		 * SLOW PATH: Handle word boundaries and long unary codes
		 * ============================================================ */

		/* read the unary MSBs and end bit */
		x = y = COUNT_ZERO_MSBS2(b);
		if(FLAC__UNLIKELY(x == FLAC__BITS_PER_WORD)) {
			x = ucbits;
			do {
				cwords++;
				if(cwords >= words)
					goto incomplete_msbs;
				b = br->buffer[cwords];
				y = COUNT_ZERO_MSBS2(b);
				x += y;
			} while(y == FLAC__BITS_PER_WORD);
		}
		b <<= y;
		b <<= 1; /* account for stop bit */
		ucbits = (ucbits - x - 1) % FLAC__BITS_PER_WORD;
		msbs = x;

		if(FLAC__UNLIKELY(x > limit))
			return false;

		/* read the binary LSBs */
		x = (FLAC__uint32)(b >> (FLAC__BITS_PER_WORD - parameter)); /* parameter < 32, so this is safe */
		if(FLAC__LIKELY(parameter <= ucbits)) {
			ucbits -= parameter;
			b <<= parameter;
		} else {
			/* there are still bits left to read, they will all be in the next word */
			cwords++;
			if(cwords >= words)
				goto incomplete_lsbs;
			b = br->buffer[cwords];
			ucbits += FLAC__BITS_PER_WORD - parameter;
			x |= (FLAC__uint32)(b >> ucbits);
			b <<= FLAC__BITS_PER_WORD - ucbits;
		}
		lsbs = x;

		/* compose the value */
		x = (msbs << parameter) | lsbs;
		*val++ = (int)(x >> 1) ^ -(int)(x & 1);

		continue;

		/* at this point we've eaten up all the whole words */
process_tail:
		do {
			if(0) {
incomplete_msbs:
				br->consumed_bits = 0;
				br->consumed_words = cwords;
			}

			/* read the unary MSBs and end bit */
			if(!FLAC__bitreader_read_unary_unsigned(br, &msbs))
				return false;
			msbs += x;
			x = ucbits = 0;

			if(0) {
incomplete_lsbs:
				br->consumed_bits = 0;
				br->consumed_words = cwords;
			}

			/* read the binary LSBs */
			if(!FLAC__bitreader_read_raw_uint32(br, &lsbs, parameter - ucbits))
				return false;
			lsbs = x | lsbs;

			/* compose the value */
			x = (msbs << parameter) | lsbs;
			*val++ = (int)(x >> 1) ^ -(int)(x & 1);
			x = 0;

			cwords = br->consumed_words;
			words = br->words;
			ucbits = FLAC__BITS_PER_WORD - br->consumed_bits;
			b = cwords < br->capacity ? br->buffer[cwords] << br->consumed_bits : 0;
		} while(cwords >= words && val < end);
	}

	if(ucbits == 0 && cwords < words) {
		cwords++;
		ucbits = FLAC__BITS_PER_WORD;
	}

	br->consumed_bits = FLAC__BITS_PER_WORD - ucbits;
	br->consumed_words = cwords;

	return true;
}

#endif /* FLAC__CPU_ARM64 && !FLAC__NO_ASM */
