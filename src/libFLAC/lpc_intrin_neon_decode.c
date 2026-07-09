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

/* Optimized LPC restore_signal for ARM64 FLAC decoder using FFmpeg's
 * 2-sample-per-iteration technique */

#include "private/cpu.h"

#ifndef FLAC__INTEGER_ONLY_LIBRARY
#ifndef FLAC__NO_ASM
#if defined FLAC__CPU_ARM64 && FLAC__HAS_NEONINTRIN

#include "private/lpc.h"
#include "FLAC/assert.h"
#include "FLAC/format.h"
#include "private/macros.h"
#include <arm_neon.h>

/*
 * FFmpeg-style 2-sample-per-iteration LPC restore signal.
 *
 * FLAC calling convention for restore_signal:
 *   residual[0..data_len-1] = residual samples
 *   data[-order..-1] = warmup/history samples (already filled by caller)
 *   data[0..data_len-1] = output (to be computed)
 *   qlp_coeff[0..order-1] = quantized LP coefficients
 *     where coeff[0] multiplies data[i-1], coeff[1] multiplies data[i-2], etc.
 *
 * The 2-sample technique computes data[i] and data[i+1] together:
 *   For data[i]:   sum0 = coeff[0]*data[i-1] + coeff[1]*data[i-2] + ...
 *   For data[i+1]: sum1 = coeff[0]*data[i]   + coeff[1]*data[i-1] + ...
 * 
 * Since data[i] = residual[i] + (sum0 >> qlevel), we can compute sum1
 * by reusing all terms shifted by one position, plus coeff[0]*data[i].
 */
void FLAC__lpc_restore_signal_intrin_neon(const FLAC__int32 *residual, uint32_t data_len, const FLAC__int32 *qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 *data)
{
	int i;
	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	/* Process 2 samples per iteration using FFmpeg's interleaved technique.
	 * We use a pointer 'history' that starts at data[-order] so that
	 * history[0] = data[i - order], history[order-1] = data[i-1] */
	for (i = 0; i < (int)data_len - 1; i += 2) {
		FLAC__int32 s = data[i - (int)order]; /* oldest history sample for data[i] */
		FLAC__uint32 c = qlp_coeff[order - 1];
		FLAC__int32 s0 = 0, s1 = 0;
		int j;

		for (j = (int)order - 2; j >= 0; j--) {
			s0 += c * s;
			s = data[i - 1 - j]; /* next history sample */
			s1 += c * s;
			c = qlp_coeff[j];
		}
		/* Last coefficient: coeff[0] * data[i-1] */
		s0 += c * s;
		/* Compute data[i] */
		data[i] = residual[i] + (s0 >> lp_quantization);
		/* For data[i+1], the last term uses the just-computed data[i] */
		s1 += c * data[i];
		data[i + 1] = residual[i + 1] + (s1 >> lp_quantization);
	}
	/* Handle the last sample if data_len is odd */
	if (i < (int)data_len) {
		FLAC__int32 sum = 0;
		int j;
		for (j = 0; j < (int)order; j++)
			sum += qlp_coeff[j] * data[i - j - 1];
		data[i] = residual[i] + (sum >> lp_quantization);
	}
}

/* Wide (64-bit accumulator) version */
void FLAC__lpc_restore_signal_wide_intrin_neon(const FLAC__int32 *residual, uint32_t data_len, const FLAC__int32 *qlp_coeff, uint32_t order, int lp_quantization, FLAC__int32 *data)
{
	int i;
	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

	for (i = 0; i < (int)data_len - 1; i += 2) {
		FLAC__int32 s = data[i - (int)order];
		FLAC__int32 c = qlp_coeff[order - 1];
		int64_t s0 = 0, s1 = 0;
		int j;

		for (j = (int)order - 2; j >= 0; j--) {
			s0 += (int64_t)c * s;
			s = data[i - 1 - j];
			s1 += (int64_t)c * s;
			c = qlp_coeff[j];
		}
		s0 += (int64_t)c * s;
		data[i] = residual[i] + (FLAC__int32)(s0 >> lp_quantization);
		s1 += (int64_t)c * data[i];
		data[i + 1] = residual[i + 1] + (FLAC__int32)(s1 >> lp_quantization);
	}
	if (i < (int)data_len) {
		int64_t sum = 0;
		int j;
		for (j = 0; j < (int)order; j++)
			sum += (int64_t)qlp_coeff[j] * (int64_t)data[i - j - 1];
		data[i] = residual[i] + (FLAC__int32)(sum >> lp_quantization);
	}
}

#endif /* FLAC__CPU_ARM64 && FLAC__HAS_NEONINTRIN */
#endif /* FLAC__NO_ASM */
#endif /* FLAC__INTEGER_ONLY_LIBRARY */