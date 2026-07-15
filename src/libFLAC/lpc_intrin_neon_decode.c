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

/*
 * NEON-optimized LPC restore_signal for ARM64 FLAC decoder.
 *
 * The LPC decoder computes:
 *   data[i] = residual[i] + (sum(qlp_coeff[j] * data[i-j-1], j=0..order-1) >> lp_quantization)
 *
 * The outer loop is inherently serial (data[i] depends on data[i-1], etc.).
 * However, the inner dot product can be vectorized using NEON:
 *   - vmull_s32:  multiply 2 int32 pairs -> 2 int64 results (1 instruction)
 *   - vmlal_s32:  multiply-accumulate 2 int32 pairs into int64x2 (1 instruction)
 *   - vaddvq_s64: horizontal add of int64x2 (1 instruction, ARM64 only)
 *
 * This processes 2 coefficient-data multiplications per NEON instruction,
 * equivalent to what SSE4.1 does with _mm_mul_epi32 on x64.
 */

#include "private/cpu.h"

#ifndef FLAC__INTEGER_ONLY_LIBRARY
#ifndef FLAC__NO_ASM
#if defined FLAC__CPU_ARM64 && FLAC__HAS_A64NEONINTRIN

#include "private/lpc.h"
#include "FLAC/assert.h"
#include "FLAC/format.h"
#include "private/macros.h"
#include <arm_neon.h>

/*
 * Helper: compute dot product of qlp_coeff[0..order-1] with data[i-1..i-order]
 * using NEON vmull/vmlal to process 2 pairs per instruction.
 * Returns the 64-bit sum.
 */
static inline FLAC__int64 neon_lpc_dotprod(
    const FLAC__int32 * const data_ptr, /* points to data[i-1] (most recent) */
    const FLAC__int32 * const coeff,
    uint32_t order)
{
    int64x2_t acc = vdupq_n_s64(0);
    uint32_t j = 0;

    /* Process 4 pairs (4 coefficients) per iteration using 2 vmlal_s32 */
    for (; j + 3 < order; j += 4) {
        /* Load 2 coefficients: coeff[j], coeff[j+1] */
        int32x2_t c01 = vld1_s32(&coeff[j]);
        /* Load corresponding data: data[i-j-1], data[i-j-2] */
        /* data_ptr[0] = data[i-1], data_ptr[-1] = data[i-2], etc. */
        int32x2_t d01 = vld1_s32(&data_ptr[-j - 1]);  /* [data[i-j-2], data[i-j-1]] */
        d01 = vrev64_s32(d01);                          /* [data[i-j-1], data[i-j-2]] */
        acc = vmlal_s32(acc, c01, d01);

        /* Load 2 more coefficients: coeff[j+2], coeff[j+3] */
        int32x2_t c23 = vld1_s32(&coeff[j + 2]);
        int32x2_t d23 = vld1_s32(&data_ptr[-j - 3]);  /* [data[i-j-4], data[i-j-3]] */
        d23 = vrev64_s32(d23);                          /* [data[i-j-3], data[i-j-4]] */
        acc = vmlal_s32(acc, c23, d23);
    }

    /* Process 2 pairs per iteration */
    for (; j + 1 < order; j += 2) {
        int32x2_t c = vld1_s32(&coeff[j]);
        int32x2_t d = vld1_s32(&data_ptr[-j - 1]);
        d = vrev64_s32(d);
        acc = vmlal_s32(acc, c, d);
    }

    /* Horizontal add: sum both 64-bit lanes */
    FLAC__int64 sum = vaddvq_s64(acc);

    /* Handle remaining odd coefficient.
     * data_ptr points to data[i-1], so data_ptr[-j] = data[i-1-j] = data[i-(j+1)]
     * which is what coeff[j] should multiply. */
    if (j < order) {
        sum += (FLAC__int64)coeff[j] * data_ptr[-j];
    }

    return sum;
}

/*
 * NEON-optimized LPC restore signal (32-bit accumulator path).
 * Used when bps + order <= 32 (no overflow risk with 32-bit intermediate).
 *
 * Uses NEON vmull/vmlal to vectorize the inner dot product,
 * processing 2 coefficient-data pairs per instruction.
 */
void FLAC__lpc_restore_signal_intrin_neon(
    const FLAC__int32 *residual,
    uint32_t data_len,
    const FLAC__int32 *qlp_coeff,
    uint32_t order,
    int lp_quantization,
    FLAC__int32 *data)
{
    int i;
    FLAC__ASSERT(order > 0);
    FLAC__ASSERT(order <= 32);

    for (i = 0; i < (int)data_len; i++) {
        FLAC__int64 sum = neon_lpc_dotprod(data + i - 1, qlp_coeff, order);
        data[i] = residual[i] + (FLAC__int32)(sum >> lp_quantization);
    }
}

/*
 * NEON-optimized LPC restore signal (64-bit accumulator / wide path).
 * Used when bps + order > 32 to avoid overflow.
 *
 * The NEON vmlal_s32 instruction naturally accumulates into 64-bit,
 * so this is the same implementation — the 64-bit accumulator is inherent.
 */
void FLAC__lpc_restore_signal_wide_intrin_neon(
    const FLAC__int32 *residual,
    uint32_t data_len,
    const FLAC__int32 *qlp_coeff,
    uint32_t order,
    int lp_quantization,
    FLAC__int32 *data)
{
    int i;
    FLAC__ASSERT(order > 0);
    FLAC__ASSERT(order <= 32);

    for (i = 0; i < (int)data_len; i++) {
        FLAC__int64 sum = neon_lpc_dotprod(data + i - 1, qlp_coeff, order);
        data[i] = residual[i] + (FLAC__int32)(sum >> lp_quantization);
    }
}

/* When building with MSVC, lpc_intrin_neon.c is excluded because it uses
 * GCC-only compound literal NEON vector initialization syntax.
 * Provide fallback implementations for the encoder NEON functions that
 * stream_encoder.c references. These call the generic C implementations. */
#ifdef _MSC_VER

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_8(const FLAC__real data[], uint32_t data_len, uint32_t lag, double autoc[])
{
    FLAC__lpc_compute_autocorrelation(data, data_len, lag, autoc);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_10(const FLAC__real data[], uint32_t data_len, uint32_t lag, double autoc[])
{
    FLAC__lpc_compute_autocorrelation(data, data_len, lag, autoc);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_14(const FLAC__real data[], uint32_t data_len, uint32_t lag, double autoc[])
{
    FLAC__lpc_compute_autocorrelation(data, data_len, lag, autoc);
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
    FLAC__lpc_compute_residual_from_qlp_coefficients(data, data_len, qlp_coeff, order, lp_quantization, residual);
}

void FLAC__lpc_compute_residual_from_qlp_coefficients_wide_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
    FLAC__lpc_compute_residual_from_qlp_coefficients_wide(data, data_len, qlp_coeff, order, lp_quantization, residual);
}

#endif /* _MSC_VER */

#endif /* FLAC__CPU_ARM64 && FLAC__HAS_A64NEONINTRIN */
#endif /* FLAC__NO_ASM */
#endif /* FLAC__INTEGER_ONLY_LIBRARY */