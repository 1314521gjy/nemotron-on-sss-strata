// src/poly/relu2_down_t.hpp - R57 Stage C: the neuron-major ("T") IQ4_NL down kernel of the ReLU^2 sidecar path.
//
// It is bit-exact with ggml's AVX2 ggml_vec_dot_iq4_nl_q8_0 (ggml/src/ggml-cpu/arch/x86/quants.c:3920-3964) by
// construction, and it reads only the live neurons of an expert's sidecar block:
//   * per 32-value IQ4_NL block it rebuilds the SAME eight int32 lane sums ggml builds (lane l = the block's values
//     4l..4l+3; a skipped neuron has hq == 0 and so contributes exactly 0 to an int32 sum, whatever the order);
//   * then the SAME float structure: d = fp16(hq block scale) * fp16(weight block scale), one _mm256_fmadd_ps per
//     (block, row) into one of TWO lane-wise accumulators (even blocks -> accum1, odd -> accum2), and at the end
//     hsum_float_8(accum1 + accum2) with ggml's exact tree;
//   * the block order is ascending 0..83, so the float addition order matches ggml's pair loop.
// The 2,688-neuron value of a neuron is 512 B: byte b holds row 2b (low nibble) and row 2b+1 (high nibble); the
// 84 groups' fp16 scales are 2048 B a group (row-major: scales[g*2048 + r*2]).
//
// The rows are processed in chunks of 64 (16 chunks): each task is (one use, one chunk), which keeps every
// neuron's 32-byte slice for the chunk in one load and the int32/float accumulators in L1.  Chunking does not
// change any row's arithmetic: every row's accumulation is the same ascending block sequence either way.
//
// Header only, so the zero-GPU parity gate C:\sp3\stageC-parity.cpp compiles and compares THIS code against
// ggml's own vec_dot (the gate is what the ids door rests on).
#pragma once

#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace strata::poly::relu2t {

constexpr int kIn = 2688;                     // neurons a row (the down matrix's input width)
constexpr int kOut = 1024;                    // output rows (the latent width)
constexpr int kBlocks = kIn / 32;             // 84 IQ4_NL blocks a row
constexpr int kNeuronBytes = kOut / 2;        // 512 B: a neuron's 1024 four-bit values
constexpr int kBlockBytes = 18;               // IQ4_NL: 2 B fp16 scale + 16 B nibbles
constexpr int kGroupBytes = kOut * 2;         // 2048 B: one 32-neuron block's fp16 scales over 1024 rows
constexpr int kQ8BlockBytes = 34;             // Q8_0: fp16 d + 32 int8
constexpr int kChunk = 64;                    // output rows a task
constexpr int kChunks = kOut / kChunk;        // 16

// ggml's kvalues_iq4nl (ggml-common.h)
inline constexpr int8_t kKValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// fp16 -> fp32, exact (no rounding possible): the same bits ggml's F16C conversion gives.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            uint32_t mm = m;
            int sh = 0;
            while (!(mm & 0x400)) {
                mm <<= 1;
                ++sh;
            }
            bits = s | ((uint32_t) (127 - 14 - sh) << 23) | ((mm & 0x3FF) << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);   // inf/nan
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// ggml's hsum_float_8 (ggml/src/ggml-cpu/arch/x86/quants.c:43) - the exact tree, in that order.
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// One (use, chunk) task.  T: one expert's neuron-major block (values then scales).  hq: the use's Q8_0 row.
// live: ascending neuron indices with hq != 0; n_live: how many.  chunk: 0..kChunks-1 (rows chunk*64 .. +63).
// out: kOut floats; this writes out[chunk*64 .. +63].
inline void down_chunk(const uint8_t* T, const uint8_t* hq, const int32_t* live, int n_live, int chunk, float* out) {
    const uint8_t* const scales = T + (size_t) kIn * kNeuronBytes;
    const int r0 = chunk * kChunk;
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kKValues));
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    // lane-major accumulators: accum1[lane][row-of-chunk], accum2[lane][row-of-chunk]
    alignas(32) float accum1[8][kChunk];
    alignas(32) float accum2[8][kChunk];
    std::memset(accum1, 0, sizeof accum1);
    std::memset(accum2, 0, sizeof accum2);

    int li = 0;
    for (int k = 0; k < kBlocks; ++k) {
        // this block's eight int32 lane sums, lane-major over the chunk's 64 rows
        alignas(32) int32_t p[8][kChunk];
        std::memset(p, 0, sizeof p);
        while (li < n_live && (live[li] >> 5) < k) ++li;
        for (int q = li; q < n_live && (live[q] >> 5) == k; ++q) {
            const int n = live[q];
            const int lane = (n & 31) >> 2;
            const int8_t hv = (int8_t) hq[(size_t) k * kQ8BlockBytes + 2 + (n & 31)];
            const __m256i hv32 = _mm256_set1_epi32((int) hv);
            const uint8_t* v = T + (size_t) n * kNeuronBytes + (size_t) (r0 >> 1);   // 32 B: rows r0..r0+63
            const __m256i b = _mm256_loadu_si256((const __m256i*) v);
            const __m256i ev = _mm256_shuffle_epi8(lut, _mm256_and_si256(b, m4b));   // even rows (low nibbles)
            const __m256i od = _mm256_shuffle_epi8(
                lut, _mm256_and_si256(_mm256_srli_epi16(b, 4), m4b));                // odd rows (high nibbles)
            const __m256i rlo = _mm256_unpacklo_epi8(ev, od);   // rows 0..15 (lo128) and 32..47 (hi128)
            const __m256i rhi = _mm256_unpackhi_epi8(ev, od);   // rows 16..31 (lo128) and 48..63 (hi128)
            const __m128i rlo0 = _mm256_castsi256_si128(rlo), rlo1 = _mm256_extracti128_si256(rlo, 1);
            const __m128i rhi0 = _mm256_castsi256_si128(rhi), rhi1 = _mm256_extracti128_si256(rhi, 1);
            const __m256i v0 = _mm256_cvtepi8_epi32(rlo0);                               // rows 0..7
            const __m256i v1 = _mm256_cvtepi8_epi32(_mm_srli_si128(rlo0, 8));            // rows 8..15
            const __m256i v2 = _mm256_cvtepi8_epi32(rhi0);                               // rows 16..23
            const __m256i v3 = _mm256_cvtepi8_epi32(_mm_srli_si128(rhi0, 8));            // rows 24..31
            const __m256i v4 = _mm256_cvtepi8_epi32(rlo1);                               // rows 32..39
            const __m256i v5 = _mm256_cvtepi8_epi32(_mm_srli_si128(rlo1, 8));            // rows 40..47
            const __m256i v6 = _mm256_cvtepi8_epi32(rhi1);                               // rows 48..55
            const __m256i v7 = _mm256_cvtepi8_epi32(_mm_srli_si128(rhi1, 8));            // rows 56..63
            _mm256_store_si256((__m256i*) &p[lane][0],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][0]),
                                                _mm256_mullo_epi32(v0, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][8],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][8]),
                                                _mm256_mullo_epi32(v1, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][16],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][16]),
                                                _mm256_mullo_epi32(v2, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][24],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][24]),
                                                _mm256_mullo_epi32(v3, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][32],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][32]),
                                                _mm256_mullo_epi32(v4, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][40],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][40]),
                                                _mm256_mullo_epi32(v5, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][48],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][48]),
                                                _mm256_mullo_epi32(v6, hv32)));
            _mm256_store_si256((__m256i*) &p[lane][56],
                               _mm256_add_epi32(_mm256_load_si256((const __m256i*) &p[lane][56]),
                                                _mm256_mullo_epi32(v7, hv32)));
        }
        // d = fp16(hq block scale) * fp16(weight block scale) a row, in row order (the chunk's scales are 128 B)
        const float qd = fp16_to_fp32([&] {
            uint16_t h;
            std::memcpy(&h, hq + (size_t) k * kQ8BlockBytes, 2);
            return h;
        }());
        alignas(32) float d[kChunk];
        for (int i = 0; i < kChunk; i += 8) {
            uint16_t h[8];
            std::memcpy(h, scales + (size_t) k * kGroupBytes + (size_t) (r0 + i) * 2, 16);
            float w[8];
            for (int j = 0; j < 8; ++j) w[j] = fp16_to_fp32(h[j]);
            const __m256 wv = _mm256_loadu_ps(w);
            _mm256_store_ps(&d[i], _mm256_mul_ps(_mm256_set1_ps(qd), wv));
        }
        if ((k & 1) == 0) {
            for (int l = 0; l < 8; ++l)
                for (int i = 0; i < kChunk; i += 8)
                    _mm256_store_ps(&accum1[l][i],
                                    _mm256_fmadd_ps(_mm256_load_ps(&d[i]),
                                                    _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*) &p[l][i])),
                                                    _mm256_load_ps(&accum1[l][i])));
        } else {
            for (int l = 0; l < 8; ++l)
                for (int i = 0; i < kChunk; i += 8)
                    _mm256_store_ps(&accum2[l][i],
                                    _mm256_fmadd_ps(_mm256_load_ps(&d[i]),
                                                    _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*) &p[l][i])),
                                                    _mm256_load_ps(&accum2[l][i])));
        }
    }

    // sumf = hsum_float_8(accum1 + accum2), one row at a time, in ggml's tree
    for (int i = 0; i < kChunk; ++i) {
        float v[8];
        for (int l = 0; l < 8; ++l) v[l] = accum1[l][i] + accum2[l][i];
        const float r0v = v[0] + v[4], r1v = v[1] + v[5], r2v = v[2] + v[6], r3v = v[3] + v[7];
        out[r0 + i] = (r0v + r2v) + (r1v + r3v);
    }
}

}  // namespace strata::poly::relu2t
