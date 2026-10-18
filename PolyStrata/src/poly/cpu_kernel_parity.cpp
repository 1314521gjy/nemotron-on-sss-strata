// src/poly/cpu_kernel_parity.cpp - R36's parity gate for the STRATA_POLY_CPU_KERNELS path: the existing
// AVX-2/AVX-VNNI row kernels of strata_kernels_cpu against ggml-cpu's per-token vec_dot, on random blocks of our
// latent-MoE expert geometry (up: Q4_K [n_ff, n_embd]; down: IQ4_NL [out_w, n_ff]) - the same shapes
// Nemotron's cpu_experts() feeds them.  Every value is compared bit for bit (memcmp of the float bits); a single
// mismatch exits non-zero, so the engine path is only enabled when this says OK.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_mismatch = 0;   // the Q4_K up rows: the hard gate (bit-exact, the rows the engine will use)
long long g_down_values = 0;
double g_down_max_rel = 0.0;   // the IQ4_NL down rows: reported, not used (the kernel is not bit-exact, #152)

uint32_t bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

void diff(const char* what, int t, int r, float got, float want) {
    ++g_mismatch;
    if (g_mismatch <= 10) {
        const uint32_t a = bits(got), b = bits(want);
        std::printf("  MISMATCH %s t=%d r=%d got=%.9g(0x%08x) want=%.9g(0x%08x)\n", what, t, r, got, a, want, b);
    }
}

void down_diff(float got, float want) {
    ++g_down_values;
    const double d = std::fabs((double) got - (double) want);
    const double w = std::fabs((double) want);
    const double rel = d / (w > 1e-12 ? w : 1e-12);
    if (rel > g_down_max_rel) g_down_max_rel = rel;
}

}  // namespace

int main() {
    // as cpu_experts.cpp's init_once(): ggml-cpu installs its per-type dot dispatch (and the activation quantizers)
    // in ggml_cpu_init(); before it the traits' vec_dot can be a stub that writes nothing.
    ggml_cpu_init();
    if (!strata::kernels::cpu::cpu_avx2_ok()) {
        std::printf("cpu_avx2_ok() = false: the row kernels must not be called on this CPU; nothing to compare\n");
        return 0;
    }
    // our Nemotron latent-MoE expert geometry
    const int n_embd = 1024;   // the experts' input width (latent)
    const int n_ff = 2688;     // the experts' hidden width
    const int out_w = 1024;    // the experts' output width (latent)
    const ggml_type gu_type = GGML_TYPE_Q4_K;
    const ggml_type d_type = GGML_TYPE_IQ4_NL;
    const ggml_type_traits_cpu* tg = ggml_get_type_traits_cpu(gu_type);
    const ggml_type_traits_cpu* td = ggml_get_type_traits_cpu(d_type);
    if (!tg || !td || !tg->vec_dot || !td->vec_dot || !tg->from_float || !td->from_float) {
        std::printf("ggml-cpu has no traits for Q4_K / IQ4_NL\n");
        return 2;
    }
    if (!strata::kernels::cpu::kq256_supported((int) gu_type)) {
        std::printf("kq256_supported(Q4_K) = false\n");
        return 2;
    }
    // the activation quantizers the engine's path uses: from_float of the type the vec_dot takes (Q8_K for Q4_K,
    // Q8_0 for IQ4_NL), not of the weights' type - cpu_experts.cpp: `traits(tg->vec_dot_type)->from_float`.
    const ggml_type_traits_cpu* ta = ggml_get_type_traits_cpu(tg->vec_dot_type);
    const ggml_type_traits_cpu* th = ggml_get_type_traits_cpu(td->vec_dot_type);
    if (!ta || !ta->from_float || !th || !th->from_float) {
        std::printf("ggml-cpu has no activation quantizer for %s / %s\n", ggml_type_name(tg->vec_dot_type),
                    ggml_type_name(td->vec_dot_type));
        return 2;
    }
    std::printf("activation types: up %s, down %s\n", ggml_type_name(tg->vec_dot_type),
                ggml_type_name(td->vec_dot_type));
    if (ggml_row_size(gu_type, n_embd) == 0 || ggml_row_size(d_type, n_ff) == 0 || n_embd % 256 != 0 ||
        n_ff % 32 != 0) {
        std::printf("geometry does not fit the kernels\n");
        return 2;
    }
    const size_t gu_row = ggml_row_size(gu_type, n_embd);
    const size_t d_row = ggml_row_size(d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);

    std::mt19937 rng(20261018u);
    std::normal_distribution<float> nd(0.f, 1.f);

    // random valid blocks: quantize random rows with ggml's own quantizers
    std::vector<float> fup((size_t) n_ff * n_embd);
    for (float& v : fup) v = nd(rng);
    std::vector<uint8_t> up((size_t) n_ff * gu_row);
    ggml_quantize_chunk(gu_type, fup.data(), up.data(), 0, n_ff, n_embd, nullptr);
    std::vector<float> fdown((size_t) out_w * n_ff);
    for (float& v : fdown) v = nd(rng);
    std::vector<uint8_t> down((size_t) out_w * d_row);
    ggml_quantize_chunk(d_type, fdown.data(), down.data(), 0, out_w, n_ff, nullptr);
    std::printf("blocks: Q4_K up %d rows x %zu B, IQ4_NL down %d rows x %zu B\n", n_ff, gu_row, out_w, d_row);

    // ---- up rows: kq256_rows (Q4_K, Q8_K activations) against ggml's vec_dot ----
    for (int nt : {1, 2, 4, 8}) {
        std::vector<std::vector<uint8_t>> act((size_t) nt, std::vector<uint8_t>(a_bytes));
        std::vector<const void*> actp((size_t) nt);
        for (int t = 0; t < nt; ++t) {
            std::vector<float> x((size_t) n_embd);
            for (float& v : x) v = nd(rng);
            ta->from_float(x.data(), act[(size_t) t].data(), n_embd);
            actp[(size_t) t] = act[(size_t) t].data();
        }
        for (const auto& span : {std::pair<int, int>{0, n_ff}, std::pair<int, int>{64, 192}}) {
            std::vector<float> got((size_t) nt * n_ff, 0.f), want((size_t) nt * n_ff, 0.f);
            std::vector<float*> outp((size_t) nt);
            for (int t = 0; t < nt; ++t) outp[(size_t) t] = got.data() + (size_t) t * n_ff;
            strata::kernels::cpu::kq256_rows((int) gu_type, up.data(), gu_row, n_embd, actp.data(), nt, outp.data(),
                                             span.first, span.second);
            for (int r = span.first; r < span.second; ++r)
                for (int t = 0; t < nt; ++t)
                    tg->vec_dot(n_embd, want.data() + (size_t) t * n_ff + r, 0, up.data() + (size_t) r * gu_row, 0,
                                actp[(size_t) t], 0, 1);
            for (int t = 0; t < nt; ++t)
                for (int r = span.first; r < span.second; ++r)
                    if (bits(got[(size_t) t * n_ff + r]) != bits(want[(size_t) t * n_ff + r]))
                        diff("Q4_K up", t, r, got[(size_t) t * n_ff + r], want[(size_t) t * n_ff + r]);
            std::printf("Q4_K up  nt=%d rows=[%d,%d): %d x %d values compared\n", nt, span.first, span.second, nt,
                        span.second - span.first);
        }
    }

    // ---- down rows: iq4nl256_down_rows (IQ4_NL, Q8_0 activations), every variant this CPU can run ----
    for (int nt : {1, 2, 4, 8}) {
        std::vector<std::vector<uint8_t>> hq((size_t) nt, std::vector<uint8_t>(h_bytes));
        std::vector<const void*> hqp((size_t) nt);
        for (int t = 0; t < nt; ++t) {
            std::vector<float> h((size_t) n_ff);
            for (float& v : h) v = nd(rng);
            th->from_float(h.data(), hq[(size_t) t].data(), n_ff);
            hqp[(size_t) t] = hq[(size_t) t].data();
        }
        std::vector<float> want((size_t) nt * out_w, 0.f);
        for (int r = 0; r < out_w; ++r)
            for (int t = 0; t < nt; ++t)
                td->vec_dot(n_ff, want.data() + (size_t) t * out_w + r, 0, down.data() + (size_t) r * d_row, 0,
                            hqp[(size_t) t], 0, 1);
        const int variants = strata::kernels::cpu::iq256_variants();
        for (int v = 0; v <= 2; ++v) {
            if (v != 0 && (variants & v) == 0) continue;
            std::vector<float> got((size_t) nt * out_w, 0.f);
            std::vector<float*> outp((size_t) nt);
            for (int t = 0; t < nt; ++t) outp[(size_t) t] = got.data() + (size_t) t * out_w;
            strata::kernels::cpu::iq4nl256_down_rows_v(v, down.data(), d_row, n_ff, hqp.data(), nt, outp.data(), 0,
                                                       out_w);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < out_w; ++r)
                    if (bits(got[(size_t) t * out_w + r]) != bits(want[(size_t) t * out_w + r]))
                        down_diff(got[(size_t) t * out_w + r], want[(size_t) t * out_w + r]);
            std::printf("IQ4_NL down nt=%d variant=0x%x: %d x %d values compared\n", nt, v, nt, out_w);
        }
    }

    if (g_mismatch) {
        std::printf("PARITY FAIL (Q4_K up, bit-exact is the gate): %d mismatches\n", g_mismatch);
        return 1;
    }
    std::printf("PARITY OK: Q4_K up rows bit-identical to ggml-cpu's vec_dot for every nt and row span\n");
    std::printf("IQ4_NL down row kernel: NOT bit-exact (%lld values differ, max rel %.3e) - the engine keeps ggml's "
                "vec_dot for the down rows\n", g_down_values, g_down_max_rel);
    return 0;
}
