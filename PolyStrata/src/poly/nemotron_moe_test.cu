// src/poly/nemotron_moe_test.cu - Nemotron's routed-expert block (NOT gated: up, relu2, down, all on the latent
// width) against the same arithmetic on the host, plus the negative controls.  usage: poly_nemotron_moe_test
// (exit 0: the block agrees and every control turned red)
//
// The reference for the math is llama.cpp nemotron-h.cpp:280-300: build_moe_ffn on inp_latent with a null gate and
// LLM_FFN_RELU_SQR (ggml_relu then ggml_sqr).  The shapes are Nemotron's: latent 1024, expert FF 2688.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::poly;

namespace {

constexpr int kLatent = 1024;   // Nemotron's moe_latent_size: the experts' input and output width
constexpr int kFF = 2688;       // expert_feed_forward_length

std::mt19937 g_rng(20261016);

void fill(std::vector<float>& v, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    for (float& x : v) x = nd(g_rng);
}

// The geometry gate.  A layer whose latent width does not match its experts' matrices throws; nothing degrades.
void validate(int n_in, int n_out, int n_ff, size_t wup, size_t wdown) {
    if (n_in <= 0 || n_out <= 0 || n_ff <= 0) throw std::invalid_argument("nemotron_moe: a width is not positive");
    if (wup != (size_t) n_ff * n_in)
        throw std::invalid_argument("nemotron_moe: ffn_up_exps is not [n_ff, latent]=[" + std::to_string(n_ff) + ", " +
                                    std::to_string(n_in) + "]");
    if (wdown != (size_t) n_out * n_ff)
        throw std::invalid_argument("nemotron_moe: ffn_down_exps is not [latent, n_ff]=[" + std::to_string(n_out) +
                                    ", " + std::to_string(n_ff) + "]");
}

// y = W_down @ act(W_up @ x).  act is relu2 when `relu2`, else the gated silu(g)*up a wrong (Qwen-shaped) layer
// would use - the negative control.  Doubles: the oracle the device result is checked against.
void host_expert(const std::vector<float>& wup, const std::vector<float>& wdown, const std::vector<float>& x,
                 std::vector<double>& y, int n_in, int n_out, int n_ff, bool relu2) {
    std::vector<double> up(n_ff);
    for (int i = 0; i < n_ff; ++i) {
        double s = 0.0;
        for (int k = 0; k < n_in; ++k) s += double(wup[(size_t) i * n_in + k]) * x[k];
        up[i] = s;
    }
    std::vector<double> h(n_ff);
    for (int i = 0; i < n_ff; ++i) {
        if (relu2) {
            const double v = up[i] > 0.0 ? up[i] : 0.0;
            h[i] = v * v;
        } else {
            h[i] = up[i] / (1.0 + std::exp(-up[i])) * up[i];   // the gated form: silu(gate)*up, gate = up
        }
    }
    y.assign(n_out, 0.0);
    for (int o = 0; o < n_out; ++o) {
        double s = 0.0;
        for (int i = 0; i < n_ff; ++i) s += double(wdown[(size_t) o * n_ff + i]) * h[i];
        y[o] = s;
    }
}

double worst_rel(const std::vector<float>& a, const std::vector<double>& b) {
    double w = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = std::fabs(double(a[i]) - b[i]);
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        w = std::fmax(w, d / (1.0 + std::fabs(b[i])));
    }
    return w;
}

template <typename T> T* to_dev(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d), v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// The device block: up through gemv_f32, the activation, down.  Both widths are the latent's.
std::vector<float> device_expert(const std::vector<float>& wup, const std::vector<float>& wdown,
                                 const std::vector<float>& x, int n_in, int n_out, int n_ff) {
    float* d_wup = to_dev(wup);
    float* d_wdown = to_dev(wdown);
    float* d_x = to_dev(x);
    float* d_up = nullptr;
    float* d_h = nullptr;
    float* d_y = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_up), (size_t) n_ff * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_h), (size_t) n_ff * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_y), (size_t) n_out * sizeof(float));
    gemv_f32(d_wup, d_x, d_up, n_in, n_ff, n_ff, 1, nullptr);
    relu2(d_up, d_h, (size_t) n_ff, nullptr);
    gemv_f32(d_wdown, d_h, d_y, n_ff, n_out, n_out, 1, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> y((size_t) n_out);
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_wup), cudaFree(d_wdown), cudaFree(d_x), cudaFree(d_up), cudaFree(d_h), cudaFree(d_y);
    return y;
}

}  // namespace

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s  sm_%d%d\n", p.name, p.major, p.minor);

    std::vector<float> wup((size_t) kFF * kLatent), wdown((size_t) kLatent * kFF), x(kLatent);
    fill(wup, 0.1f);
    fill(wdown, 0.1f);
    fill(x, 1.0f);

    int bad = 0;
    const double tol = 1e-4;

    // positive: the device relu2 block against the relu2 oracle
    std::vector<double> want;
    host_expert(wup, wdown, x, want, kLatent, kLatent, kFF, true);
    const std::vector<float> got = device_expert(wup, wdown, x, kLatent, kLatent, kFF);
    const double e_relu2 = worst_rel(got, want);

    // negative 1: the same device output against the GATED oracle it must NOT match - the gate/relu2 confusion
    std::vector<double> want_gated;
    host_expert(wup, wdown, x, want_gated, kLatent, kLatent, kFF, false);
    const double e_gated = worst_rel(got, want_gated);

    // negative 2: a wrong latent width must throw, not compute
    bool threw = false;
    std::string what;
    try {
        validate(4096, kLatent, kFF, wup.size(), wdown.size());   // the experts are 1024-wide, latent said 4096
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }

    const bool green = e_relu2 < tol;
    const bool red1 = e_gated > 100.0 * tol;
    const bool red2 = threw;
    std::printf("relu2 block: align %.3g (tol %.0e)  |  neg gated-oracle %.3g  |  neg latent-width throw %s (%s)\n",
                e_relu2, tol, e_gated, threw ? "yes" : "NO", what.c_str());
    std::printf("=>  %s  (neg1 %s, neg2 %s)\n", (green && red1 && red2) ? "GREEN" : "RED",
                red1 ? "red ok" : "BAD", red2 ? "red ok" : "BAD");
    bad += !(green && red1 && red2);

    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
