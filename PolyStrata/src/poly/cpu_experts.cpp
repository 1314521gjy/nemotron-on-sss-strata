// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#include "relu2_down_t.hpp"
#include "threads.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
// R36: 32 -> 64 for the up rows (the row kernels below decode an expert's weights once for several uses, so longer
// runs per task pay; the arithmetic is unchanged).  R41: STRATA_POLY_CPU_ROWS=96|128 gives a thread a longer
// contiguous range, so the hardware prefetcher streams the weights; the default stays 64.
int rows_per_task() {
    static const int rows = [] {
        const char* e = std::getenv("STRATA_POLY_CPU_ROWS");
        const int r = e ? std::atoi(e) : 64;
        return r >= 16 && r <= 512 ? r : 64;
    }();
    return rows;
}
constexpr int kRowsPerDownTask = 16;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i)
        th_.emplace_back([this] {
            affinity::pin_worker_thread();   // --host-core/--aux-cpus: no-op when no plan was made
            loop();
        });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

// An MXFP4 value is kvalues_mxfp4[nibble] times the block's power of two halved; Q5_0's is (q - 16) times its half.
bool mxfp4_block_to_q5_0(const uint8_t* src, uint8_t* dst) {
    constexpr int kLow = -24, kHigh = 15;   // the powers of two a half holds, subnormals included
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    static const auto half = [] {
        std::array<uint16_t, kHigh - kLow + 1> h{};
        for (int p = kLow; p <= kHigh; ++p) h[(size_t) (p - kLow)] = ggml_fp32_to_fp16(std::ldexp(1.0f, p));
        return h;
    }();
    const int p = (int) src[0] - 128;
    if (p < kLow || p > kHigh) return false;
    std::memcpy(dst, &half[(size_t) (p - kLow)], 2);
    uint32_t qh = 0;
    for (int j = 0; j < 16; ++j) {   // nibble j's low half is value j, its high half value j + 16, in both types
        const int q0 = kv[src[1 + j] & 15] + 16, q1 = kv[src[1 + j] >> 4] + 16;
        dst[6 + j] = (uint8_t) ((q0 & 15) | (q1 & 15) << 4);
        qh |= (uint32_t) (q0 >> 4) << j | (uint32_t) (q1 >> 4) << (j + 16);
    }
    std::memcpy(dst + 2, &qh, 4);
    return true;
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd,
                 int n_ff, const float* x, float limit, bool gelu, float* out, int n_out,
                 const std::function<void()>* meanwhile, double* gate_up_ms, double* quant_ms, double* down_ms,
                 Relu2Probe* probe, Relu2Active* relu2) {
    // R49's ledger: the three phases of the CPU expert pool, so its milliseconds can be read apart (the teacher's
    // q38_perf pool line).  The timing is asked for only by the ledger print; nothing changes without it.
    const auto now_ms = [] { return std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t_phase = now_ms();
    init_once();
    const int out_w = n_out > 0 ? n_out : n_embd;   // a latent MoE's output row is narrower than the model's
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    // R36: the row kernels in strata_kernels_cpu (kq256_rows for the Q4_K up rows) on the non-gated latent experts:
    // one decode of an expert's weights per run of its uses instead of one per use.  The kernel path is bit-exact
    // against ggml's per-token dots (src/poly/cpu_kernel_parity.cpp), but R36 measured no decode gain (B1/B0 = -1.1%,
    // the window's expert runs hold ~1.2 uses), so it stays OFF by default: the release path is the one below it,
    // byte-identical to the last build.  STRATA_POLY_CPU_KERNELS=1 turns it on for an experiment.
    static const bool kernels = [] {
        const char* e = std::getenv("STRATA_POLY_CPU_KERNELS");
        return e != nullptr && std::atoi(e) != 0 && strata::kernels::cpu::cpu_avx2_ok();
    }();
    // R41: memory-level parallelism for the CPU expert leg.  The weights are streamed from DRAM at ~32 GB/s, and the
    // per-row dots expose that latency one row at a time; prefetching the next rows and keeping four rows in flight
    // gives a thread enough outstanding loads to cover it.  Only the order the loads are issued in changes: every
    // value is still the same vec_dot's, so the ids stay bit-identical.  Off by default: STRATA_POLY_MLP=1 turns it
    // on, so the release path (no variable set) is the one the engine gave before it existed.
    static const bool mlp = [] {
        const char* e = std::getenv("STRATA_POLY_MLP");
        return e != nullptr && std::atoi(e) != 0;
    }();
    // Only the Q4_K up rows: kq256_rows is bit-exact against ggml's per-token dot here (poly_cpu_kernel_parity).
    // The IQ4_NL down rows have an AVX-2/VNNI row kernel too, but it is NOT bit-exact (it rounds differently alone
    // than in a group - native_expert.cpp's #152 note; the tree's own parity holds it to rel <= 1e-5 against ggml),
    // so the down rows keep ggml's vec_dot and the ids stay bit-identical.
    const bool fast = kernels && gu_type == (int) GGML_TYPE_Q4_K && n_embd % 256 == 0;
    if (fast) {
        static std::once_flag said;
        std::call_once(said, [] {
            std::fprintf(stderr, "strata-poly: CPU expert row kernels: Q4_K up rows (bit-exact); down rows stay "
                                 "ggml's vec_dot (the IQ4_NL row kernel is not bit-exact)\n");
        });
    }

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    // STRATA_CUT_HANDOFF: for a few rows the conversion is cheaper on this thread than the worker pass' handshake,
    // and the values are the same function of the same row (each row's bytes are this call's; the tasks below only
    // read what is written here, and each row is converted exactly once).  Off by default (=1 turns it on); with it
    // off the workers get the whole rows pass as they always did, byte for byte.
    static const bool cut_handoff = [] {
        const char* e = std::getenv("STRATA_CUT_HANDOFF");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (cut_handoff && rows <= 32) {
        for (int t = 0; t < rows; ++t) act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd);
    } else {
        workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    }

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    // The key is the expert's own up matrix, not its gate: a non-gated expert (Nemotron) has a null gate, so keying
    // on the gate shares one key across every use and collapses the layer into a single run that then applies the
    // first expert's up/down to all of them (t2, P3-20261016\t2-cellprobe-20261016.md).
    const auto gkey = [&](int i) -> const uint8_t* {
        const CpuExpert& x = experts[i];
        return x.gate ? x.gate : x.up;
    };
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return gkey(a) < gkey(b); });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || gkey(by[(size_t) i]) != gkey(by[(size_t) i - 1])) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int gu_tasks = (n_ff + rows_per_task() - 1) / rows_per_task();
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    t_phase = now_ms();
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[byp[i0]];
        const bool gated = e.gate != nullptr;   // a latent MoE's experts have no gate
        const int r0 = (t % gu_tasks) * rows_per_task(), r1 = std::min(n_ff, r0 + rows_per_task());
        if (fast && !gated) {
            // the run's up rows in one multi-token kernel: the expert's weights decode once for every use of it
            const int nu = i1 - i0;
            static thread_local std::vector<const void*> aptr;
            static thread_local std::vector<float*> fptr;
            aptr.resize((size_t) nu);
            fptr.resize((size_t) nu);
            for (int i = 0; i < nu; ++i) {
                const int j = byp[i0 + i];
                aptr[(size_t) i] = actp + (size_t) experts[j].row * a_bytes;
                fptr[(size_t) i] = hp + (size_t) j * n_ff;
            }
            strata::kernels::cpu::kq256_rows((int) gu_type, e.up, gu_row, n_embd, aptr.data(), nu, fptr.data(), r0, r1);
            for (int i = 0; i < nu; ++i) {   // no gate: relu2 of up (llama.cpp's LLM_FFN_RELU_SQR), and no clamp
                float* row = fptr[(size_t) i];
                for (int r = r0; r < r1; ++r) {
                    const float v = row[r] > 0.f ? row[r] : 0.f;
                    row[r] = v * v;
                }
            }
            return;
        }
        const auto one_up_row = [&](int r) {
            const uint8_t* gate = gated ? e.gate + (size_t) r * gu_row : nullptr;
            const uint8_t* up = e.up + (size_t) r * gu_row;
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                float g = 0.f, u = 0.f;
                if (gated) tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                tg->vec_dot(n_embd, &u, 0, up, 0, a, 0, 1);
                if (!gated) {   // no gate: relu2 of up (llama.cpp's LLM_FFN_RELU_SQR), and no clamp
                    const float v = u > 0.f ? u : 0.f;
                    hp[(size_t) j * n_ff + r] = v * v;
                    continue;
                }
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        };
        if (mlp) {
            // four rows in flight: the next block's weights are brought in while this one is dotted, so a thread has
            // several rows' loads outstanding instead of one row's (the values are the same vec_dots)
            constexpr int kAhead = 4;
            for (int r = r0; r < r1; r += kAhead) {
                const int re = std::min(r + kAhead, r1);
                for (int q = r + kAhead; q < std::min(r + 2 * kAhead, r1); ++q) {
                    _mm_prefetch((const char*) (e.up + (size_t) q * gu_row), _MM_HINT_T0);
                    if (gated) _mm_prefetch((const char*) (e.gate + (size_t) q * gu_row), _MM_HINT_T0);
                }
                for (int q = r; q < re; ++q) one_up_row(q);
            }
            return;
        }
        for (int r = r0; r < r1; ++r) one_up_row(r);
    });
    if (gate_up_ms) *gate_up_ms += now_ms() - t_phase;
    // R51: the ReLU^2 sparsity probe (read only; asked for by STRATA_PROBE_RELU2=1).  h[] now holds every use's
    // activated up values: a zero neuron, a 32-value IQ4_NL block that is all zero (the down dot could skip its
    // weight block), and a 64-byte weight line (~113.8 values: 3.56 blocks of 18 bytes) that no nonzero block
    // touches (the line could be skipped whole).  Nothing is changed; the counts are printed and that is all.
    if (probe) {
        const int nb = n_ff / 32;
        const int nline = (nb * 18 + 63) / 64;
        for (int j = 0; j < n; ++j) {
            const float* row = hp + (size_t) j * n_ff;
            for (int r = 0; r < n_ff; ++r) {
                ++probe->values;
                if (row[r] == 0.f) ++probe->zeros;
            }
            for (int b = 0; b < nb; ++b) {
                ++probe->blocks;
                bool all0 = true;
                for (int r = b * 32; r < b * 32 + 32 && all0; ++r) all0 = row[r] == 0.f;
                if (all0) ++probe->blocks_zero;
            }
            for (int l = 0; l < nline; ++l) {
                const int b0 = (int) (((long long) l * 64) / 18);
                const int b1 = (int) (((long long) (l + 1) * 64 - 1) / 18);
                if (b0 >= nb) break;
                ++probe->lines;
                bool all0 = true;
                for (int b = b0; b <= b1 && b < nb && all0; ++b)
                    for (int r = b * 32; r < b * 32 + 32 && all0; ++r) all0 = row[r] == 0.f;
                if (all0) ++probe->lines_zero;
            }
        }
    }
    t_phase = now_ms();
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
    if (quant_ms) *quant_ms += now_ms() - t_phase;
    // R56 (Stage B, STRATA_RELU2_DOWN=1): the live-neuron index and the sidecar bytes a Stage C kernel would read.
    // Read only: h and hq are the arrays this path already made, and nothing below changes.  This scan sits between
    // the quant and down phase clocks, so the ledger's down GB/s keeps being the unchanged kernel's number.
    if (relu2 && td->vec_dot_type == (int) GGML_TYPE_Q8_0) {
        relu2->idx.clear();
        relu2->off.assign(1, 0);
        relu2->active = relu2->active_q8 = relu2->groups = relu2->bytes = relu2->uses = 0;
        relu2->n_ff = n_ff;
        const int nb = n_ff / 32;
        const size_t qb = h_bytes / (size_t) nb;   // Q8_0: 34 B a block, 32 int8 at +2
        for (int j = 0; j < n; ++j) {
            const float* row = hp + (size_t) j * n_ff;
            const int8_t* q = (const int8_t*) (hqp + (size_t) j * h_bytes);
            long long act = 0;
            for (int i = 0; i < n_ff; ++i)
                if (row[i] != 0.f) { relu2->idx.push_back(i); ++act; }
            long long q8n = 0, gn = 0;
            for (int b = 0; b < nb; ++b) {
                const int8_t* qs = q + (size_t) b * qb + 2;
                bool any = false;
                for (int i = 0; i < 32; ++i)
                    if (qs[i] != 0) { any = true; ++q8n; }
                if (any) ++gn;
            }
            relu2->active += act;
            relu2->active_q8 += q8n;
            relu2->groups += gn;
            relu2->bytes += q8n * 512 + gn * 2048;
            relu2->off.push_back((int) relu2->idx.size());
            ++relu2->uses;
        }
        if (relu2->sidecar) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) {
                const double u = (double) std::max<long long>(relu2->uses, 1);
                std::fprintf(stderr,
                             "relu2-down: sidecar on, active=%.0f/%d, bytes_read=%.1f KB (vs 1548 KB), uses=%lld, "
                             "q8_active=%.0f, groups=%.1f/84\n",
                             (double) relu2->active / u, n_ff, (double) relu2->bytes / u / 1000.0, relu2->uses,
                             (double) relu2->active_q8 / u, (double) relu2->groups / u);
            }
        }
    }
    t_phase = now_ms();

    // R57 (Stage C, STRATA_RELU2_DOWN=1): the neuron-major down kernel.  It is bit-exact with the vec_dot path
    // below by construction (relu2_down_t.hpp reproduces ggml's eight int32 lanes per block, even/odd float
    // accumulators, FMA and hsum tree; a skipped neuron has hq == 0 and so adds exactly 0 to an int32 sum).  It
    // reads only this use's live neurons from the expert's sidecar block.  Falls back when a use has no sidecar
    // block or the geometry is not the packed one (2688 neurons / 1024 rows).
    bool downT = relu2 != nullptr && td->vec_dot_type == (int) GGML_TYPE_Q8_0 && n_ff == relu2t::kIn &&
                 out_w == relu2t::kOut && relu2->uses == n;
    for (int j = 0; downT && j < n; ++j) downT = experts[j].downT != nullptr;
    if (downT) {
        // NOTE: sd is this (the calling) thread's buffer; the workers get a pointer to it, not the thread_local
        // variable - a lambda body that named sd would resolve to each worker's own empty instance (R57 crash 1).
        static thread_local std::vector<float> sd;   // every use's 1024 down outputs
        sd.assign((size_t) n * out_w, 0.f);
        float* const sdp = sd.data();
        workers.run(n * relu2t::kChunks, [&, sdp](int t) {
            const int i = t / relu2t::kChunks, c = t % relu2t::kChunks;
            relu2t::down_chunk(experts[i].downT, hqp + (size_t) i * h_bytes, relu2->idx.data() + relu2->off[i],
                               relu2->off[i + 1] - relu2->off[i], c, sdp + (size_t) i * out_w);
        });
        // a token's sum in the order its experts were given, exactly as the row path below sums it
        int j = 0;
        for (int row = 0; row < rows; ++row) {
            float* o = out + (size_t) row * out_w;
            std::fill(o, o + out_w, 0.f);
            for (; j < n && experts[j].row == row; ++j) {
                const float w = experts[j].weight;
                const float* s = sd.data() + (size_t) j * out_w;
                for (int r = 0; r < out_w; ++r) o[r] += w * s[r];
            }
        }
        static std::atomic<bool> said_t{false};
        if (!said_t.exchange(true)) {
            const double u = (double) std::max<long long>(relu2->uses, 1);
            std::fprintf(stderr,
                         "relu2-down: T kernel on, uses=%lld, live_avg=%.1f/%d, bytes/use=%.1f KB (vs %.1f KB) "
                         "[block-order gather, bit-exact]\n",
                         relu2->uses, (double) relu2->active / u, relu2t::kIn, (double) relu2->bytes / u / 1000.0,
                         1548.288);
        }
        if (down_ms) *down_ms += now_ms() - t_phase;
        return;
    }

    const int d_tasks = (out_w + kRowsPerDownTask - 1) / kRowsPerDownTask;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use
        one.resize((size_t) n);
        float* s = one.data();
        const int r0 = t * kRowsPerDownTask, r1 = std::min(out_w, r0 + kRowsPerDownTask);
        const auto one_down_row = [&](int r) {
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                for (int i = runp[u]; i < runp[u + 1]; ++i)
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * out_w + r] = acc;
            }
        };
        if (mlp) {
            // four rows in flight, the next block's rows prefetched for every expert run in this task: the down
            // weights of one row are 2.7 KB, four rows' 10.6 KB - enough loads in flight to cover DRAM latency
            // while the dots of the current block are computed (the values are the same vec_dots)
            constexpr int kAhead = 4;
            for (int r = r0; r < r1; r += kAhead) {
                const int re = std::min(r + kAhead, r1);
                for (int q = r + kAhead; q < std::min(r + 2 * kAhead, r1); ++q)
                    for (int u = 0; u < n_run; ++u)
                        _mm_prefetch((const char*) (experts[byp[runp[u]]].down + (size_t) q * d_row), _MM_HINT_T0);
                for (int q = r; q < re; ++q) one_down_row(q);
            }
            return;
        }
        for (int r = r0; r < r1; ++r) one_down_row(r);
    });
    if (down_ms) *down_ms += now_ms() - t_phase;
}

}  // namespace strata::poly
