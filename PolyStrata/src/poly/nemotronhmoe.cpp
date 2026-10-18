// src/poly/nemotronhmoe.cpp - the layers of Nemotron 3.5 Super VL 120B A12B.  The reference for the math is
// llama.cpp's nemotron-h graph (src/models/nemotron-h.cpp) and the ggml ops its Mamba-2 block builds.
//
// Scope: the Mamba-2 mixer and the attention layer are wired; the latent MoE layer is not (the engine's expert path
// is gated and n_embd-wide, Nemotron's is not) and load_layer throws a loud, named error for it rather than loading
// something the engine would compute wrongly.  See P3-20261016/READING-P3-status.md.
#include "nemotronhmoe.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;   // the attention's scores of the tokens it takes at a time

}  // namespace

const Names* NemotronHMoe::tensor_alias(const char*) const { return nullptr; }

// llama.cpp's nemotron-h reads attention.layer_norm_epsilon (nemotron-h.cpp:20-21); this engine's shared loader
// names that key attention.layer_norm_rms_epsilon.  The GGUF-measured key is the former.
const Names* NemotronHMoe::meta_alias(const char* key) const {
    static const Names kMeta[] = {
        {"attention.layer_norm_rms_epsilon", {"attention.layer_norm_epsilon", nullptr}},
    };
    for (const Names& n : kMeta)
        if (std::strcmp(n.name, key) == 0) return &n;
    return nullptr;
}

void NemotronHMoe::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_mamba_ = n_attn_ = n_moe_ = 0;
}

// A layer's kind from the tensors it has: the mixer's ssm_in, the MoE's router, or the attention's q.  The GGUF's
// attention.head_count_kv is 0 for the mixer and the MoE layers alike (llama.cpp separates them by n_ff), so it is
// only a cross-check here.
NemotronHMoe::Kind NemotronHMoe::kind_of(int il) const {
    if (tensor(il, "ssm_in.weight")) return kMamba;
    // The NextN/MTP block is the one block with both an attention sub-layer and a MoE sub-layer: llama.cpp's
    // nemotron-h creates attn_q and ffn_gate_inp for it (nemotron-h.cpp:142-181).  It must be seen here, or the
    // attention half is silently dropped and the block is loaded as a plain MoE layer (the R29 finding: 0-4%
    // acceptance because the draft block had no attention).
    if (tensor(il, "attn_q.weight") && tensor(il, "ffn_gate_inp.weight")) return kAttnMoe;
    if (tensor(il, "ffn_gate_inp.weight")) return kMoe;
    if (tensor(il, "attn_q.weight")) return kAttn;
    throw std::runtime_error("nemotron_h_moe: layer " + std::to_string(il) +
                             " has no ssm_in, ffn_gate_inp or attn_q: its kind is unknown");
}

void NemotronHMoe::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };

    // attention (llama.cpp nemotron-h.cpp:93-98)
    n_head_ = shape_i("attention.head_count");
    need(n_head_ > 0 && n_embd_ % n_head_ == 0 && n_head_ <= 256, "bad attention.head_count");
    head_dim_ = n_embd_ / n_head_;
    n_rot_ = head_dim_;
    need(head_dim_ % 2 == 0 && head_dim_ <= 256, "bad head size");
    // rope.freq_base is optional in the GGUF (llama.cpp defaults it to 10000); Nemotron's file leaves it out
    const MetaValue* freq = meta_in(*gg_, "rope.freq_base");
    theta_scale_ = std::pow(freq ? (float) freq->f : 10000.f, -2.f / (float) n_rot_);

    // Mamba-2 (llama.cpp nemotron-h.cpp:4-8)
    d_conv_ = shape_i("ssm.conv_kernel");
    d_inner_ = shape_i("ssm.inner_size");
    d_state_ = shape_i("ssm.state_size");
    ssm_heads_ = shape_i("ssm.time_step_rank");
    n_group_ = shape_i("ssm.group_count");
    need(d_conv_ >= 2 && d_conv_ <= 8, "bad ssm.conv_kernel");
    need(d_inner_ > 0 && ssm_heads_ > 0 && d_inner_ % ssm_heads_ == 0, "bad ssm.inner_size / time_step_rank");
    need(d_state_ > 0 && d_state_ <= 512, "bad ssm.state_size");
    need(n_group_ > 0 && d_inner_ % n_group_ == 0 && ssm_heads_ % n_group_ == 0, "ssm.group_count does not divide");
    channels_ = d_inner_ + 2 * n_group_ * d_state_;
    d_in_proj_ = 2 * d_inner_ + 2 * n_group_ * d_state_ + ssm_heads_;

    // the MoE's latent width (llama.cpp nemotron-h.cpp:28); the load path of the experts is deferred
    const MetaValue* latent = find("moe_latent_size");
    n_latent_ = latent ? (int) latent->u : 0;
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;

    // The layer table comes from the tensors present, NOT from attention.head_count_kv: that array is 0 for BOTH the
    // mixer and the MoE layers (llama.cpp separates them by n_ff, which the GGUF exposes only through the tensors).
    // The reader keeps only an array's first 64 items (gguf_reader.hpp:349) and Nemotron's array is 89 long, so the
    // key heads come from attn_k's own width; the array is a cross-check where the reader reaches.
    const MetaValue* kv = find("attention.head_count_kv");
    need(kv && kv->type == MetaType::ARRAY && kv->count >= (uint64_t) n_layer_,
         "attention.head_count_kv has no value for every layer");
    kv_heads_.assign((size_t) n_layer_, 0);
    int n_mamba = 0, n_attn = 0, n_moe = 0;   // the members are counted again by load_layer, after begin_pass resets them
    for (int il = 0; il < n_layer_; ++il) {
        const Kind k = kind_of(il);
        if (k == kMamba) { ++n_mamba; continue; }
        if (k == kMoe) { ++n_moe; continue; }
        ++n_attn;
        int h = (size_t) il < kv->items.size() ? (int) kv->items[(size_t) il].u : 0;
        if (h == 0) {   // past what the reader kept: the key heads are in attn_k's row count
            const TensorInfo* tk = tensor(il, "attn_k.weight");
            h = tk && tk->shape.size() == 2 ? (int) (tk->shape[1] / head_dim_) : 0;
        }
        need(h > 0 && n_head_ % h == 0, "an attention layer's key heads do not divide its query heads");
        need(n_kv_ == 0 || h == n_kv_, "the attention layers have different key heads");
        n_kv_ = h;
        kv_heads_[(size_t) il] = h;
    }
    need(n_attn > 0 && n_kv_ > 0, "no attention layer");
    // the latent width: a hard gate in the production path, and not only in the test.  Every expert matrix and both
    // latent projections are checked against it as they load (load_moe()'s exps() and upload()), so a wrong
    // moe_latent_size throws before any of them is read - not a silent wrong width.
    need(n_moe == 0 || n_latent_ > 0, "a MoE layer whose moe_latent_size is 0");
    std::fprintf(stderr, "strata-poly: nemotron_h_moe %d layers: %d mamba, %d attention, %d moe (latent %d)\n",
                 n_layer_, n_mamba, n_attn, n_moe, n_latent_);

    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void NemotronHMoe::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.kind = kind_of(il);
    // counted here, not in load_geometry: begin_pass() resets them before every pass, so the count that alloc() and
    // recurrent_floats() read is the one this pass left (load_geometry's own count was wiped by begin_pass - the
    // 8585-experts smoke caught it as an illegal access when the Mamba state turned out to be zero floats).
    // The MTP block counts on both sides: its attention cache and its routed experts exist.
    if (B.kind == kAttn || B.kind == kAttnMoe) ++n_attn_;
    if (B.kind == kMamba) ++n_mamba_;
    if (B.kind == kMoe || B.kind == kAttnMoe) ++n_moe_;
    L.fixed = true;

    if (B.kind == kAttn || B.kind == kAttnMoe) {
        // llama.cpp nemotron-h.cpp:91-98: q/k/v then the output; the q and k norms and the output bias are optional
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.q_norm = tensor(il, "attn_q_norm.weight") ? f32("attn_q_norm.weight", head_dim_) : Dev{};
        B.k_norm = tensor(il, "attn_k_norm.weight") ? f32("attn_k_norm.weight", head_dim_) : Dev{};
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        B.wo_b = tensor(il, "attn_output.bias") ? f32("attn_output.bias", n_embd_) : Dev{};
        if (B.kind == kAttn) return;
        // the MTP block, second half: its pre-FFN norm is a tensor of its own (only this block has it).  The engine's
        // draft_step() norms with Layer::ffn_norm before the experts, which was a null weight before this line.
        L.ffn_norm = f32("post_attention_norm.weight", n_embd_);
    }

    if (B.kind == kMamba) {
        // llama.cpp nemotron-h.cpp:74-90
        B.ssm_in = mat("ssm_in.weight", n_embd_, d_in_proj_);
        B.conv = f32("ssm_conv1d.weight", d_conv_, channels_);
        B.conv_b = tensor(il, "ssm_conv1d.bias") ? f32("ssm_conv1d.bias", channels_) : Dev{};
        B.dt_b = f32("ssm_dt.bias", ssm_heads_);
        B.ssm_a = f32("ssm_a", 1, ssm_heads_);        // no "weight" suffix
        B.ssm_d = f32("ssm_d", 1, ssm_heads_);
        B.ssm_norm = f32("ssm_norm.weight", d_inner_ / n_group_, n_group_);
        B.ssm_out = mat("ssm_out.weight", d_inner_, n_embd_);
        return;
    }

    // the latent MoE (llama.cpp nemotron-h.cpp:100-117, :270-311): the routed experts are not gated and eat the
    // latent width (Nemotron's 1024), with a latent projection once a layer and a shared expert on the model's
    // width.  load_moe() takes the shape; the layers keep the projections for moe_router()/moe_latent().
    const int latent = n_latent_ > 0 ? n_latent_ : n_embd_;
    B.latent_down = mat("ffn_latent_down.weight", n_embd_, latent);   // {n_embd, moe_n_embd}
    B.latent_up = mat("ffn_latent_up.weight", latent, n_embd_);       // {moe_n_embd, n_embd}
    load_moe(il, true, false, latent, latent);
    L.latent_down = B.latent_down;
    L.latent_up = B.latent_up;
    B.probs_b.assign((size_t) n_expert_, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (!b || b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + (b ? shape_of(*b) : "missing") +
                                 ", expected F32 [" + std::to_string(n_expert_) + "]");
    home(*b).payload(*b);
    std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
}

size_t NemotronHMoe::recurrent_floats() const {
    return size_t(n_mamba_) * (size_t(channels_) * (d_conv_ - 1) + size_t(d_inner_) * d_state_);
}

int NemotronHMoe::widest() const { return std::max(n_embd_, d_inner_); }

void NemotronHMoe::alloc_caches() {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.kind == kAttn || B.kind == kAttnMoe) {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        } else if (B.kind == kMamba) {
            B.conv_state = st;
            st += size_t(channels_) * (d_conv_ - 1);
            B.ssm_state = st;
            st += size_t(d_inner_) * d_state_;
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void NemotronHMoe::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * d_in_proj_);
    cv_ = fbuf(n * channels_);
    ssm_y_ = fbuf(n * d_inner_);
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

void NemotronHMoe::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (B.conv_state) {
        const size_t bytes = size_t(channels_) * (d_conv_ - 1) * sizeof(float);
        ck(cudaMemcpy(dst + (size_t) (B.conv_state - state_) * sizeof(float), B.conv_state, bytes,
                      cudaMemcpyDeviceToHost),
           "save state");
    }
    if (B.ssm_state)
        ck(cudaMemcpy(dst + (size_t) (B.ssm_state - state_) * sizeof(float), B.ssm_state,
                      size_t(d_inner_) * d_state_ * sizeof(float), cudaMemcpyDeviceToHost),
           "save state");
}

size_t NemotronHMoe::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void NemotronHMoe::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void NemotronHMoe::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// The Mamba-2 mixer: one projection gives z | xBC | dt (llama.cpp mamba-base.cpp:190-200), the convolution and the
// selective scan follow, then the gate and the grouped norm, then the output projection.  xn_ holds the normed input.
void NemotronHMoe::mamba(Block& B, int il, int n) {
    lin(B.ssm_in, xn_, proj_, n, xq_);   // {d_in_proj, rows}: z (d_inner) | xBC (channels) | dt (ssm_heads)
    const float* p = proj_;
    const int ds = d_in_proj_;           // a token's row of the fused projection
    mamba2_conv_silu(p + d_inner_, B.conv_state, fp(B.conv), B.conv_b.p ? fp(B.conv_b) : nullptr, cv_, channels_,
                     d_conv_, n, ds, st_,
                     snap_bits_ ? snap_at_ + (B.conv_state - state_) : nullptr, rec_n_, snap_bits_);
    mamba2_scan(cv_, p + d_inner_ + channels_, fp(B.ssm_a), fp(B.ssm_d), fp(B.dt_b), ssm_y_, B.ssm_state, d_state_,
                d_inner_, ssm_heads_, n_group_, n, ds, st_,
                snap_bits_ ? snap_at_ + (B.ssm_state - state_) : nullptr, rec_n_, snap_bits_);
    if (snap_bits_) snap_ok_ = true;   // the recurrent state after a window's tokens is snap_, for keep() to return to
    mamba2_gated_norm(p, ssm_y_, fp(B.ssm_norm), d_inner_, n_group_, n, rms_eps_, ds, st_);
    lin(B.ssm_out, ssm_y_, ao_, n);
}

// The attention: normed and turned queries and keys, the keys and values into the caches (llama.cpp
// nemotron-h.cpp:244-258).  The same shape as lfm2moe's, whose reference this mirrors.
void NemotronHMoe::attn(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    // Nemotron's attention uses NEITHER RoPE NOR a q/k norm: llama.cpp's llama_model_rope_type(LLM_ARCH_NEMOTRON_H_MOE)
    // is LLAMA_ROPE_TYPE_NONE (llama-model.cpp:2901-2926), and the GGUF has no attn_q_norm/attn_k_norm (measured).
    // head_norm_rope() would RMS-scale and rotate them regardless (kernels.cu:958-975: scale with w == NULL, and the
    // n_rot pairs turn), so q and k go in exactly as the projections gave them.
    const float* qp = qraw_;
    const float* kp = kraw_;
    if (at) {
        store_f16_at(kp, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
    } else {
        store_f16(kp, B.keys, n_kv_ * head_dim_, pos, n, st_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
    }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    if (n >= kStageRows)
        attend_batch(blas_, qp, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(qp, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
               at, at ? room(n) : 0, score_n_);
    lin(B.wo, att_, ao_, n);
    if (B.wo_b.p) add_bias(fp(B.wo_b), ao_, n_embd_, n, st_);
}

// A layer: the norm, its one module, and the residual.  Nemotron has no FFN beside the module, so the mixer's
// output is added to the residual here and layer_tail does nothing (llama.cpp nemotron-h.cpp:186-211).
void NemotronHMoe::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.kind == kAttn) {
        attn(B, pos_, n);
        dbg("attn", il, ao_, n_embd_);
        axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
    } else if (B.kind == kMamba) {
        mamba(B, il, n);
        dbg("mamba", il, ao_, n_embd_);
        axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        // a MoE layer has no mixer: the engine's experts run after this and layer_tail adds their sum.  What the
        // router needs first is here, as every other architecture's layer_head does it (lfm2moe.cpp:265-267): the
        // scores, and the experts' input quantized - without it the experts read a stale buffer.
        moe_router(L, n);
    }
}

// A MoE layer's residual: the experts' sum (routed through latent_up, plus the shared expert), which the engine left
// in fo_.  A mamba or attention layer's residual was added by layer_head.
void NemotronHMoe::layer_tail(Layer& L, int il, int n) {
    if (blocks_[(size_t) il].kind != kMoe) return;
    axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
}

// The NextN/MTP block (llama.cpp nemotron-h.cpp:142-181): one block, an attention sub-layer and a MoE sub-layer.
// The trunk's layer() never runs it; the engine's draft_rows() and draft_step() do, through the two calls below.
//
// The block's attention keeps its own key/value rows for the model's own tokens: draft_rows() has already fused the
// block's input into xn_/xq_, so this is the same projection and the same cache write attn() makes, without the
// attention itself and without the output projection (a row of keys and values is all a later guess reads).
void NemotronHMoe::block_rows(int64_t pos, int n) {
    Block& B = blocks_[(size_t) n_layer_];
    if (!B.keys || n <= 0) return;
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    store_f16(kraw_, B.keys, n_kv_ * head_dim_, pos, n, st_);
    store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
}

// One guess through the block's attention, at pos (the draft block's own rows are all it reads).
void NemotronHMoe::block_attend(int64_t pos, int n) { attn(blocks_[(size_t) n_layer_], pos, n); }

// The block that guesses ahead must be the one llama.cpp's nemotron-h builds for NextN/MTP.  A file that lacks one of
// its halves is refused loudly here: before this check the block was loaded as a plain MoE layer and the engine
// drafted from half a block (silent 0-4% acceptance - P3-20261016\鍥炴墽\R29-MTP淇濈湡搴?璇荤爜-20261018.md).
void NemotronHMoe::check_draft_block() {
    static const char* const kParts[] = {
        "nextn.eh_proj.weight",   "nextn.enorm.weight",        "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight", "attn_norm.weight",   "attn_q.weight",
        "attn_k.weight",          "attn_v.weight",             "attn_output.weight",
        "post_attention_norm.weight", "ffn_gate_inp.weight",    "exp_probs_b.bias",
        "ffn_latent_down.weight", "ffn_latent_up.weight"};
    for (const char* name : kParts)
        if (!tensor(n_layer_, name))
            throw std::runtime_error("nemotron_h_moe: the block that guesses ahead (blk." + std::to_string(n_layer_) +
                                     ") has no " + name + ": this engine runs the NextN/MTP block as an attention" +
                                     " sub-layer and a MoE sub-layer (llama.cpp nemotron-h.cpp:142-181)");
    if (blocks_.size() <= (size_t) n_layer_ || blocks_[(size_t) n_layer_].kind != kAttnMoe)
        throw std::runtime_error("nemotron_h_moe: the block that guesses ahead (blk." + std::to_string(n_layer_) +
                                 ") is not an attention + MoE block");
}

void NemotronHMoe::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// Nemotron's routing (llama.cpp build_moe_ffn with LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, expert_weights_norm and
// expert_weights_scale): sigmoid gates, the n_used largest of gate + bias, weights from the bare gates, normalized.
void NemotronHMoe::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    static constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the sum
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    // One pass over the experts: the n_used_ largest picks, value descending and a tie the lower index first, which
    // is what the scan this replaced gave (it took the first strict maximum, so equal values came out in index order;
    // only a pick of exactly -INFINITY, which a finite bias cannot make, fell through to repeated index 0 there).
    if (n_used_ <= 32) {
        float best_pick[32];
        int best_idx[32];
        int n_best = 0;
        for (int e = 0; e < n_expert_; ++e) {
            const float v = pick[e];
            if (v == -INFINITY) continue;                 // the scan would only have reached index 0 again
            int at = n_best < n_used_ ? n_best : n_used_;
            while (at > 0 && v > best_pick[at - 1]) --at;
            if (at >= n_used_) continue;                  // not among the largest n_used_
            const int last = std::min(n_best, n_used_ - 1);
            for (int k = last; k > at; --k) {
                best_pick[k] = best_pick[k - 1];
                best_idx[k] = best_idx[k - 1];
            }
            best_pick[at] = v;
            best_idx[at] = e;
            if (n_best < n_used_) ++n_best;
        }
        for (int k = 0; k < n_used_; ++k) {
            const int e = k < n_best ? best_idx[k] : 0;
            ids[k] = e;
            w[k] = probs[e];
            sum += w[k];
        }
    } else {
        for (int k = 0; k < n_used_; ++k) {
            int best = 0;
            for (int e = 1; e < n_expert_; ++e)
                if (pick[e] > pick[best]) best = e;
            pick[best] = -INFINITY;
            ids[k] = best;
            w[k] = probs[best];
            sum += w[k];
        }
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

// The shared expert (llama.cpp nemotron-h.cpp:302-308): it reads the model's width, not the latent, and is not gated
// (relu2).  It writes fo_; the routed experts' latent sum is mapped in by moe_latent() and added after.
void NemotronHMoe::moe_shared(const Layer& L, int il, int n) {
    // STRATA_SHARED_OVERLAP=1: decode's calls (n <= 8 tokens) run both products on sh_, beside the routed experts'
    // work on st_, and moe_latent() waits for the event before the routed sum is added into fo_ (the only other
    // writer).  xn_ is only read here - the next write is the next layer's rms_norm, after that wait - and the
    // block's rows and quantized inputs are its own (sh_g_/sh_h_/q8_sh_), so the routed path's g_/h_/q8_ are
    // untouched.  Prefill (n >= kStageRows) and the recorded-graph path (all_cached()) keep the old call.
    // A graph capture on st_ must not have the shared expert's work on another stream: it would be outside the
    // graph (and the replay would run it on st_ beside nothing).  A capture in progress takes the old path.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cs(st_), &cap) != cudaSuccess) {
        cudaGetLastError();
        cap = cudaStreamCaptureStatusNone;
    }
    if (shared_overlap_ && n <= 8 && !all_cached() && L.latent_up.p && cap == cudaStreamCaptureStatusNone) {
        // st_ owns xn_ until its head/rms_norm is done; sh_ must not read it before.  The event is recorded where
        // the caller has just finished the layer's head on st_ (moe_shared_timed() is called right after it).
        ck(cudaEventRecord((cudaEvent_t) ev_st_, cs(st_)), "shared expert");
        ck(cudaStreamWaitEvent(cs(sh_), (cudaEvent_t) ev_st_, 0), "shared expert");
        kernels::native_quantize_q8_1(xn_, q8_sh_, n_embd_ * n, 1, sh_);
        lin(L.sh_up, xn_, sh_g_, n, q8_sh_, 0, sh_);   // {n_embd, n_ff_shexp}
        relu2(sh_g_, sh_h_, (size_t) n * n_ff_shexp_, sh_);
        kernels::native_quantize_q8_1(sh_h_, q8_sh_, n_ff_shexp_ * n, 1, sh_);
        lin(L.sh_down, sh_h_, fo_, n, q8_sh_, 0, sh_);  // {n_ff_shexp, n_embd} -> fo_
        ck(cudaEventRecord((cudaEvent_t) ev_sh_, cs(sh_)), "shared expert");
        sh_pending_ = true;
        return;
    }
    // xq_ holds the experts' 1024-wide latent rows here (moe_router refilled it), so the 4096-wide sh_up must not
    // read it: pass nullptr and let lin quantize xn_ (n_embd) afresh into q8_ (t2, P3-20261016\t2-cellprobe-20261016.md).
    lin(L.sh_up, xn_, g_, n, nullptr);  // {n_embd, n_ff_shexp}
    relu2(g_, h_, (size_t) n * n_ff_shexp_, st_);
    lin(L.sh_down, h_, fo_, n);         // {n_ff_shexp, n_embd} -> fo_
}

}  // namespace strata::poly
