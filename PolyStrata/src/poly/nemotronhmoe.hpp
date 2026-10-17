// src/poly/nemotronhmoe.hpp - NVIDIA's Nemotron 3.5 Super VL 120B A12B (llama.cpp architecture "nemotron_h_moe").
//
// One residual stream, and every layer holds exactly one module (llama.cpp nemotron-h.cpp:191-199: is_recr -> the
// Mamba-2 mixer; n_ff == 0 -> attention; else -> routed experts).  No layer has a FFN beside its mixer.
//   * a Mamba-2 layer: norm, gated convolution + selective scan + gated grouped norm, out projection.
//   * an attention layer: norm, attention over a kv cache, out projection.
//   * a MoE layer: norm, a latent block of routed experts (relu2, no gate) and a shared expert.
// A layer's kind is read from the tensors present (as qwen35.cpp does), not from attention.head_count_kv alone:
// that array is 0 for both the mixer and the MoE layers (llama.cpp separates them by n_ff, which the GGUF exposes
// per layer only through the tensors).
#pragma once

#include "engine.hpp"

namespace strata::poly {

class NemotronHMoe final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer& L, int il, int n) override;
    /// The NextN/MTP block after the model's layers (blk.<n_layer_>): an attention sub-layer and a MoE sub-layer in
    /// one block (llama.cpp nemotron-h.cpp:142-181), so its machinery is wired here, not in the trunk's layer().
    void check_draft_block() override;
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;

private:
    /// A layer's kind, from the tensors it has.  kAttnMoe is the MTP block alone: the only block with both an
    /// attention sub-layer and a MoE sub-layer.
    enum Kind { kMamba, kAttn, kMoe, kAttnMoe };

    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        Kind kind = kMamba;
        // Mamba-2 mixer (llama.cpp nemotron-h.cpp:74-90).  ssm_in gives z | xBC | dt in one row of d_in_proj values.
        Dev ssm_in, conv, conv_b, dt_b, ssm_a, ssm_d, ssm_norm, ssm_out;
        float* conv_state = nullptr;   ///< channels * (d_conv - 1) past inputs, oldest first a channel
        float* ssm_state = nullptr;    ///< d_inner * d_state, the scan's state
        // attention
        Dev wq, wk, wv, q_norm, k_norm, wo, wo_b;
        uint16_t* keys = nullptr;      ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
        // MoE (its routed experts and the latent projections).  The routed experts and the shared expert go through
        // load_moe() (non-gated, latent); the layers keep the latent projections here and copy them onto the Layer.
        Dev latent_down, latent_up;
        std::vector<float> probs_b;   ///< the selection bias the router's gate adds (exp_probs_b.bias)
        Dev probs_b_dev;
    };

    void mamba(Block& B, int il, int n);
    void attn(Block& B, int64_t pos, int n);
    Kind kind_of(int il) const;

    std::vector<Block> blocks_;    ///< one a layer of layers_
    std::vector<int> kv_heads_;    ///< a layer's key heads; 0 for the mixer and the MoE layers (a cross-check)
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    float theta_scale_ = 0;

    // Mamba-2 geometry (llama.cpp LLM_KV_SSM_*)
    int d_state_ = 0, d_inner_ = 0, d_conv_ = 0, n_group_ = 0, ssm_heads_ = 0, channels_ = 0, d_in_proj_ = 0;
    int n_latent_ = 0;             ///< moe_latent_size (Nemotron 1024); 0: the experts are as wide as the model
    int n_mamba_ = 0, n_attn_ = 0, n_moe_ = 0;
    bool w_norm_ = true;
    float w_scale_ = 0.f;
    int chunk_ = 1;                ///< tokens the attention takes at a time

    // device scratch
    float *proj_ = nullptr, *cv_ = nullptr, *ssm_y_ = nullptr;
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    float* awork_ = nullptr;       ///< a prompt batch's attention (attend_batch)
    size_t score_n_ = 0;
};

}  // namespace strata::poly
