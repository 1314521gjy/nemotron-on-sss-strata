# 证据：b2-static-grep（批② 静态核对）

- 来源：`C:\sp3\evidence\b2-static-grep-20261018.txt`（5,711 B，mtime 2026-10-18 13:40:02）
- 本件为该文本的仓内副本（原样收录；仅加本头）

```text
# Stage D batch-2 static evidence (final build): every L.exp_down / exp_down read, classified
at: 2026-10-18 13:39:52

criterion: hot-path GGUF down reads = 0; ONLY allowed GGUF down read = load-time 4-expert check (6.2 MB, once).

=== every exp_down hit in engine.cpp, with its classification ===
606: if (std::memcmp(rebuilt.data(), L.exp_down + (size_t) e * L.d_bytes, L.d_bytes) == 0) ++okv;  ==> ALLOWED (lead ruling: the ONLY GGUF down read; on-load 4-expert check, 6.2 MB once; anchors sidecar to real GGUF bytes)
1126: L.exp_down = home(td).data(td);  ==> ALLOWED #21: load_moe pointer assign (does not read)
1181: if (!mxfp4_block_to_q5_0(L.exp_down + b * kIn, out.get() + b * kOut)) bad.store(true);  ==> ALLOWED #17: mxfp4->q5_0 conversion, MXFP4 models only (this IQ4_NL model never enters)
1189: L.exp_down = out.get();  ==> ALLOWED #17: same mxfp4 path assigns a new buffer
1392: if (!L.moe || !L.exp_up || !L.exp_down) continue;  ==> ALLOWED: pointer null test only
1409: if (good && !r2d_mode) reg(L.exp_down, dn_len, "down");  ==> ALLOWED #11: pin registration, r2d_mode skips it (and no data read)
1433: if (!exp_down_dev_.empty()) {  ==> ALLOWED #12: pcie-direct guard (refuses switch; no read)
1450: const cudaError_t r = cudaHostUnregister((void*) L.exp_down);  ==> ALLOWED: inside dead pcie/switch branch, not reached with r2d
1652: else ck(cudaMemcpy(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),  ==> SWITCHED: else-branch of materialize_down (r2d_base_ null only)
1805: else ck(cudaMemcpyAsync(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice,  ==> SWITCHED: else-branch of materialize_down (r2d_base_ null only)
1836: if (std::memcmp(buf.data() + L.slot_down(), L.exp_down + es * L.d_bytes, L.d_bytes) != 0) return false;  ==> ALLOWED #16: borrow-verify debug hook; env refused at load with r2d
2014: else ck(cudaMemcpy(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),  ==> SWITCHED: grow_cache else-branch (r2d_base_ null only)
2063: std::memcpy(s->pinned + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes);  ==> SWITCHED: swap else-branch (r2d_base_ null only)
2440: if (!r2d_no_down_touch) spans.emplace_back(L.exp_down, (size_t) n_expert_ * L.d_bytes);  ==> ALLOWED #13: map_pages down span; skipped when r2d (r2d_no_down_touch)
2589: exp_down_dev_.assign(layers_.size(), nullptr);  ==> ALLOWED #14: pcie direct device-alias table (no read)
2592: if (!L.moe || !L.exp_up || !L.exp_down || il >= pin_layer_.size() || !pin_layer_[il]) continue;  ==> ALLOWED #14: pcie direct pointer table (no read)
2596: cudaHostGetDevicePointer(&dn, (void*) L.exp_down, 0) != cudaSuccess || dn == nullptr) {  ==> ALLOWED #14: cudaHostGetDevicePointer on an alias (no data read)
2602: exp_down_dev_[il] = (const uint8_t*) dn;  ==> ALLOWED #14: pcie alias table (no read)
3062: L.exp_down + e * L.d_bytes, w[k], t, downT};  ==> SWITCHED: moe() CPU CpuExpert.down field kept for =0; r2d path uses downT (asserted in cpu_experts)
3100: exp_down_dev_[(size_t) il] != nullptr;  ==> ALLOWED #14: pcie direct alias null test
3126: pcie_src_down[j] = exp_down_dev_[(size_t) il] + e * L.d_bytes;  ==> ALLOWED #14: pcie direct alias DMA source (pcie refused at load with r2d)
3132: pcie_src_down[j] = L.exp_down + e * L.d_bytes;  ==> ALLOWED #14: pcie non-direct source (pcie refused at load with r2d)
3490: L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice, cs(stream)),  ==> SWITCHED: stage() pin-branch else (r2d_base_ null only)
3507: else if (!r2d_base_) std::memcpy(h + c * gu2 + j * L.d_bytes, L.exp_down + e * L.d_bytes, L.d_bytes);  ==> SWITCHED: stage() non-pin host gather, skipped when r2d (else-if)
3711: L.exp_down + e * L.d_bytes, w[i], i / n_used_, downT});  ==> SWITCHED: moe_rows host_ CpuExpert.down kept for =0; CPU-share uses downT (asserted)
3971: if (std::memcmp(got.data(), L.exp_down + e * L.d_bytes, L.d_bytes) != 0)  ==> ALLOWED #15: RING2_CHECK debug memcmp; env refused at load with r2d

=== down_src (removed helper; must be NONE) ===
NONE

=== materialize_down call sites (the read-source switches) ===
395: void Engine::materialize_down(const Layer& L, int il, size_t e, uint8_t* dst, void* stream) {
1651: materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), nullptr);
1804: materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), cs(st_));
2013: materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), nullptr);
2061: if (err == cudaSuccess) materialize_down(L, s->il, e, s->slot + L.slot_down(), cs(swap_st_));
3487: materialize_down(L, il, e, dst + down_at + (size_t) (j0 + (int) j) * L.d_bytes, cs(stream));
3515: materialize_down(L, il, e, dst + down_at + ((size_t) j0 + j) * L.d_bytes, cs(stream));

=== r2d_slots_t_ (D1 wiring must stay dormant: no assignment to true) ===
3252: if (r2d_slots_t_) {   // Stage D D3: the slot's down segment holds the neuron-major T layout
3318: if (r2d_slots_t_) {   // Stage D D3: the slot's down segment holds the neuron-major T layout
3425: if (r2d_slots_t_) {   // Stage D: the card's routed down kernel reads the row-major slot; the T layout is D3+

=== runtime no-silent-fallback assertion (cpu_experts.cpp) ===
448: throw std::runtime_error("relu2 T down: the live index / downT does not cover this call; refusing the GGUF "

=== correction: line 606 classification (the reviewer-allowed one) ===
606: on-load 4-expert check, relu2_from_T(sidecar) vs GGUF bytes: ALLOWED (lead ruling: the only GGUF down read; 4 experts x 1,548,288 B = 6.2 MB, load-time once; it anchors the sidecar to real GGUF bytes)

```