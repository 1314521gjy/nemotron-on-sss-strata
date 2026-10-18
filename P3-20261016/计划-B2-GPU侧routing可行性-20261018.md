# 计划 · B2（GPU 侧 routing）可行性 — 2026-10-18 · builder4（零 GPU/零编译，只读码 + 静态分析）

**判决：建议判否（B2 GPU routing 不进）。** 理由：① ids 位精确只有"反汇编复刻 UCRT expf"或"GPU 预选+host 精确复核回退"两条路，工作量/风险都大；② 但它的目标——decode 的 `router_wait 7.7ms`——**不是 D2H 开销，而是流水线依赖等待**，GPU routing 省不掉；③ 真正能不改数值就拿到收益的是"把 D2H 从 kernel 写 pinned 改成 copy engine"（见⑤）。

## ① host 端完整语义（`src/poly/nemotronhmoe.cpp:409-462`）
- `probs[e] = 1.f / (1.f + std::exp(-logits[e]))`（`:414`）——`std::exp(float)` → CRT `expf`，**fp32**；`pick[e] = probs[e] + bias[e]`（`:415`，bias=`blocks_[il].probs_b`）。
- 选 n_used(=22) 个最大 `pick`：一遍插入序（`:421-439`），**值降序、并列取更小 index**（`v > best_pick[at-1]` 严格比较 + e 升序插入），`-INFINITY` 跳过；另有 n_used>32 的 O(n·k) 分支（`:446-456`）。
- `w[k]=probs[id]`（`:443`），`sum` 累加，`denom=max(sum, 6.103515625e-5f)`（w_norm 时；`:457`），`w/=denom`（`:459`），`w*=w_scale`（`:460`）。
- 调用点：decode `engine.cpp:3377`（`moe()` 内逐 token）；prefill `engine.cpp:4034-4042`（`moe_rows`，16 线程按 token 分片）。

## ② D2H 传的到底是什么（`engine.cpp:3325-3339`）
- `router_scores`（`:3320`，`lin` 在原 st_ 上算 512 logits/token）→ latent_down/quantize → `copy2(rl_, h_rl_, n*n_expert_, in, h_x_, n*in_w, st_)`（`:3338`）；`copy2` 是 **CUDA kernel**（`kernels.cu:814 copy2_kernel`，`:2269 void copy2`）**由 device 直接写 pinned host** `h_rl_`（并非 copy engine）。
- **prefill：n≈6912 → 13.9 MB/层**（6912×512×4B ≈14.16MB，与实测口径一致）；**decode：n=1 → 2 KB/层**（×40 层=80 KB/轮，可忽略）。
- **不能减半成 bf16**：`probs/pick/top-k` 全由 logits 位决定 ⇒ bf16 是 class-E，ids 门必挂；fp32 全量不能省（选择需要全部 512 个）。⇒ **decode 的 7.7ms 与 D2H 大小无关**（2KB×40 的传输 <1µs 级），它是 host `cudaEventSynchronize(ev_)`（`:3355`）等卡上 router lin/前序工作完成的时间。

## ③ 位精确候选（风险逐个列）
- **A. GPU 算 probs+topk**：CUDA `expf` 与 MSVC/UCRT `expf` **无位一致保证**（CUDA 文档 expf 最大 2 ulp，非正确舍入；UCRT 实现不公开）⇒ `pick` 在近并列时可翻转 ⇒ ids 漂。**否决**。
- **B. GPU 用 double 算 probs→fp32**：更准 ≠ 与 host 的 fp32 expf 同值（舍入路径不同）⇒ 仍可能不同。**否决**。
- **C. device 复刻 MSVC expf**：后续 `1+`、除法（IEEE 正确舍入）、加法/比较在两边都确定 ⇒ **只有 expf 挡路**；理论上可位精确，但需从本机 UCRT 反汇编 expf 并 device 复刻 + 穷举/随机差分验证；工作量 ~300–500 行 + 一个验证门，且 UCRT 版本差异会破坏"以后也精确"的前提。**理论可行、成本高**。
- **D. GPU 预选 + host 精确复核（带回退）**：GPU（double）选 top-(22+G) 并回传候选 logits + 第 22/23 名 gap；host 用原语义算 probs、验证"未选者不可能进 top22"（gap 不足则回退整层全量 D2H）。**可做到位精确**，但 host 仍需每层 round-trip 拿候选，**decode 的依赖 stall 仍在**；收益只在 prefill 的 13.9MB⇒~0.3MB（~0.5% 级 prefill）与省 host 选 22 的小时间。实现复杂、收益小。
- **E. 只回传 bf16/候选**：语义漂（class-E）⇒ 与验收冲突。**否决**。

## ④ 判决
**B2 GPU routing：建议判否。** 位精确要么不可证（A/B/E），要么成本高（C/D）；而它对 decode 的最大目标 `router_wait 7.7ms` **无效**（那是 host 等 GPU 的依赖 stall，不是 D2H 带宽/延迟）；对 prefill 只有 ~0.5–1% 级且实现复杂。**改为做⑤的两条"不改数值"优化。**

## ⑤ 不改数值的替代优化（top-2，均纯拷贝/纯调度，ids 必同）
1. **D2H 改 copy engine（推荐，先做）**：现在 `copy2_kernel` 由 device 直接写 pinned host（PCIe 分散写，估 3–6GB/s）；改成 **device→device 暂存 + `cudaMemcpyAsync(..., D2H)`**（或直接 `cudaMemcpyAsync(rl_→h_rl_, pinned)`，bit 相同却走 copy engine ~20–25GB/s）。prefill 每层 13.9MB、40 层，保守估可省 **~50–100ms**（prefill 1–2%），ids 零风险；decode 2KB 影响可忽略。
2. **拆 `router_wait` 的依赖（decode）**：`route_wait 7.7ms` 是 `cudaEventSynchronize(ev_)` 等 st_ 到 router 为止的全部工作（含上一层残留与 router lin）；用**单独 stream/更细 event**只等"router lin 完成"，或在 `layer_head` 后尽早发 router lin（与上一层的 CPU 腿重叠），把 stall 拆小；再顺手实测 host `route()` 单线程 512×expf 的耗时（估 0.1–0.2ms/轮），若可观再考虑"预计算 probs 表"之类仍位精确的提法（不改 libm 调用顺序）。
- 已排除：`copy2` 已是 pinned/kernel 写（"mapped + poll" 已实现，省 launch 只有 µs 级）；D2H 用 bf16/压缩 = class-E。
