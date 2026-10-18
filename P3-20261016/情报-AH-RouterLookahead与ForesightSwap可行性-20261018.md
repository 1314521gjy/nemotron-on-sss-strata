# 情报-AH · RouterLookahead + ForesightSwap 可行性（零 GPU，读码 + 离线算）— 2026-10-18

```
状态: 达成（老师形态/搬运已读清；打点方案 ≤25 行；判决 + 1.9/1.11 量化）
席位: 验收（reviewer）· 任务: lead 派（可行性研究）
纪律: 零 GPU、零编译、不改码、未 spawn 子代理；行号为本次读码时源码
树: 老师 = up/0.1.44.1（E:\视频\_flashnext\...\upstream-0.1.44.1）；我们 = PolyStrata\src\poly
```

## §0 判决（先说结论）

1. **老师机制读清**：RouterLookahead = 「layer l 的 MoE 输入 h × **layer l+1 的 `ffn_gate_inp`（BF16 host copy）** → 每 token top-k（k 默认 **10**）→ 去掉 cache/RAM 驻留 → `warm()` 只预热页」；ForesightSwap = 每层 N 个 **VRAM 槽**（copy stream），装「刚 miss 的 + 下一层预测的」，**槽从 expert cache 的 VRAM 扣**。老师实测 FS decode **0.91–0.98**（4/16 槽，3060/P100/5070）⇒ **left off**（`docs/DETAILS.md:1736-1744`）。
2. **我们 0 命中**：全树 grep `lookahead/foresight/warm(` **全 0**；我们只有按"近期使用"换入（`follow()`），没有预测；没有 `--dump-routing`（0 命中；只有 `--draft-dump`/`--dump` checksum）。
3. **判决：值得做，但必须先打点**——预测准确率无盘上数据（老师先验 recall 27–34%、precision ~8%）；打点 **~22–25 行、1 编译、2 跑**，离线评估零 GPU；达到 go 门（recall≥30%）再实现 FS-over-idle-stage（~150–250 行）。
4. **1.9 不是杠杆**：per-layer 等配额 67.01% vs 全局 67.37%（−0.36pp）；"按曲线贪心"≡全局贪心=当前 ⇒ **0 增益**。**1.11**：输出头 440.4MB = +142 槽 ⇒ prefill **+4.7–6.1%**（D）。

## §1 A. 老师实现（file:line）

### 1.1 RouterLookahead 预测形态

- 接口原文：`include/strata/core/expert_source.hpp:241-245`：**"when the CPU pool starts layer `l`, a worker thread applies layer `l+1's router (BF16, host copy) to layer l's MoE input - the residual stream changes little from one layer to the next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the source to `warm` the rest"**；预测**只预热页，不改计算**（`:244-245`）。
- 数据：`routers_[l]` = layer l 的 `ffn_gate_inp` **BF16 bits**，n_expert×n_embd（`:253-254`, `:279`）；`k_` 默认 **10**（`:281`）；`depth_` 默认 **1**（`STRATA_IO_PREFETCH_DEPTH` 1–4，`:263-266`）。
- 调度：`submit(layer, x, n_tok, host_res)` 拷贝 ≤8 token 的 host h，**不等待**（`:260-262`, `:2239-2252`）；`run()` 线程评分 → top k → 去掉 `host_res` 驻留 → `src_->warm()`（`:2349-2458`）；预测中带 `ForesightSwap*` 时只对**下一层**（`dj==0`）调 `fs->predict()`（`:2447-2449`）。
- 自带准确率统计：`STRATA_LOOKAHEAD_STATS=1` 记 top k/k+4/k+10，并按 voterank top M（M=4/8/12）统计 recall/precision（`hpp:267-272`、`cpp:2266-2348`）。
- 老师实测（先验）：`bench/results/2026-10-04-community-rtx-5070ti-windows/README.md:117` **recall 27–34%、precision ~8%（fetch set）**，最好遮 ~28% PCIe wait，但带宽浪费 ~12× ⇒ **closed**；`bench/results/2026-10-01-iq4xs-64gb/README.md:104-105` 64GB（专家全在 RAM）**无可测效果**（预热无意义）；Unsloth Q4 的 1.5–2× 只在冷盘场景。

### 1.2 ForesightSwap 搬运机制

- 开关：`STRATA_FS_SLOTS=N` = **每层 N 个 VRAM 槽**；copy stream 用「刚 miss 的专家 + 下一层 router 预测的专家」填（`STRATA_FS_AHEAD=0` 只填 miss）；`STRATA_FS_VERIFY=1` 可把落地槽读回逐字节校验（0 bad in 2,300+ reads）。
- **槽的来源：从 expert cache 的 VRAM 里拿**（`--vram-reserve-mib` 要加约 N×80 MiB）——这是老师测负的主因。
- 实测：`docs/DETAILS.md:1736-1744`：4 槽 → RTX 3060 **0.95/0.91**、P100 0.97/1.00、5070 0.98；P100 16 槽 0.96 ⇒ **"smaller cache cost more than the slots returned"**，**left off**。
- 实现：`ForesightSwap::take/hold/unhold/predict/note_miss/request_locked`（`src/core/expert_source.cpp:4916-5172`）；`fill_loop` 在 copy stream 上按队列搬（`:4991-5016`）。

### 1.3 我们的缺口（grep 全量）

| 模式 | 我们 `src/poly` 命中 | 老师 0.1.44.1 |
|---|---|---|
| `lookahead` | **0** | expert_source.* + docs |
| `foresight` | **0** | foresight_swap.hpp + expert_source |
| `warm(` | **0** | ExpertSource::warm |
| `next_layer` | **0** | RouterLookahead 的 depth 用 |
| `DUMP_ROUT` / `dump-routing` | **0** | `--dump-routing` + 工具 |

- 我们现有的"换入"是**事后**的：`Engine::follow()`（`engine.cpp:2250`）、`follow_loop()`（`:2186`）、`--expert-swaps`（`engine.hpp:68`）、`swap_half/swap_ratio`（`engine.hpp:96-97`）——按**近期使用计数**，不是预测。

## §2 B. 打点方案 + 离线评估

### 2.1 现状：无法直接估预测准确率

- 我们只有 **profile 计数**（`p_master.txt`，20,275 行 `层 专家 计数`），没有"每 token 每层 routed ids 序列"；`--expert-profile-write` 也只写计数。
- 引擎**没有路由 dump**（0 命中；`--draft-dump` 只打 draft 猜测，`main.cpp:345/523-638`；`--dump` 是 stage checksum，`engine.hpp:53`）。

### 2.2 最小打点（≤30 行，估 22–25 行）

- env：`STRATA_DUMP_ROUTING=<path>`（不开=`NULL` 行为不变）；可选 `STRATA_DUMP_ROUTING_H=1` 同时写 h。
- 钩点：`Engine::moe()` 的 route 之后（`engine.cpp:3243`，decode 路径；`moe_card()` 只在**全部专家驻留**时走，delivery 28% 驻留不触发，可后补）。
- 记录（每 token 每 MoE 层一行）：`il, pos, tok_id, n_ids, ids[n_used=22]`；`_H=1` 时再写 `xn_` 的 `n_embd=4096` floats（h = 该层 MoE 输入，engine 里 `xn_`，路由 `ffn_gate_inp` 的输入；`nemotronhmoe.cpp:409-462` 的 sigmoid+bias+top22 语义）。
- 只 dump decode：`if (n <= 8)`；单线程顺序写，无需锁；文件预计 **~84MB/条**（128 token×40 层×(22 ids+4096 f32)）。
- 改动量：helper（open/env/header）~12 行 + `moe()` 调用 ~6 行 + env 解析 ~4 行 ≈ **22–25 行**；**1 编译**；**2 跑**（alt 293 / p6792，decode 128 token，~30 s/跑）→ **零 GPU 离线评估**。

### 2.3 离线评估（零 GPU，拿到 dump 后）

1. 解析 dump → 每 (pos, layer) 实际 routed ids（22 个）。
2. 从 GGUF 取下一层 router：`blk.L.ffn_gate_inp.weight`（本模型 **F32**，shape [4096,512]，A 级已核）+ `blk.L.exp_probs_b.bias`（F32 512）。
3. 预测：`s = sigmoid(W_{l+1} · h_l + b_{l+1})`；按 `s + bias` 取 top-k（k=10 与 22 两档）⇒ pred 集合。
4. 指标（对齐老师 `stats_score`）：`recall = |pred ∩ actual_miss_l+1| / |actual_miss_l+1|`、`precision = |pred ∩ actual_miss| / |pred|`、每层预测数/predicted-per-layer、voterank top M 的 union recall；`actual_miss` 用 profile top-5721 作静态驻留代理（或 dump `host_res` 位图更准，+8 行）。
5. 基线：id 序列 bigram/transition 预测（同 dump 可算），与 h×router 法对比。
6. 理论收益公式（本机 alt 交付口径读数）：wall **53.9 ms/token**、CPU pool **37.1 ms（68.8%)**、miss **340.9/token**、`u≈108.8 µs/miss`（ledger cpu_pool/round ÷ miss/round）。若预测把 fraction f 的 miss 变成 GPU hit：`ΔW ≈ −f·m·u + f·m·g` ⇒ 增益 ≈ `f × 68.8% × (1 − g/u)`；f 受 recall、槽数与 copy 带宽限制：
   - 槽：stage_+ring2 3 GB ⇒ **968 槽/模型**（≈24/层）；每层 miss ≈8.5/token ⇒ 槽够；
   - 带宽：层时 ≈1.35 ms × 25 GB/s ≈ 34 MB ≈ **11 专家/层** ⇒ 带宽够（若预测准）；
   - 上限：recall 30% ⇒ f≤0.3 ⇒ +14–20%（理想，g→0）；现实（g>0/精度损耗）**+8–15%（D）**；recall 10% ⇒ +3–7% 边缘。

## §3 C. 判决 + 计划

**值得做（有条件），先打点**：

- **Step 1（零 GPU 决策步，1 编译 + 2 跑 + 离线）**：dump → 算 recall/precision/投票覆盖。**go 门**：`recall ≥ 30%`（对 miss）且 predicted-miss/层 ≤ 24 槽（否则槽不够）；`precision` 只影响带宽浪费（老师 ~8% 也接受），不作为门。
- **Step 2（实现，~150–250 行，默认关）**：把 **decode 期间闲置的 stage_+ring2（3GB）改造成 FS slot pool**（与老师的区别：**不从 cache 扣槽** ⇒ 不重演老师 0.91–0.98 的负结果）；复制老师的 RouterLookahead（BF16 权重 host copy、thread、warm/predict），对接我们的 `route()`/`host_res`；与 prefill borrow/stage 互斥（prompt 期间不跑）；开关 env；ids/质量门不要求变（预测不改计算，落地槽只是把 CPU miss 换 GPU 算）。
- **风险**：预测准度（对我们模型/负载未知）；copy 抢带宽；槽生命周期与 prefill 冲撞；老师 FS 负结果（槽从 cache 拿时亏）——我们不从 cache 拿，但仍要实测。
- **h 不可得？不成立**：h 在引擎里可得（`xn_`，4096 floats，`moe()` 内），router 权重在 GGUF（F32），不依赖老师专属数据结构。

## §4 D. 1.9 / 1.11 量化

### 4.1 1.9 Coverage Curve / per-layer slot 配额（本席自己模拟）

数据：`p_master.txt`（40 层 × 507 专家 = 20,275 条，总计数 **53,957,200**；每层总计数**恰好相等** 1,348,930）。容量 = 交付 5,721 槽。

| 分配策略 | 命中（频次加权） | Δ vs 全局 | 每层槽数 |
|---|---|---|---|
| **全局贪心（现交付）** | **67.37%** | — | min 96 / max 183 / mean 143 |
| 每层等配额 143 | 67.01% | **−0.36pp** | 143–144 |
| 按层总量比例 | 67.00% | −0.37pp | ≈143 |
| "按曲线贪心"（逐槽取全局边际最大） | **67.37%** | **0** | = 全局贪心 |

⇒ **per-layer 配额不是杠杆**：频次加权命中下，全局贪心已是上界；"按曲线贪心"与现交付**同一个集合**（0 增益）；等配额最多损失 0.36pp。与 `情报-W` 的 "+0.4 点/层间不倾斜"一致。真实杠杆是**表质量（24.1 点：98.26% 计数来自 prefill、缺 blk.88）**与**槽比例（16.6 点，27.9%→46.5%）**。
（`情报-W`：top-5721 = 67.4%，等分 143 = 67.0% ⇒ 层间最多 0.4 点；我们复算 67.37 vs 67.01。）

### 4.2 1.11 输出头 alias（440.4MB 借给 staging）

- 输出头大小：`output.weight` Q6_K 4096×131072，payload = 4096×131072×210/256 = **440.4 MB**（本次从 GGUF 载荷核算，A 级；`dense_` 里 VRAM，`engine.cpp` 的 `dense_ = dmalloc(...)`）。
- +142 槽：440.4 / 3.0966 MB = **142.2**。
- 敏感度：R54 P1 = `--stage-mib 2048→1024`（缓冲 −50%）⇒ prefill **−11.0%**（`回执\R54-prefill瓶颈探针-20261018.md:5/20/33`）。
  - 线性外推：+142 槽 = +21.5%（对 2048 MiB）⇒ **+4.7% prefill**；按实际 ring2 有效缓冲 1512 MiB 算 +27.7% ⇒ **+6.1%**。
  - ⇒ **+4.7–6.1% prefill（D 级，需先补一次 `--stage-mib 4096` 实测标定上界）**；decode 无影响。
- 实现代价：head 目前在 `dense_` VRAM 池里（与其它 dense 权重同分配），要"借给 staging"需把 head 拆成独立 buffer + 借/还逻辑（~30–60 行，中风险：末 token logits 需要、与 borrow/stage 复用互斥）；不建议作为下一步第一优先。

## §5 未验 / 边界

- **未编译、未跑**；预测准确率**无盘上记录**（这正是要打点的原因）；`--dump-routing` 不在我们树里（0 命中），打点是新增。
- 老师先验是 AMD/不同模型/不同驻留 regime：`recall 27–34% / precision ~8%` 是 **C 级**；`64GB 全驻留时 lookahead 无可测效果` 说明"页预热"在我们（全 RAM 驻留）上≈0——但我们的 FS 是"换腿"（CPU→GPU），不是预热，属于**新用法，无先例**。
- 1.9 模拟基于 `p_master`（prefill-heavy、无 blk.88）；decode 真实流量表未知（`情报-W` 24.1 点）。
- 1.11 的 +142 槽是"能借到"的算术；实际头在 `dense_` 池中、且 prefill 期间是否真闲置需读码确认（`output_` 只在最后 token logits 用；本次未逐行核）。
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AH-RouterLookahead与ForesightSwap可行性-20261018.md`
- 回执：`...\P3-20261016\回执\RV-20261018-RouterLookahead可行性.md`
