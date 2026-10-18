# 回执 · RouterLookahead + ForesightSwap 可行性（零 GPU）— 2026-10-18

```
状态: 达成（读码 + 离线模拟 + 判决；零 GPU/未编译/未改码/未 spawn 子代理）
席位: 验收（reviewer）· 任务: lead 派（情报-AH）
```

## 交付物

| 绝对路径 | 内容 |
|---|---|
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AH-RouterLookahead与ForesightSwap可行性-20261018.md` | 主件：老师形态/搬运 file:line · 打点方案（≤25 行）· 离线评估 · 判决/计划 · 1.9/1.11 量化 |
| **本件** `...\P3-20261016\回执\RV-20261018-RouterLookahead可行性.md` | 五段回执 |

## 判据读数

| 项 | 读数/依据 |
|---|---|
| RouterLookahead 形态 | `expert_source.hpp:241-245`：layer l+1 的 router（**BF16 host copy**）作用在 layer l 的 MoE input 上（"residual stream changes little"）→ 每 token top-k（`k_=10`，`:281`）→ 丢掉 cache/RAM 驻留（`host_res`）→ `warm()` 只预热页；depth 1–4（`:263-266`）；stats 记 k/k+4/k+10（`:267-272`、`cpp:2266-2348`）；submit/run `cpp:2239-2252/2349-2458`；只对下一层喂 FS（`cpp:2447-2449`） |
| ForesightSwap 搬运 | `STRATA_FS_SLOTS=N` 每层 N 个 VRAM 槽，copy stream 填「刚 miss + 下一层预测」；`STRATA_FS_AHEAD=0` 只 miss；**槽从 expert cache VRAM 扣**（reserve +N×80MiB）；老师实测 4/16 槽 decode **0.91–0.98**（3060/P100/5070）⇒ left off（`docs/DETAILS.md:1736-1744`；实现 `expert_source.cpp:4916-5172`） |
| 我们先验 | 全树 grep `lookahead/foresight/warm(` **0 命中**；只有按近期使用换入（`follow()` engine.cpp:2250、`--expert-swaps` engine.hpp:68）；**无路由 dump**（`DUMP_ROUT`/`dump-routing` 0 命中） |
| 老师准确率先验 | `bench/results/2026-10-04-community-rtx-5070ti-windows/README.md:117`：recall **27–34%**、precision **~8%**（fetch set），遮 ~28% PCIe wait、带宽浪费 ~12× ⇒ closed；`2026-10-01-iq4xs-64gb/README.md:104-105`：64GB 全驻留时 lookahead **无可测效果** |
| 打点方案 | env `STRATA_DUMP_ROUTING=path`（可选 `_H=1` 写 h）钩 `moe()` route 后（engine.cpp:3243），写 `il,pos,tok,ids[22]`(+`xn_` 4096 f32)；只 dump decode（n≤8）；est **22–25 行**、1 编译、2 跑（~84MB/条） |
| 离线评估 | 读 GGUF `blk.L.ffn_gate_inp.weight`（F32 [4096,512]，A 级核对）+ `exp_probs_b.bias`；`s=sigmoid(W·h+b)` top-k 对比实际 l+1 ids；算 recall/precision/voterank；收益 ≈ `f×68.8%×(1−g/u)`（wall 53.9ms、cpu pool 37.1ms、miss 340.9、u≈108.8µs；968 槽≈24/层；带宽 11 专家/层可行）⇒ recall 30% 上限 +14–20%、现实 +8–15%（D）；recall 10% ⇒ +3–7% 边缘 |
| 判决 | **值得做，先打点**：Step1 dump+离线（1 编译+2 跑+零 GPU 评估），go 门 recall≥30% 且 predicted-miss/层 ≤24 槽；Step2 FS-over-idle-stage（stage_+ring2 3GB，**不从 cache 扣槽**）~150–250 行、默认关、与 prefill 互斥 |
| 1.9 模拟（本席复算） | p_master 40 层 × 507 专家、总计数 **53,957,200**（每层总计数相等 1,348,930）；全局贪心 **67.37%**（每层 96–183 槽）；等配额 143 = **67.01%（−0.36pp）**；按层比例 67.00%；"按曲线贪心"≡全局贪心=当前 ⇒ **0 增益**（与情报-W 的 +0.4 点一致；真实杠杆=表质量 24.1 点/槽比例 16.6 点） |
| 1.11 量化 | 输出头 440.4MB（output.weight Q6_K 4096×131072×210/256，A 级）= +142 槽；R54 P1（staging 2048→1024，−50%）⇒ prefill **−11.0%**（回执 R54:5/20/33）⇒ +142 槽 ≈ **+4.7%**（对 2048MiB）~**+6.1%**（对 1512MiB 有效缓冲）⇒ **+4.7–6.1% prefill（D，需先补 stage-mib 4096 实测标定）** |

## 红臂（怎么变红）

- **"老师预测形态是 n-gram/表"** ⇒ 红：读码证明是 **h×下一层 router 权重**（`expert_source.hpp:241-245`），不用表（`情报-Z` §3 亦证）。
- **"FS 可以白拿"** ⇒ 红：老师 FS 槽从 cache 扣，实测 0.91–0.98（`DETAILS:1736-1744`）；我们必须用闲置 stage 才有别于老师，否则重演负结果。
- **"1.9 per-layer 配额能涨"** ⇒ 红：频次加权下全局贪心是上界，模拟 −0.36pp（等分）/0（曲线）。
- **"1.11 +142 槽值很多"** ⇒ 红：线性外推只有 +4.7–6.1%（D），且 head 在 dense_ 池里要拆；先补 `--stage-mib 4096` 实测。
- **"预测准确率已知"** ⇒ 红：我们无 trace；老师 prior 是 AMD/不同模型（C 级），必须打点。

## 关键决定

1. **不直接实现**：先打点（≤25 行，1 编译，2 跑）+ 离线评估；用 recall 决定是否做 FS-over-idle-stage。
2. **FS 设计要点写死**：**槽来自闲置 stage_+ring2（3GB≈968 槽），不从 cache 扣**；与 prefill borrow/stage 互斥；预测只改"在哪算"，不改 ids（质量门不受影响）。
3. **1.9 判定为否**：per-layer 配额无增益（曲线贪心=当前），不建议立项；1.11 只值 +4.7–6.1% prefill（D），且实现有拆 head 的风险，排后。
4. **先验说明**：老师 recall 27–34%/precision 8% 仅作 C 级参考；64GB 全驻留下"页预热"收益≈0 ⇒ 我们不能照抄 warm，只能做 FS 换腿。

## 未验 / 阻塞

未验：未编译/未跑（预测准确率正是要打点的内容）；老师数据是 AMD/不同模型、不同驻留 regime（C 级）；1.11 的 "+142 槽"需先跑 `--stage-mib 4096` 标定，head 是否真正闲置需逐行核（`output_` 只在最后 token logits 用，本轮未核）；dump 文件 ~84MB/条未实跑验证。
阻塞：无。零 GPU / 未编译 / 未改码 / 未 spawn 子代理。
