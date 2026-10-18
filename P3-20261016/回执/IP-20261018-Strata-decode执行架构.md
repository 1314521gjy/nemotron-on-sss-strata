# 回执 · 情报P：Strata decode 执行架构逐行拆（三机制）— 2026-10-18

## 1. 任务
lead 派：逐行拆 Strata 的 decode 执行架构，专盯三样（零 GPU，只负责 Strata 侧）：
1) **专家图**：有没有"提前知道下一层要哪些专家"的机制（历史命中图/连续层相关/训练表？），file:line；
2) **边搬边算**：DMA/PCIe 拷贝与计算如何重叠、谁与谁并行、同步原语、double-buffer/环/门铃，file:line；
3) **CPU-GPU 按比例原地计算**：`pcie-frac`/`PCIE_BALANCE` 是"整专家搬"还是"同一专家的行按比例分给 CPU/GPU 原地算"；读 0.1.40.2 与 0.1.44.1（`04be241a` 的 `PCIE_BALANCE`）——给出粒度 + 伪代码。
每节：机制是什么 → file:line → 伪代码/时序（≤10 行）→ 解决什么问题（对应老师账哪一段）；最后一句：三样里哪一样是我们完全没做的、且最可能解释"老师 24.3 ms/轮 vs 我们 86 ms/token"。纪律：零 GPU/零编译/零改码；每条 file:line；区分读码/推断；不许 spawn；查不到明写"已查 X/Y/Z 无命中"。

## 2. 状态
**达成（3/3 + 结论）**。两棵树分开写：T=`E:\strata`（老师侧，`dec-pool2.txt:42` engine 0.1.40.2；树内 `engine/BUILD.json` 是 0.1.43.1 的 exe、CMake tag 陈旧 0.1.20，均已在正文标注）、U=`upstream-0.1.44.1`（v0.1.44.1，HEAD 296b59d2）。三节都有伪代码/时序（≤10 行）与账本对位。

## 3. 结论（一句话）
**①专家图：U 树有（`RouterLookahead` 用下一层 router 权重对本层 MoE 输入预跑 top-k → `warm` 文件页 / `ForesightSwap` 预拷 VRAM 槽；不是学习/共现图），T 树（老师 0.1.40.2）没有，我们 poly 完全没有；②边搬边算：T 树是"每层两张 capture 图、以 router 处 doorbell 环切开、GPU 半段先发、CPU pool 与 GPU 常驻/PCIe 半段并行、PIe DMA 在 copy 流上"，其 `ms_wait/ms_pool` 就是老师账本 wait for rings/pool 的直接来源；③按比例：两版都是"整专家、按数量比例（`m=(nmiss*pcie_num)>>8`）、按 routing order 取最后 m 个"，**不是按行、不是同一专家两边原地算**；`PCIE_BALANCE` 与 `04be241a` 在本地两棵树查无（0 命中 / git 对象不存在），无法确认；我们 poly 的 `--pcie-frac` 同粒度同规则且已有 direct 模式 ⇒ 没理解错、只是交付没开。三样里我们完全没有的是①，但它解释不了老师 24.3（老师那版也没有①）；24.3 vs 86 的第一解释量是 CPU 专家数/token 差 11×（45 vs 496.7；命中 96.52% vs 51.7%；top10×48 vs top22×40），三样中唯一直击这个数的是③。**

## 4. 证据（要点）
- **老师账本（A）**：`E:\strata\_staging\dec-pool2.txt:71-76/81/83-84/87`：verify window `wait for rings 12.378 / pool 7.269 / host 0.101 / commit 0.037 ms/round`；CPU experts 0.93 distinct/1.14 routed per layer；pool multi gate/up 4.923 + quantize 0.016 + down 1.945，13.4 GB/s，CPU pool call 7.159 ms/round；dispatch run 6.888；adaptive tier 818 swaps；**pcie experts 0.16/layer（share 90/256）**；mtp 2.360；per token 9.240 ms；R4 命中 **115669/119845=0.9652**。
- **①（U 树）**：`include/strata/core/expert_source.hpp:241-246`（定义与"只预热页，不改算谁"）、`:263-266`（depth 1..4）、`:247-307`（类接口）；`src/core/expert_source.cpp:2220-2237`（start）、`:2239-2252`（submit，layer+1）、`:2349-2458`（run：`bf16_rows_dot_multi` 下一层 router→topk→`src_->warm` :2446→`fs->predict` :2447-2448）、`:2266-2346`（lookahead stats：recall/precision/votes 覆盖 miss 与 PCIe share）；warm 实现 `:2153-2192`（Windows `PrefetchVirtualMemory` :2160-2190，POSIX `madvise(WILLNEED)`，io 预读线程 `io_enqueue` :2173-2174）；ForesightSwap `include/strata/core/foresight_swap.hpp:1-107`（独立 copy stream、depth 4、budget 32、`take`/`note_miss`/`predict`）。
- **①（T 树没有）**：`RouterLookahead`/`foresight`/`CS-T` 在 `E:\strata\src` 全量 0 命中；`lookahead` 仅 2 处（`src/prefill/prefill.cpp`，KV/slot 前瞻）；`warm` 15 处均为注释/warmup，无 `ExpertSource::warm` 函数；老师用静态 profile+adaptive tier+设备侧 hit：`src/program/generate.cpp:1786-1790/2358/2391/2401/3991-3996/4068`。
- **②（T 树）**：`src/core/session.cpp:186-187`（每层两图、router 处切开）、`:606-745`（pre[0]→收环→hits(Launch)/pool→y_miss 拷回→combine→post[l]→pre[l+1]；`ms_host` :735-744 明说这段是 GPU idle）、`:675-678`（"THE GPU'S HALF GOES FIRST…"）；`src/core/layer.cpp:370-385`（doorbell_publish + ring last）；`src/core/verify.cpp:796-844`（逐层 spin 等环 → pool → 发布 flagA/flag → `ms_wait`/`ms_pool`）、`:558/587/595`（图内 `wait_flag_ge(m_flagA_/m_flagB_/m_flag_)`）；`src/core/expert_source.cpp:284-367`（plan 先建、`P.fetch` :367 "the copy engine, beside the CPU's work"）；`src/core/pinned.cu`、`kernels/cpu/pool.cpp:31-90`（pool workers 按物理核/SMT 感知 pin）。
- **②（U 树新增）**：ForesightSwap filler 线程 + I/O 预取（`expert_source.hpp:711-718/730-744`，`expert_source.cpp:2153-2192`）；GpuPlanSink 三种 PCIe 送达模式（`expert_source.hpp:309-334`，pcie_mode 0=DMA staging/1=直接 device alias 读 mapped/2=graph 内 copy kernel）。
- **③**：U 树 `src/core/expert_source.cpp:3720-3721`（"Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU"）、`:3726-3745`（nmiss）、`:3770-3776`（`m=(nmiss*pcie_num)>>8`、PCIE_MIN1）、`:3781-3815`（`miss_rank >= nmiss-m` 且 `src->pinned` 才 fetch）、`:3831-3845`（整 blob `bb`、`P.ptr2` staging 或 device_alias）；T 树 `src/core/expert_source.cpp:284-285/301-302/306-357/367`；T 树输出 `src/program/generate.cpp:4070-4073`；老师实测 `dec-pool2.txt:75` 0.16/layer share 90/256。
- **我们 poly**：`lookahead`/`foresight`/`expert_graph`/`PrefetchVirtualMemory` 全 0；`prefetch` 14 处 = `map_pages` 整文件页 touch 注释；`--pcie-frac` 同粒度同规则（`src/poly/engine.cpp:2253-2283`，`m_uses=(n_host*pcie_num)>>8`，取 routing order 最后 m 个），并有 `direct` 模式原地读 pinned 映射（`:2265-2293`）。
- **查无（明写）**：`PCIE_BALANCE`/`pcie_balance`/`PCIE balance` 在两棵树全树（src/include/docs/tools，排除 .git/third_party/bench results）**0 命中**；`04be241a` 在两仓库都不是可解析对象（`git cat-file -t` fatal）⇒ 无法读该提交；`expert_graph`/`cooccur`/`co-occur` 均 0 命中。

## 5. 未验 / 边界
1. **`PCIE_BALANCE` / `04be241a` 无法确认**：本地无该提交、无该字符串；若它真按行分，是本地源码没有的新机制（明写"未查到"）。
2. T 树版本口径：老师日志写 engine 0.1.40.2，但 `E:\strata\engine\BUILD.json` 是 0.1.43.1、`CMakeLists.txt` project VERSION 0.1.20（陈旧 tag）；本件把"T 树源码特征"按 0.1.40.2 处理并在 §1 标注（若老师实际跑的是 0.1.43.1，①/② 的"没有/有"结论需以 0.1.43.1 源码再核；但 T 树内没有 0.1.44.1 的 lookahead 是确定的）。
3. 老师账本里 `wait for rings` 在 T 树有两个相似打印（`generate.cpp:4051` round 版、`:4219` token graph 版）；本件用 `:4051` 版（与 dec-pool2 行文一致）。
4. "哪一样解释 24.3 vs 86"里的 11× 分解是按老师 `dec-pool2` 与我们 R34/R28 两组 A 级读数静态对比（不同模型几何：老师 10×48、我们 22×40），未做同模型对照；结论是"机制都改常数、不改 CPU 专家数"。
5. 未运行、未编译、未改码；所有结论来自读码+已存日志。

## 6. 落盘路径（绝对）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-P-Strata-decode执行架构-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IP-20261018-Strata-decode执行架构.md`
