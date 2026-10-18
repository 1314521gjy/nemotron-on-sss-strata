# 情报-P · Strata decode 执行架构逐行拆：专家图 / 边搬边算 / CPU-GPU 按比例（2026-10-18）

- **席位**：`scout-b`｜**零 GPU、零编译、不改码**；只读源码/日志。
- **源（两个版本，口径分开写）**：
  - **T 树 = `E:\strata`（老师侧）**：`_staging/dec-pool2.txt:42` 明写 `session is up (engine 0.1.40.2)`；注意树内 `engine/BUILD.json` 写 `version 0.1.43.1`（该目录里现成的 exe 更新；源码特征与本日志时代相符），`CMakeLists.txt:11` 的 project VERSION 仍是 0.1.20（陈旧 tag，勿据此判版本）。
  - **U 树 = `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1`**（`git describe` = `v0.1.44.1`，HEAD `296b59d2`）。
- **答案速览（先给结论）**：
  1. **①专家图**：**U 树有，T 树没有，我们也没有**。U 树是 `RouterLookahead`（在 CPU 线程上拿**下一层的 router 权重**对**本层 MoE 输入**再算一次 top-k，提前 `warm` 文件页 / 预拷 VRAM 槽）＋ `ForesightSwap`（VRAM 槽由独立 copy stream 预填）。**不是学习出来的图、不是历史共现表**——每层现算，默认只看 1 层（可 1–4 层），"只预热页，不改变算谁"（`expert_source.hpp:241-246`）。
  2. **②边搬边算**：T 树是**每层两张 capture 图、以 router 处的 doorbell 环切开**：`pre` 跑 GPU 前半段→响铃；host 收环→跑 CPU pool；**GPU 那一半（常驻/PCIe 专家）先发**，与 CPU 半段并行；PCIe 拷贝走 copy engine 自己的流，图内靠 flag 等待。U 树在此之上又加了 `ForesightSwap` filler 线程 + I/O 预取线程。
  3. **③CPU-GPU 按比例**：**不是按行、不是"同一个专家的行分给两边"**。两版都是**整专家**：按 `pcie_num/256` 的比例，从**本层 miss 的 distinct 专家按 routing order 取最后 m 个**，整专家 blob 搬去 GPU（或"在原地被 GPU 直接读"，见 3 种 pcie_mode），GPU 按整专家分组核计算。我们 poly 的 `--pcie-frac` **同粒度、同"取最后 m 个"规则**（`engine.cpp:2253-2283`），没有理解错；且我们也有 `--pcie-mode direct`（原地读 pinned 映射，`engine.cpp:2287-2293`）。
  4. **"哪一样能解释老师 24.3 vs 我们 86"**：**不是这三样**。老师那版（0.1.40.2）连①都没有；老师的账里 **CPU 专家只有 0.93 distinct/层**（48 层≈45/token，`dec-pool2.txt:71`）、pool 7.159 ms/token、R4 命中 **115669/119845=96.52%**（`:87`）；我们 **496.7 miss/token、51.7% 命中、CPU 腿 65–72 ms/token** ⇒ 差距的第一解释量是 **CPU 专家数/token 差 11×（命中率/驻留选择 + top-k=22×40 层 vs 10×48 层的模型几何）**；三样里只有 ③ 能直接砍这个数（老师自己只用了 0.16 专家/层，share 90/256，说明它靠命中率而不是搬）。

---

## §1 老师账本（A 级，先钉死，后面每节都回指它）

`E:\strata\_staging\dec-pool2.txt`（Qwen3.8-Flash-Next IQ3_S，4080S，decode 200 token = 1847.9 ms = 108.23 tok/s，`:79`）：

| 段 | 数值 | 出处行 |
|---|---|---|
| `verify window` | **wait for rings 12.378 · pool 7.269 · host 0.101 · commit 0.037 ms/round**；CPU experts 0.93 distinct / 1.14 routed per layer | `:71` |
| `pool multi` | gate/up 4.923 · quantize 0.016 · down 1.945 ms/round；**13.4 GB/s** over rows；CPU pool call **7.159 ms/round** | `:72` |
| `dispatch` | plan 0.135 · activation quantize 0.120 · jobs 0.009 · **run 6.888** ms/round | `:73` |
| `adaptive tier` | 818 experts swapped（every 4 rounds, 0.175 ms/round） | `:74` |
| `pcie experts` | **0.16 distinct experts/layer** over PCIe（share **90/256** of the misses） | `:75` |
| `mtp` | **2.360 ms/round** drafting（75 rounds），MTP prompt 11.9 ms | `:76` |
| per token | **9.240 ms/token**（WALL: embed/layers/head/sample）；round = 2.63 token | `:81/:67` |
| CPU pool | **7.159 ms/token over 48 layers**（76 positions, 3648 dispatches）；phases wait-park 0.000 / **drain 6.886** / re-park 0.000 | `:83/:84` |
| R4 命中 | **115669 of 119845 = 0.9652**；0 admitted / 4176 refused；cache 100% full | `:87` |

**账本闭合**：`12.378+7.269+0.101+0.037+2.360 = 22.145`，加尾部 ≈2.17 ⇒ **24.31 ms/round** ✓；÷2.63 = **9.24 ms/token** ✓。

---

## §2 机制①：专家图（提前知道下一层要哪些专家）

### 2.1 它是什么

**U 树（0.1.44.1）有，T 树（0.1.40.2）没有，我们的 poly 也没有。** U 树是两层机制，全部围绕"下一层"：

1. **`RouterLookahead`（CS-T）**：CPU 池开始算第 `l` 层时，把该层的 MoE 输入 `x`（host float）交给一个**常驻辅助线程**；该线程用**第 `l+1` 层的 `ffn_gate_inp` router（BF16 host 拷件）**对 `x` 算 logits（`bf16_rows_dot_multi`），取每 token top-k，去掉 GPU cache/RAM 副本已持有的，剩下的调 **`ExpertSource::warm(l+1, want)`** 去预热文件页；同时若开了 `ForesightSwap`，还调 `fs->predict()` 把预测到的专家**预拷进该层的 VRAM 槽**。
   - 依据："when the CPU pool starts layer `l`, a worker thread applies layer `l+1`'s router (BF16, host copy) to layer `l`'s MoE input - the residual stream changes little from one layer to the next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the source to `warm` the rest, so their pages are on the way while layer `l` computes. **A prediction only warms pages: it never changes which experts are computed or how.**"（`upstream-0.1.44.1/include/strata/core/expert_source.hpp:241-246`）
   - 深度：`STRATA_IO_PREFETCH_DEPTH` 默认 1 层，可 1–4 层（越深越不准，`expert_source.hpp:263-266`）。
   - 实测/度量开关：`STRATA_LOOKAHEAD_STATS=1` 只测量，打印 **recall / precision / votes top-4/8/12 covers 多少 miss 和多少 PCIe share**（`src/core/expert_source.cpp:2266-2346`，打印在 `:2322-2346`）。
2. **`ForesightSwap`**（opt-in `STRATA_FS_SLOTS>0`）：每层每卡**专家 cache 之外**的少量 VRAM 槽，由**自己的 copy stream + filler 线程**预填；来源两类：① `note_miss()`——刚 miss 的专家（给下几个窗口用）；② `predict()`——lookahead 的预测（`STRATA_FS_AHEAD=1` 默认随槽开启）。池子若查到该专家在 slot 里且拷贝已落地，就把它当**普通 GPU group / 类 cache-hit** 算；未落地就置若罔闻继续给 CPU。安全性：slot 只有在"可能读过它的窗口都完成"后才复用（`STRATA_FS_DEPTH` 默认 4 个完成的 verify window；每卡每完成窗口 budget 32 次拷贝）。
   - 依据：`include/strata/core/foresight_swap.hpp:1-107`（slot 结构/`take`/`note_miss`/`predict`/depth/`fill_loop`）；dispatch 里 `fs->take(...)` 在 `src/core/expert_source.cpp:3801-3803`、`fs->note_miss(...)` 在 `:3804`；lookahead 的 `fs->predict(...)` 在 `:2447-2448`。
3. **不是"学习出的依赖图/共现表"**：全树 `expert_graph`/`cooccur`/`co-occur`/`expert_plan` 均 **0 命中**（见 §6）；它就是"下一层自己的 router 在 CPU 上再算一次"（近似，不需训练）。

**T 树（0.1.40.2，老师实际跑的那版）没有①②**：`RouterLookahead` 0 命中、`foresight` 0 命中、`CS-T` 0 命中；`lookahead` 只有 2 处在 `src/prefill/prefill.cpp`（prefill 的 KV/slot 前瞻，不是专家）；`ExpertSource::warm()` 不存在（`warm` 的 15 处命中全是注释/测试 warmup/引擎 warm 启动）。老师只有**静态 profile 排名的 cache** + **自适应换入**：
- `--expert-profile` → "24576 ranked pairs"（`dec-pool2.txt:19`）、"expert cache 11428 slots"（`:34`）、"pre-filled 11428 of 11428 slots from the profile in 1.1 s (20312 MB/s)"（`:39`）、"R4 hit path ON - resident experts are computed on the GPU"（`:40`）；
- adaptive tier：818 swaps（`:74`）、代码 `src/program/generate.cpp:3991-3996`（每 `adapt_every` 轮在独立线程 ranking/提拷贝）、打印 `:4068`；
- token graph "decided on the device"：`src/program/generate.cpp:2358/2391/2401`（全常驻时的设备侧 hit 路径）。

### 2.2 伪代码（U 树，≤10 行）

```
每层 l: CPU pool 开始算 l 的专家
  aux 线程 RouterLookahead::run():
    x = 第 l 层的 MoE 输入（host float，≤8 token）          # submit(l, x)
    for dj in 0..depth-1:                                     # 默认 depth=1
      logits = bf16_routers[l+1+dj] · x                       # 下一层的 router 权重！
      want = topk(logits, k) 去掉 host_res>=0（已在 GPU）      # 只留真 miss
      src->warm(l+1+dj, want)                                 # PrefetchVirtualMemory/madvise/io 预读
      if dj==0: fs->predict(l+1+dj, want)                     # 预拷进该层 VRAM 槽（copy stream）
主线程: 继续算本层 CPU 专家；GPU 同时算常驻/PCIe 半段
```

### 2.3 它解决什么（对应老师账里哪一段）

U 树的 `warm()`/`predict()` 目标是把**文件/VRAM 取数**藏到**当前层 CPU 计算**后面——对应老师账本里 `pool 7.269`（CPU 专家读写）与 `wait for rings 12.378` 中"专家没到位"的等待；`STRATA_LOOKAHEAD_STATS` 的 recall/precision 就是给"下一层预测值不值得"做判据。**BUT：老师那版（0.1.40.2）账本里没有它**，所以它不能用来解释老师的 24.3 ms/轮；它是 0.1.44.1 针对"专家 miss 太多/太贵"的新增机制（对我们这种 497 miss/token 的场景才是对症的）。

---

## §3 机制②：边搬边算（DMA/copy 与计算重叠、每层流水）

### 3.1 它是什么（T 树 = 老师版，正是账本来源）

**每层两张 capture 图，以 router 处的 doorbell 环切开**：

- 图结构："**TWO GRAPHS PER LAYER, SPLIT AT THE ROUTER.** `pre` ends with the doorbell rung; the host then runs the CPU pool on what it published; `post` combines those experts with THIS layer's weights."（`src/core/session.cpp:186-187`；capture 主体 `:163-245`）。
- **门铃（doorbell）**：`moe_route()` 在捕获序列里把 `x / ids / weights` 写进 mapped pinned buffer，**最后写 ring**（"THE RING IS LAST, so a host that sees it knows every byte above is in place"）——`src/core/layer.cpp:370-385`；ring kernel 在 `src/kernels/cuda/elementwise.cu`（`layer.cpp:111` 注释指向）。
- **host 收环**：`verify.cpp`（decode 的 T-token window 路径）逐层 `while (*seq < want) { _mm_pause(); ... }`，每 2 ms 用 `cudaStreamQuery` 兜底（`src/core/verify.cpp:796-818`）；然后调 CPU pool（`:825-827`），再发布 **flagA（GPU plan）**（`:832-841`）与 **flag（CPU share 就绪）**（`:842`），并统计 `ms_wait`（`:843`）/`ms_pool`（`:844`）——**这就是账本 `wait for rings` 与 `pool` 的直接来源**。同文件 `:850` 等 copy 流同步（窗口尾）。
- **图内三处等待**（captured wait_flag 节点）：`:558-595`
  ```
  wait_flag_ge(flagA, ring)   // pool 已发布 GPU plan（哪些专家 GPU 算）
  wait_flag_ge(flagB, ring)   // PCIe share 已在 staging（DMA）或已在 mapped
  wait_flag_ge(flag,  ring)   // CPU 的那一份已写进 mapped rows
  ```
- **GPU 半段先发**：pool 在把"哪个专家给 GPU"的 plan 发布之后才开算 CPU 半段；GPU 侧拿到 flagA 即开始算常驻 + PCIe 组（`src/core/expert_source.cpp:284-367`：plan 先建、`P.fetch` 发 PCIe DMA，注释 `:367` 原文 "**the copy engine, beside the CPU's work**"）；token 路径更直白："**THE GPU'S HALF GOES FIRST, SO IT RUNS WHILE THE CPU DOES ITS HALF.** `Launch` only enqueues…"（`src/core/session.cpp:675-678`、`:680-702`）。
- **一串到底的顺序**（`session.cpp:606-745`）：`pre[0]` 先发 → for l: 收环 → hits(Launch)/pool → `y_miss` 拷回 pinned → hits(Combine) → `post[l]` → `pre[l+1]`；`ms_host`（老师账 `host 0.101` 同类）测"从看到环到发完 pre[l+1]"的 **GPU 空转窗口**（`:735-744`）。
- **拷贝侧**：CPU 结果走 **pinned** `y_miss`（`session.cpp:587-602`，注释解释了为什么必须 pinned：pageable 的 `cudaMemcpyAsync` 会同步 bounce）；PCIe 拷贝走 **`copy_` 流 + flagB**（`verify.cpp:850` 等它）；U 树里再加 `ForesightSwap` 的 per-card `card_stream` + 每槽 `cudaEvent_t ev`（`foresight_swap.hpp:36-101`）。
- **T 树有没有 double-buffer/环/门铃**：有门铃 + `pre/post` 两图；**prefill** 另有 N-slot 有序流式 ring（`src/prefill/prefill.cpp:121-134/164/230-314`，之前情报-D 已拆）；decode 这里的"环"是**每层 doorbell 环**，不是专家 ring。

### 3.2 时序图（≤10 行，老师 decode 一轮/一层）

```
GPU:  pre[l]  ──────────────► ring(l) ──► [等 flagA] 算常驻/PCIe 专家 ──► [等 flag] combine ──► post[l]
host: 收 ring(l) → 发布 plan(flagA) → 跑 CPU pool（本层 miss 专家）→ 拷 y_miss → 发 post[l] → 发 pre[l+1]
                 ↑ wait for rings 12.378/round 的来源         ↑ pool 7.269/round 的来源（此段 GPU 只跑自己的半段）
copy: PCIe DMA（P.fetch）在 copy 流上与上面并行；flagB 落地前 GPU 只算常驻组
```

### 3.3 它解决什么（对应账本哪一段）

- `wait for rings 12.378`：GPU `pre`（dense/attn/router）的墙钟，host 只能在环上等（GPU 的这半段没有可供 CPU 并行的工作，除非 window 多 token 分摊）。
- `pool 7.269`：CPU 专家池（0.1.40.2 里 ~45 专家/token，13.4 GB/s）；**GPU 的常驻/PCIe 半段在这段时间里并行**（这是"边搬边算"真正省下的部分）。
- `host 0.101`：环后到 pre[l+1] 之间的纯 driver 时间（GPU 空转），老师刻意把它计成"GPU idle"（`session.cpp:735-744`）。
- `commit 0.037`：窗口提交/采样；`mtp 2.360`：drafter 并行段。

**U 树（0.1.44.1）在②上又加了**：`ForesightSwap` filler 线程（独立 copy stream 预填 VRAM 槽，depth/budget 控制）；ExpertSource 的 I/O 预取线程池（`io_ahead_`、整 blob `pread`、Windows `PrefetchVirtualMemory`/POSIX `madvise(WILLNEED)`：`expert_source.hpp:711-718/730-744`；实现 `expert_source.cpp:2153-2192`）；`GpuPlanSink` 的三种 PCIe 送达模式（§4）。

---

## §4 机制③：CPU-GPU 按比例原地计算（pcie share / "PCIE_BALANCE"）

### 4.1 它到底是什么粒度：**整专家、按数量比例、按 routing order 取最后 m 个**（不是按行）

**U 树（0.1.44.1）原文与代码**（`src/core/expert_source.cpp`）：
- `:3720-3721`：*"plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works. **Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.**"*
- `:3726-3745`：先在 entries 里数 `nmiss`（distinct、且不在 host_res/cache、不是 peer/helper 持有）。
- `:3770-3776`：`pcie_ok = pcie_num>0 && src->pcie_layer(layer)`；**`int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;`**（+`STRATA_PCIE_MIN1` 至少 1 个）。
- `:3781-3815`：遍历 distinct 专家；`miss_rank >= nmiss - m` 的（= **routing order 里最后 m 个 miss**）进入 fetch 列表，但**仅当 `d.src->pinned(layer,e)` 为真**（页面已 pin 才可 DMA），否则仍退回 CPU。
- `:3831-3845`：每个 fetch 对应一个 **整 blob**（`bb = lay.blob_bytes(layer)`）；`P.ptr2[q] = P.pcie_mode != 0 ? device_alias(...) : staging + q*bb`，即：**整个专家矩阵由 GPU 按整专家分组计算**，不是拆行。
- 计划先发：`GpuPlanSink`（`include/strata/core/expert_source.hpp:309-334`）文档写明"groups of entries that share a blob; the blob is a VRAM slot or a pinned host blob read over PCIe"；`fetch` 负责"DMA 到 staging 槽并在落地时给 GPU 信号"。
- **三种 PCIe 送达模式**（`expert_source.hpp:330-333`）：`pcie_mode = 0` copy-engine 拷进 staging；`1` grouped kernel **直接按 device alias 读 mapped arena**（原地读，不经拷贝）；`2` copy kernel 在 graph 里 stage。

**T 树（0.1.40.2，老师版）同粒度**（`src/core/expert_source.cpp`）：
- `:284-285` 同样注释 "Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU"；`:301-302` `pcie_ok` / `m = (nmiss * d.pcie_num) >> 8`；`:306-357` fetch 列表/staging；`:367` **`if (P.fetch) P.fetch(P.ctx, dma_src, …);   // the copy engine, beside the CPU's work`**；`:333-357` VRAM groups vs PCIe groups 分开登记。
- 老师版打印：`src/program/generate.cpp:4070-4073`：`pcie experts %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)`；实测 `dec-pool2.txt:75` = **0.16 distinct experts/layer（share 90/256）**——小到可忽略。
- 老师版也有 3 种 pcie mode：`generate.cpp:950` (`--pcie-mode`)、`:2568/:3834` `ver.set_pcie_mode(dma=0/direct=1/kernel=2)`。

### 4.2 伪代码（两版相同，≤10 行）

```
pool_dispatch_multi(layer, ids[n_tok*k], host_res):
  distinct = 按 routing order 首次出现去重; nmiss = distinct 中 host_res<0 且非 peer/helper 的个数
  发布 plan 到 mapped sink（flagA）                       # GPU 拿到就能先开算
  m = (nmiss * pcie_num) >> 8                             # pcie_num = frac*256，0..256
  for q in distinct:
     if 在 cache:            kind=0（GPU，cache slot 指针）
     elif fs 槽命中:         kind=0（Foresight slot 指针，落地才算）
     elif 在最后 m 个 miss 里 且 src->pinned(e): kind=1（PCIe：DMA 到 staging，或 pcie_mode=1 直接 device alias）
     else:                   kind=-1（CPU pool）
  fetch(dma_src): 在 copy 流上发 PCIe DMA，落地后置 flagB
  CPU 线程算 kind=-1 的专家；GPU 并行算 kind=0/1；最后由 graph 里 flag 等待 + combine
```

### 4.3 它解决什么 + 对"我们理解错了"的回答

- 它解决的是"**本层 miss 的数个专家，选一部分整专家送到 GPU 算**"，把 CPU 腿按 `pcie_num/256` 缩短；代价是 PCIe H2D 带宽（老师：`pcie-frac` 的默认值由启动时 probe 的 H2D 带宽决定，见 T 树 `generate.cpp:1187-1203`）。
- **"按行比例、同一个专家两边原地算"在可查源码里不存在**：`PCIE_BALANCE` 在 **T 树与 U 树全树（src/include/docs/tools，排除 `.git`/`third_party`/bench results）均 0 命中**；commit **`04be241a` 在两个本地仓库都不是可解析对象**（`git cat-file -t 04be241a` → `fatal: Not a valid object name`）。**⇒ 无法确认有该机制；若它真按行，那是一个本地源码里没有的新东西，需要拿到那版提交再判（明写：未查到）**。
- 我们 poly 的 `--pcie-frac` **同粒度、同规则**：`src/poly/engine.cpp:2253-2283`——"the last m = (this block's misses * frac) of them, in routing order, go over PCIe instead of to the CPU: their experts are DMA'd out of the pinned mapping into the staging slots by a stream of their own, and the compute stream reads the slots as cache slots"；`m_uses = (n_host * pcie_num) >> 8`（`:2272-2273`），distinct 按 routing order 取最后 m 个（`:2274-2283`），**且我们还有 `direct` 模式**（`:2265-2293`："read in place: no slot and no copy, the kernels take the pinned mapping's own pointers"）——对应 Strata 的 pcie_mode=1。故**我们没把粒度理解错**；差别只在默认值（我们 0、且交付未开）、staging 槽数（16 vs Strata 最多 64 fetches/窗口）和 plan 发布/等待的细节。

---

## §5 三样里哪一样最能解释"老师 24.3 ms/轮 vs 我们 86 ms/token"

**先把两个数摊开（A 级）**：

- 老师：24.31 ms/round ÷ 2.63 token = **9.24 ms/token**；其中 wait rings 4.71、**pool 2.76**（7.159 ms/token 里的一部分）、host 0.04、commit 0.01、MTP 0.90、尾 0.83（每 token）。
- 我们（R34/R28 实测）：**86.1 ms/token**，其中 **CPU 专家 65.3 busy + 6.6 交接 = 71.9 ms/token**（83.5%），其余 14.2 = 真 router 等待 + lm_head + 同步/图缝隙。

**差距的乘法分解**（同一台 4080S、同类 MoE 模型）：
| 量 | 老师 | 我们 | 倍数 |
|---|---|---|---|
| CPU 专家 distinct/token | 0.93/层 ×48 ≈ **45** | **496.7 miss/token** | **11×** |
| 每个 CPU 专家耗时 | 7.159ms/44.6 ≈ **160 µs** | 144.7 µs（R28） | **≈1×** |
| CPU 腿/token | **7.16 ms** | **71.9 ms** | **10×** |
| R4 命中率 | **96.52%**（115669/119845） | **51.7%**（R34） | — |
| cache 槽/专家总数 | 11,428 / 24,576 = 46.5% | 5,721 / 20,480 = 27.9% | — |
| top-k × 层数 | 10 × 48 = 480 次 lookup/token | 22 × 40 = **880** | 1.83× |

⇒ **第一解释量 = CPU 专家数/token 差 11×**，而它来自三处与我们三机制无关的东西：**模型几何（top-10×48 vs top-22×40）**、**驻留选择（profile 预填 + adaptive tier + 96.52% 命中）**、**cache 相对占比（46.5% vs 27.9%）**。三机制（预热图 / 重叠 / 按比例搬）都只改"等与搬的常数"，**没有一个能把每 token 要算的 CPU 专家数降下来**——除非 ③ 把 11%–40% 的 miss 搬给 GPU（老师自己只搬 0.16 个/层，因为它已经不需要）。

**回答"哪一样是我们完全没做的、且最可能解释"：**
1. **"完全没做" = ①专家图（RouterLookahead + ForesightSwap）**：poly 全源码 `lookahead`/`foresight`/`expert_graph`/`PrefetchVirtualMemory` 均 **0 命中**；我们的"预取"只有 `map_pages()` 的整文件页 touch（`engine.cpp` 14 处 `prefetch` 注释即此），**没有"下一层要什么"的预测**。
2. **但①解释不了老师的 24.3**：老师那版（0.1.40.2）自己也没有①（grep 0 命中；`warm` 函数不存在）——①是 0.1.44.1 才加的。若目标是"复现老师 24.3"，先复制的是**命中率/驻留选择**（profile 预填 + adaptive tier + 设备侧 hit 路径）。
3. **若只从三样里选一样来补、并且直接砍我们的 CPU 腿：选 ③**（整专家按 `pcie_num/256` 比例搬给 GPU），因为它是唯一直接减少"CPU 要算的专家数"的机制；我们已有同粒度实现但交付里 `--pcie-frac 0` 没开（且老师的数据说明：当命中率高时③几乎不用）。②我们已有部分（GPU cached 专家与 CPU 并行），缺的是老师那种"每层 pre/ring/pool/post 两张图 + GPU 半段先发"的流水（我们 poly 的 MoE 段掉出图、CPU 腿占 83.5% wall）。

**一句话交付**：**我们完全没做的是①"专家图/下一层预取"（但老师 0.1.40.2 也没有，它解释不了 24.3）；24.3 vs 86 的第一解释是 CPU 专家数/token 差 11×（命中率 96.52% vs 51.7%）；三样里唯一直击这个数的是③整专家按比例搬（不是按行，我们没理解错、同粒度已实现但没开），②是老师版的"每层 doorbell 预/后图 + GPU 半段先发"，我们只做了其中 GPU 常驻专家与 CPU 并行的部分。**

---

## §6 "查无命中"清单（明写，供复核）

在这两个位置全量搜（无 `-First N` 截断）：
- T 树 `E:\strata`：`src/`、`include/`、`docs/`、`tools/`（排除 `.git`、`third_party`、`bench/results`、`.venv`）
- U 树 `upstream-0.1.44.1`：`src/`、`include/`、`docs/`、`tools/`（同上排除）

| 关键词 | T 树 | U 树 | 说明 |
|---|---|---|---|
| `PCIE_BALANCE` / `pcie_balance` / `PCIE balance` | **0** | **0** | lead 提到的"0.1.44.1 的 PCIE_BALANCE"在本地两棵树都不存在 |
| `04be241a` | 仓库**不是可解析对象**（git cat-file fatal） | 同 | 无法读该提交 |
| `expert_graph` | **0** | **0** | 没有叫这个名字的图 |
| `cooccur` / `co-occur` | **0** | 未单列（U 树 `expert_graph` 同 0） | 没有共现表 |
| `RouterLookahead` / `ForesightSwap` / `CS-T` / `foresight` | **0** | 有（§2） | 老师在 0.1.40.2 没有① |
| `lookahead` | **2**（仅在 `src/prefill/prefill.cpp`，指 prefill 的 KV/slot 前瞻） | 49（含专家 lookahead） | — |
| `ExpertSource::warm` | **不存在**（`warm` 15 处全是注释/warmup） | 有（`expert_source.cpp:2153-2192`） | 老师版没有页预热接口 |
| 我们的 poly `lookahead/foresight/expert_graph/PrefetchVirtualMemory` | — | — | **均 0**；`prefetch` 14 处 = `engine.cpp` 的页 touch 注释 |

---

## §7 落盘 / 复现

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-P-Strata-decode执行架构-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IP-20261018-Strata-decode执行架构.md`
- 快速复核：
```powershell
# 老师账本
Get-Content -LiteralPath 'E:\strata\_staging\dec-pool2.txt' | Select-Object -Skip 70 -First 17
# ① T 树没有 / U 树有
Get-ChildItem -LiteralPath 'E:\strata\src' -Recurse -File -Include *.cpp,*.cu,*.hpp | Select-String -Pattern 'RouterLookahead','foresight','CS-T' -SimpleMatch
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\include\strata\core\expert_source.hpp' -Pattern 'RouterLookahead','warm','lookahead'
# ② 每层两图/门铃
Select-String -LiteralPath 'E:\strata\src\core\session.cpp' -Pattern 'TWO GRAPHS PER LAYER','THE GPU''S HALF GOES FIRST','ms_host'
Select-String -LiteralPath 'E:\strata\src\core\verify.cpp' -Pattern 'm_flagA_','m_flagB_','ms_wait','ms_pool'
# ③ 整专家按比例
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\core\expert_source.cpp' -Pattern 'pcie_num','last pcie_num/256','P.ptr2'
Select-String -LiteralPath 'E:\strata\src\core\expert_source.cpp' -Pattern 'pcie_num','P.fetch'
```
