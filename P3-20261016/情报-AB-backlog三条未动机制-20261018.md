# 情报-AB · backlog B 段三条未动机制（#3/#4/#5）的读取与落地评估（2026-10-18）

- **席位**：`scout-b`｜**零 GPU、零编译、不改码**；只读源码/git/docs/日志。
- **素材**：T = `E:\strata`（老师侧 0.1.40.2）；U = `upstream-0.1.44.1`（v0.1.44.1 + 端口）；bare clone `C:\sp3\upstream-hist\strata.git`（commits 可读；`blob:none` 的 `git show --stat` 需拉 blob，改用 `diff-tree --name-status`）；我们的 `PolyStrata\src\poly\engine.cpp`。
- **一句话判决**：**三条本体今天都不值得做——#3 只作用于"resident RAM complement（紧凑 pinned RAM 副本）"模式、#4 只作用于"device residency table + 异步 adaptive tier"、#5 只作用于"RAM 装不下全专家集"的低内存模式；这三种前提我们都不成立/不需要**（我们是 mmap 直读、全部专家页常驻、无紧凑 RAM 副本、decode 无 device 表）。**如果今天一定要动一处：做 #3 的"本机变体"——`follow_loop` 对已 pin 层把"先 memcpy 进 pinned 暂存再 H2D"改成"直接从 pinned 映射 DMA 进 VRAM"（去掉 3.1 MB/次换入的 host memcpy），~20–30 行、位精确、预期 D 级 0–6%（按 R43/R44 约 16 swap/token 粗估 ~2.5 ms/token）。**

---

## §0 结论表（机制 | 上游改动 | 我们有没有 | 移植面 | 位精确? | 本机预期收益（估值+等级） | 风险 | 今天做?）

| # | 机制 | 上游改动（sha / file:line） | 我们有没有 | 移植面 | 位精确? | 本机预期收益 | 风险 | 今天做? |
|---|---|---|---|---|---|---|---|---|
| 3 | `EXCHANGE_ROTATE`（交换缓冲"所有权轮转"，消掉"被换出专家写回 RAM 副本"的那一次 host memcpy） | `fa1d1d1e`（2026-10-14；改 `exchange_storage.hpp`/`expert_source.{hpp,cpp}`/`generate.cpp`/docs/tests）；U 树里的 uniform 版：`expert_source.cpp:3265-3323`（`reserve_exchanges` 只在 requested+uniform+完全 pinned/映射时启用）、`:3325-3329`（`exchange_buffer`）、`:3331-3343`（`stage_exchange`）、`:3371-3439`（`commit_exchanges`/`commit_flip` 的 ownership 交换）、`exchange_storage.hpp:14-120`（`commit()` 只挪 slot id，不 memcpy）；docs：`docs/EXCHANGE_ROTATION.md` 全文（含 A/B 表） | **没有也不需要**：我们的"RAM 副本"就是 **整个 GGUF 的只读映射**（`pin_expert_arena` `engine.cpp:1261` 起；R28-A3:82 "pinned 40 layers (59.06 GiB)… 0 refused"），被换出的专家**不需要写回任何地方**（映射里永远有）；`follow_loop`（`engine.cpp:1926-1958`）只拷**换入**专家：mapping→`s->pinned`（`:1941-1943`）→H2D（`:1944`） | **本体 0 行（N/A）**。**本机变体 #3′**：`follow_loop` 对 `pin_layer_[il]` 的层改为 `cudaMemcpyAsync(s->slot, L.exp_up + e*L.gu_stride, …)` 直拷（与 prefill `stage()` 的 pin 分支同一机制，`engine.cpp:3317-3323`），else 保持原路；~20–30 行 | **是**（字节相同；docs 的 1,024-token A/B IDs 也逐位相同） | **本体 0（D）**；#3′：每次 swap 省 ~3.1 MB host memcpy → 按 ~16 swap/token（R43/R44 量级）=50 MB/token ≈ **2.5 ms/token @20 GB/s（D 级；不带 id 风险）**；若 swap 少则 ≈0 | 本体 N/A 无风险；#3′ 低（有 `pin_layer_` 门 + 原路 fallback；`cudaMemcpyAsync` from registered read-only mapping 已被 prefill 验证） | **本体否；#3′ 可做（唯一候选）** |
| 4 | `ADAPT_NOWAIT` + 驻留表一致性补丁（host `host_res` vs device `d_res` 不一致的修复） | `3f2db5e2`（2026-10-14；改 `expert_source.{hpp,cpp}`、`verify.{hpp,cpp}`、`verify_kernels.{hpp,cu}`、`generate.cpp`）；细节：`generate.cpp:1373-1385`（`adapt_nowait()`/`adapt_lag()`）、`:8726/:8745`（pending 年龄与 gate）、`:8806-8815`（`apply_pending`）、`:12611-12629`/`:13811-13813`（NOWAIT/LAG 的调用点）、`:8758`（`ver.set_adapt_gate(&agate)`）；修复机制=每次 verifier launch 前的 hook 用 `res_patch` 小 kernel 把 changed entries 补进 device 表（整表上传兜底）、batch-groups 用 `set_always_publish`、`STRATA_ADAPT_CHECKRES=1` 回读校验 | **没有同类问题**：decode 不用 device 驻留表——`d_slots_` 只在 `moe_card()`/`card_moe()` 用（`engine.cpp:2556` 分配；`moe_card` 对非门控直接 `return false`，`:3275`；`card_moe` 还要求 all_cached），我们的 decode 计划全在 host `L.slot` 指针上；async swap 的槽指针**只在 `cudaEventQuery(s.done)==cudaSuccess` 之后才发布**（`follow()` `:1976-1982`），因此不存在"host 说不在、device 说在"的窗口 | **本体 0 行（N/A）**；只有将来给非门控模型开 `moe_card`/device plan 时才需要把 `res_patch` hook 一起搬 | 教师默认（等待）位精确；NOWAIT/LAG arm 本身可能改变答案（commit 记录 LAG=2 从 token 228 起变了）⇒ **是正确性修复，不是提速项**（hook −1.1%） | **本体 0（A 级事实：我们无此表）**；若照搬也不会更快（hook 本身 -1.1%） | 0（不适用） | **否** |
| 5 | resident RAM 副本跟随 decode 路由（低内存 resident 模式的 RAM 副本成员动态化） | `d7298883`（2026-10-14；改 `expert_source.{hpp,cpp}`/`generate.cpp`/docs/tests；commit 正文："Let the resident RAM copy follow decode routing (opt-in) … replace_resident now drops a Foresight slot for the outgoing expert while its RAM place is rewritten"）+ 端口基线：`generate.cpp:6960-7005`（resident RAM mode 的构造/约束：`--resident-experts`=`--mmap-experts --resident-cpu-experts`，`--resident-budget-gib` 限 RAM；adaptive tier 只在 GPU cache 与 resident RAM 副本之间交换）、`expert_source.hpp:552-580`（`pin_cache_complement`/预算）、`:132`（`choose_resident_keep_from`）、`expert_source.cpp:140`（`make_cache_complement_plan`）、`:2672-2790`（补集计划/预算/合法化）、`:3371-3439`（交换 commit）；基线行为见 `docs/DETAILS.md:168-169`（"a swap copies the evicted expert back from VRAM into the RAM place of the one that replaces it"） | **没有也不需要**：`pin_expert_arena` 把**整个专家映射**（59.06 GiB，40 个 MoE 层，0 refused）注册/常驻；全部 63.4 GB 专家字节已在 RAM（映射页常驻），**没有"哪些专家放 RAM"的成员选择**，没有紧凑副本、没有 `--resident-budget`；96 GB 机器装得下；CPU 腿是算力/线程受限（有效 ~21–23 GB/s），不是 RAM 缺页受限；31.7 GB sidecar 已判不可用 | **本体 0 行（N/A）**；若要抄，需要先建"紧凑 pinned 补集 + RAM 预算 + 成员替换 API"（`expert_source.cpp:140/220/2672-2790` + `generate.cpp:6950-7160`，数百行），为 0 收益 | 是（resident 模式与 mmap 模式同字节/同答案，`generate.cpp:6965-6966` "the answers are the plain mmap mode's"） | **0（A/D 级）**：前提是"RAM 装不下专家集"，我们是全集常驻；仅在低内存机器有意义 | N/A；若强上会占 RAM/增复杂度，且收益 0 | **否** |

---

## §1 #3 `EXCHANGE_ROTATE`（fa1d1d1e）读码结论

### 1.1 轮转的是什么

- **不是 VRAM 专家槽，也不是换入/换出队列，而是"exchange buffer 的归属（ownership）"**：一次 adaptive swap = 把 RAM 副本里的 `in` 传进 VRAM 里 `out` 的槽；旧路径随后要把 `out` 的字节 **memcpy 到 `in` 原来的 RAM 位置**（`docs/DETAILS.md:168-169`）。
- Rotation 的做法：**把 `out` 落地的临时交换缓冲直接"变成" `in` 原来的常驻位置，把 `in` 原来的 RAM 位置变成下一次的临时缓冲**——只改 slot id/指针归属，不 memcpy（`exchange_storage.hpp:86-113` 的 `commit()`；`expert_source.cpp:3371-3439` 的 `commit_exchanges`/`commit_flip`；docs 原文："**Rotation removes that last host copy: the temporary buffer becomes resident, and the old RAM slot becomes the next temporary buffer.**"）。
- 触发条件：`STRATA_EXCHANGE_ROTATE=1`（只有精确 "1" 开），且**仅 resident RAM 模式 + 每层 blob 等大 + 补集完全 pinned 且可映射 + 交换缓冲可 pinned/mapped**（U 版 `reserve_exchanges()` `expert_source.cpp:3285-3322` 的 `eligible` 检查；docs "Configurations without a resident complement do not use it"）。`fa1d1d1e` 把"等大 blob"限制放宽为 **per block size 的 spares**（commit 正文："ExchangeStorage now keeps its spares per block size (initialize_sized)… commit() refuses a spare of another size and an expert of another layer"）——本地 U 树快照里仍是 uniform 版（`initialize_sized` 在 U 树 **0 命中**），所以 `fa1d1d1e` 的 per-size 泛化未在本地快照中（**未验**其代码，只读 commit 正文）。
- 与 adaptive tier/驻留表的耦合：它**不改变**驻留策略/命中率，只改 swap 落地的"RAM 副本字节搬运方式"；仍由 adaptive tier 选 `in`/`out`（decay 计数、gain>1.5 等），仍要等 CPU/GPU 读者完成才 commit（docs："Rotation updates slot ownership … Compute readers and GPU transfers must still finish before the existing serialized commit; rotation adds no overlapping compute or new synchronization policy."）。

### 1.2 上游宣称与 A/B

- `fa1d1d1e` 正文（5090 32GB/i7-13700K/Swift 1.5 IQ3_S/128K/`--resident-experts`/`--adapt-every 1 --adapt-swaps 80`，blocking tier）：**161.6–165.1 → 184.9–188.7 tok/s**；同 cache 尺寸两臂 `--window-hashes` 434 窗口全同 ⇒ token-identical；`--serve` 23 轮：147.8/159.2/158.4 → 175.2/184.9/172.8；`--adapt-async 1`：166.0/170.1 → 185.6/182.5。机制解释（commit 正文）："**the copy path moved about 50 MiB of RAM per window**"（tier 每窗约 30 个 swap）。
- `docs/EXCHANGE_ROTATION.md` 的 native A/B（RTX PRO 6000 96GB，Q2_0，12,000 slots，16.19 GiB resident complement）：**104.90 → 113.44 tok/s（+8.1%）**，3,802 exchanges，**avoided host-copy payload 5,255,884,800 bytes**，1,024 输出 IDs 逐位相同（token-identical）。
- 即：**+8.1%~+14% 的前提是"有 resident RAM 副本 + 每窗几十个 swap"**。

### 1.3 我们有没有对应物

- **没有 resident RAM 副本**：`pin_expert_arena()`（`engine.cpp:1261`，`STRATA_PIN_ARENA_GIB=59.1`）把整个专家映射 **只读注册**（R28-A3:82 `pinned 40 layers (59.06 GiB) of 88, 0 refused; took 2 ms`），专家字节留在 **GGUF mmap 的页**里；被换出 VRAM 的专家**不需要"RAM 位置"**（映射里天然有）。
- 我们的 swap 路径（`follow_loop engine.cpp:1926-1958`）：`memcpy(s->pinned, L.exp_up/exp_gate/exp_down + e*stride, …)`（`:1941-1943`）→ `cudaMemcpyAsync(s->slot, s->pinned, …, H2D, swap_st_)`（`:1944`）——**只拷"换入"专家，且不写回换出专家**。所以 **EXCHANGE_ROTATE 所消除的那一次"换出专家写回 RAM 位置"的 memcpy，在我们的架构里根本不存在** ⇒ 本体 **移植面 0、收益 0**。
- 但它的**目标**（减少 swap 路径的 host 拷贝）在我们这里有一个等价可做的点：既然 40 层映射已注册（`pin_layer_`），`follow_loop` 可以**直接从映射 DMA 进 VRAM**（跳过 `s->pinned` 这一跳）——prefill 的 `stage()` 已经这么干（`engine.cpp:3317-3323`：`if (pin) cudaMemcpyAsync(dst + at, L.exp_gate + e*L.gu_stride, …, cudaMemcpyHostToDevice, …)`）。这就是 **#3′**：改 `follow_loop` 一处，`pin_layer_[il]` 走直拷、否则走原路；~20–30 行；字节完全相同 ⇒ 位精确；预期 = 省掉每次 swap 的 3.1 MB host memcpy（读+写）。按 R43/R44 臂 ~4,000 swaps/256 token ≈ 16 swap/token ⇒ ~50 MB/token host 拷贝 ≈ **2.5 ms/token（20 GB/s 假设，D 级）**；若 swap 量低（如交付臂 0–1k）则 ≈0。

---

## §2 #4 `ADAPT_NOWAIT` + 驻留表一致性补丁（3f2db5e2）读码结论

### 2.1 表不一致是什么

- 两个表：**host 的 `host_res`**（哪层哪专家在 VRAM；CPU pool/plan 用它）与 **device 的 `d_res`**（doorbell publish 的每层"是否全 resident"标志读它）。
- `adapt()` 立即把被换出的专家在 `host_res` 标成非驻留；`d_res` 要等本轮拷贝落地后在 `apply_pending` 才更新（`generate.cpp:8806-8815`）。当 tier **不等待**（`STRATA_ADAPT_NOWAIT=1` 或 `STRATA_ADAPT_LAG>1`）时，**下一个 verify window 在 `d_res` 更新之前就开跑**：`doorbell_publish_res` 读 `d_res`，把已被换出的专家仍当 resident ⇒ 该层"全部 resident"⇒ **不发布该层 activation**；而 pool 依据 `host_res` 把它当 CPU 专家计算，却读不到 activation（或用旧 activation）⇒ 层-window 计算错误/mismatch。
- 修复（同一 commit）：**每次 verifier launch 前**跑一个 pre-launch hook，用 `res_patch` 小 kernel 按 changed-entry 列表把 device 表补好（列表溢出/不可 host-map 时整表上传），`apply_pending` 整表上传失败时把条目留给 hook；`--batch-groups` 用 `set_always_publish`（并关掉 device plan）；`STRATA_ADAPT_CHECKRES=1` 回读比对、统计 mismatch；`STRATA_ADAPT_EVICT_SYNC=0` 恢复旧行为。改动文件：`expert_source.hpp/.cpp`、`verify.hpp/.cpp`、`verify_kernels.hpp/.cu`、`generate.cpp`（diff-tree 名单）。
- 实测（commit 正文）：RTX 5090/IQ2_XS/`--expert-cache 12000`/32K/600 tok：**NOWAIT=1 34 of 222 windows (2470 entries), 1 layer-window → 0/0/0**；LAG=2 **55 of 223 (3222 entries，每 swap 都错) → 0/0/0**；LAG=2 可复现，修复把答案从 **token 228** 起改对；hook 代价 **−1.1%**（chat）/ −1.2%（32K），即**正确性代价**不是收益；默认（tier 等待）不装 hook，tokens/first-token logits 与旧版相同。

### 2.2 与 `ADAPT_OVERLAP` 的区别

- `ADAPT_OVERLAP`（我们已判否：Windows −2.6%）= **让 adaptive tier 的工作与窗口/循环重叠**（异步 tier 的调度）；`ADAPT_NOWAIT`/`ADAPT_LAG` = **不等拷贝落地就开下一个窗口**（进一步把等待挪后）；#4 是这条路的**正确性补丁**（在不等的情况下用 device 表 patch 防止 CPU/GPU 认知不一致）。三者是同一异步谱系：overlap（重叠工作）→ nowait（跳等待）→ patch（修表）。
- 结论：**#4 是"正确性修复"而非提速项**（hook 本身 −1.1%）；它的价值只在"已经采用 NOWAIT/LAG 且要避免错答"时成立。

### 2.3 我们有没有同类不一致

- **没有**：我们的 decode 不用 device 驻留表。`d_slots_` 只在 `moe_card()` 用（`engine.cpp:2556` 分配；`moe_card` 对非门控模型直接 `return false`——`engine.cpp:3275`；`card_moe` 还要求 all_cached），而 `moe_card/card_moe` 在本模型 decode **不会成功**；decode 的"哪些专家 GPU 算"完全由 host 的 `L.slot[e]` 决定（`moe()` 里），没有 host/device 两套表。
- 我们的 swap 异步但不产生 stale：`follow()` 只在 `cudaEventQuery(s.done)==cudaSuccess` 后才把 `layers_[..].slot[e] = s.slot`（`engine.cpp:1976-1982`）；换出时 `follow()` 立即 `given=nullptr`（`follow` 的 `put` 段，`engine.cpp:2020-2060`）。⇒ **host 视野单调、从不超前**，不存在"host 认为不在、device 仍认为在"的窗口。
- 移植面 **0 行**（除非将来给非门控开 device plan；那时才需要 `res_patch` hook 一起搬）；风险 0；收益 0。

---

## §3 #5 resident RAM 副本跟随 decode 路由（d7298883）读码结论

### 3.1 它的机制

- 仅在 **resident RAM 模式**（`--resident-experts` = `--mmap-experts --resident-cpu-experts`；或 `--resident-cpu-experts` + 静态 `--expert-profile`）下存在：把"GPU cache 装不下的专家"从 `experts.bin` 读进一块 **紧凑的 pinned RAM 副本**（`pin_cache_complement`；`--resident-budget-gib` 限预算），adaptive tier 只在 **GPU cache 与这个 RAM 副本之间**交换（`generate.cpp:6960-7005`；基线行为 `docs/DETAILS.md:168-169`："a swap copies the evicted expert back from VRAM into the RAM place of the one that replaces it"）。
- `d7298883` 让 **RAM 副本的"成员"跟随 decode 路由**（opt-in），而不是只按初始 profile/借槽计划固定：commit 正文 "**Let the resident RAM copy follow decode routing (opt-in) … replace_resident now drops a Foresight slot for the outgoing expert while its RAM place is rewritten (#2134's copies read those places)**"。改动文件（diff-tree）：`expert_source.hpp/.cpp`、`generate.cpp`、tests、docs。
- **注意（未验）**：本地 U 树快照里 `replace_resident` **0 命中**（`grep replace_resident` 全树 0），说明这个端口可能不在本地 0.1.44.1 快照里（或改了名）；我们只能读 commit 正文与 resident 模式基线代码，**具体实现代码未读到**。

### 3.2 我们现状与空间

- 我们 `pin_expert_arena()`（`engine.cpp:1261`；`STRATA_PIN_ARENA_GIB=59.1`）把**整个专家映射**注册/常驻（R28-A3:82：40 层 59.06 GiB，0 refused，`took 2 ms`），不是"紧凑副本"，也没有"哪些专家放 RAM"的选择——**全部专家字节都在 RAM**（映射页）。
- 我们机器 96 GB，专家集 63.4 GB（模型文件 70.1 GB）在 RAM 内；31.7 GB sidecar 已判不可用；CPU 腿是**算力/线程受限**（有效 ~21–23 GB/s，我们 `情报-T`/`情报-W` 已给），不是"RAM 装不下→读文件/SSD"的问题。
- ⇒ #5 **没有空间**：它的价值只在"RAM 装不下全集、需要挑一部分放 RAM"的低内存模式；我们的 target 是 96GB 且全集常驻。**移植面 = 0 行（不需要）**；若强上（建紧凑副本 + 预算 + 替换 API）是数百行、纯增复杂度，本机预期 0（D 级）。
- 位精确性：resident 模式本身与 mmap 模式同字节/同答案（`generate.cpp:6965-6966`），但与我们无关。

---

## §4 三条的依赖前提 vs 我们的现实（判定依据）

| 机制 | 依赖的上游前提 | 我们的现实 | 判定 |
|---|---|---|---|
| #3 EXCHANGE_ROTATE | resident RAM **紧凑副本**（fully pinned/mapped complement）+ 每窗几十个 swap；`docs` 明说"Configurations without a resident complement do not use it" | 我们是 **mmap 只读全集**（无紧凑副本；换出专家无需写回）；swap 只拷换入 | **无对应损失，0**；#3′ 可借用"少一次 host memcpy"的目标 |
| #4 ADAPT_NOWAIT+patch | **device residency table**（`d_res`/doorbell 每层全 resident 判断）+ 异步 adaptive tier（NOWAIT/LAG） | decode 无 device 表（`moe_card` 非门控直接 false）；swap 槽指针在 event 落地后才发布 | **无此不一致，0** |
| #5 resident RAM follow | **RAM 装不下全集**的低内存 resident 模式 + 紧凑副本 + 预算 | 96 GB，全集常驻（59.06 GiB pinned mapping）；CPU 腿是算力/线程问题 | **无空间，0** |

另：**老师 0.1.40.2 的 dec-pool2 run 也没有这三条**——T 树 grep：`EXCHANGE_ROTATE`=0、`ADAPT_NOWAIT`=0、`resident RAM`=0；且该 run 用的是 **expert arena** 模式（`dec-pool2:26-30` "expert arena read unbuffered … loaded 46.84 GiB at 2.64 GiB/s"），而 `docs` 说 arena 模式"every expert is in the arena already … a round has no exchange and no RAM move"（U `generate.cpp:9028-9031`）⇒ **三条机制对老师那次 24.3 ms/round 的账没有贡献**（它们是 0.1.44.1+ 面向 resident 模式/异步 tier 的优化）。

---

## §5 要不要今天做？

- **#3/#4/#5 本体：不做**（前提不成立；照搬的预期收益都是 0，还要引入新数据结构/一致性风险）。
- **唯一值得今天的：`#3′`（follow_loop 直拷变体）**：
  1. 改 `follow_loop`（`engine.cpp:1926-1958`）：`if (!L.gated && pin_layer_[s->il]) cudaMemcpyAsync(s->slot, L.exp_up + e*L.gu_stride, …) + 同 down`；`gated` 或未 pin 时走原 `s->pinned` 路径；`cudaEventRecord(s->done)` 不变。
  2. 自证：`--expert-swaps` 开、跑 45 token；对比 `swap_dmas_/swap_dma_bytes_`、decode tok/s；ids 45/45；`swap_error_` 空；不触发 pin fallback（日志 `pinned … 0 refused`）。
  3. 预期：按 R43/R44 换入量（~16 swap/token）≈ 50 MB/token host memcpy 消除 ≈ **~2.5 ms/token（D 级，±）**；若换入少则 ~0。位精确（同一段字节，`cudaMemcpyAsync` vs 先 memcpy 再 H2D）。
  4. 风险：低（`pin_layer_` 门 + 原路 fallback；`stage()` 已证明 mapping→device 直拷可行，`engine.cpp:3317-3323`）；唯一注意：`cudaMemcpyAsync` 的源是 registered read-only mapping（已注册；未 pin 层绝不走这条）。
- **#4 相关的“正确性”提醒**：若将来 R1/R2 决定给非门控开 `moe_card`/device plan，请把 `res_patch` hook（commit `3f2db5e2` 的文件清单：`verify_kernels.cu` 的 `res_patch`、`verify.cpp` 的 pre-launch hook）一并移植；现在我们没有 device 表，不用动。

---

## §6 查无 / 未验（明写）

1. **`fa1d1d1e` 的 per-size 泛化代码未在本地 U 树**：`initialize_sized` 在 U 树 **0 命中**（U 快照是 uniform 版 rotation；commit 正文已读，泛化实现未读）。
2. **`d7298883` 的代码未读到**：U 树 `replace_resident` **0 命中**，该端口可能不在本地 0.1.44.1 快照里/改名；本件对 #5 的机制描述来自 commit 正文 + resident 模式基线代码（`generate.cpp:6960-7005`、`expert_source.hpp:552-580`、`docs/DETAILS.md:168-169`），**实现细节未验**。
3. **`git show --stat <sha>` 在 bare clone 上会尝试从 GitHub promisor 拉 blob 并失败**（`Recv failure`）；改用 `diff-tree --no-commit-id --name-status -r` 成功（本件文件清单即此法）。
4. **#3′ 的收益是估算（D 级）**：swap/token 取自 R43/R44 日志的累计 swap 计数（~4,000/256 token）与每 swap 3.1 MB；未实测 host memcpy 的实际带宽/是否与 H2D 重叠；`cudaMemcpyAsync` 直拷在 pin 层的实际吞吐未测（prefill 的 `stage()` 直拷是既有证据，但 decode swap 路径未跑过）。
5. **`ADAPT_NOWAIT` 的 hook 代价**（−1.1%）来自 commit 正文（RTX 5090/IQ2_XS/3 pairs），非本机。
6. **我们 `follow_loop` swapped-in 路径的 `pin_layer_` 覆盖**：`pin_expert_arena` 只 pin 了 40 个 MoE 层中的前 N 个（预算 59.1 GiB；日志 40 层）；blk.88（MTP）等未 pin 的层必须走 fallback（本件已写进 #3′ 的条件）。

## §7 落盘 / 复现

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AB-backlog三条未动机制-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IAB-20261018-backlog三条未动机制.md`
```powershell
# #3
git -C 'C:\sp3\upstream-hist\strata.git' show -s --format='%B' fa1d1d1e
git -C 'C:\sp3\upstream-hist\strata.git' diff-tree --no-commit-id --name-status -r fa1d1d1e
Get-Content -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\docs\EXCHANGE_ROTATION.md'
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\core\expert_source.cpp' -Pattern 'reserve_exchanges','stage_exchange','commit_flip','exchange_storage_'
# #4
git -C 'C:\sp3\upstream-hist\strata.git' show -s --format='%B' 3f2db5e2
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\program\generate.cpp' -Pattern 'adapt_nowait','adapt_lag','apply_pending','set_adapt_gate'
# #5
git -C 'C:\sp3\upstream-hist\strata.git' show -s --format='%B' d7298883
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\program\generate.cpp' -Pattern 'resident RAM mode','--resident-experts','pin_cache_complement'
# 我们
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\engine.cpp' -Pattern 'pin_expert_arena','follow_loop','cudaMemcpyAsync(s->slot','if (pin)','moe_card','d_slots_'
```
