# 回执 · 情报R：Strata 的 MTP draft 成本结构 vs 我们的（2026-10-18）

## 1. 任务
lead 派：查"老师跑 3.8-Flash 时，MTP draft 每个 token 到底走不走 MoE"这一可能被漏掉的成本结构差异。要回答四问：① 老师 draft 前向是完整 MTP 块（attention+MoE）/ 只 attention+共享专家 / 别的廉价路径 + 每 draft token 成本结构；② 老师 verify 时 k 个候选 token 过 MoE，专家权重按"并集读一次算 k 行"还是"按 token 各读一次"；③ 我们的实现对照（`engine.cpp:3653-3693` draft、`main.cpp:432-517` 链式、`engine.cpp:2253-2293` CPU 专家分流）：draft 走不走 blk.88 MoE、verify 的 CPU miss 按并集还是按 token；④ 结论：两边差在哪一步（若我们按 token、老师按并集 = 实现缺陷可修，否则"命中率决定一切"）。纪律：零 GPU/零编译/零改码；file:line；区分读码/推断；不许 spawn；查不到明写。

## 2. 状态
**达成（4/4）**。T 树（`E:\strata` 0.1.40.2，`dec-pool2.txt:42`）+ U 树（`upstream-0.1.44.1`）两版本都查；四问均有 file:line 与成本/字节数；给出修法排序与未验项。

## 3. 结论（一句话）
**老师每个 MTP draft token 走完整 MTP 块（attention + router + 512 routed experts + shared expert + combine + head），不是跳过 MoE、也不是只看共享专家；差别不在"走不走 MoE"，而在老师把 MTP 块的 512 个 routed 专家一次性放进专用 VRAM 常驻区（675 MiB，`mtp.cpp:138-151/389-399`；日志 `dec-pool2.txt:23`），draft 的 MoE 是纯 GPU grouped 核、无 cache/CPU/磁盘；我们 blk.88 的专家混在主 cache（5,721 槽、命中 51.7%）里，每个 draft token 的 22 个 routed 专家按普通层走 GPU/CPU（~10.6 miss/draft ≈1.5ms，链 3 ≈4.6ms/轮），且 head 默认全词表 131,072 行=440.4MB（老师 106,299 行=212.9MiB）——这两项≈ +2.3ms/token，与 R40 的 +3.2% 吻合。验证并集问题：老师是 window 级 distinct 并集（`expert_source.cpp:257-444`，CPU/GPU 都按 distinct 专家一次读、算多行）；我们的 CPU 读**也是并集**（`cpu_experts.cpp:242-257/271-316/343-348` 按专家指针合并 uses），但**作用域只有一个 `moe()` 调用**（halves 时半窗）且**计数器 per-use**（`engine.cpp:2357`）——所以 496.7 miss/token 与老师 0.93 distinct/层不同口径；即使全部改成 window 并集，top-22/512 重叠仅 4.5%，最多省 4.5%，救不了 MTP。**

## 4. 证据（要点）
- **老师 MTP 必走 MoE（A 读码）**：`E:\strata\src\core\mtp.cpp:389-399` 注释 "MoE: router, the 512 resident experts, the shared expert, the combine, the write"；`:391` router GEMM、`:392-393` top-10、`:395` `moe_group_resident`、`:398` `moe_grouped_s2`、`:400-411` shared、`:412-414` combine。`skip/no_moe/nomoe/shared_only/only_shared` 在 `mtp.cpp` 全 0 命中（排除 b/c）。
- **512 专家常驻 VRAM（A 读码+日志）**：`mtp.cpp:138-151` `bytes = g.n_expert * BLOB`、`cudaMalloc(&experts_, bytes)`、读 `experts.bin`；`:239-241` 打印 VRAM 组成；日志 `dec-pool2.txt:23` "draft layer loaded, 836 MiB of VRAM (experts 675, dense 111)"；`:76` mtp 2.360 ms/round（75 rounds）。
- **draft 步法（A 读码）**：`MtpDrafter::draft` `mtp.cpp:552-591`：先 `round_exec_[T]`（`capture_round` `:472-495`，其中 `record_forward(T,-1)` `:487` 只写 K/V、在 `full=false` 时 `:354` 提前 return；`record_forward(1,ra)` `:492` 完整层跑接受行）；链式 `capture_step(j)` `:498-511` 每步 `record_forward(1,row)` = 单行完整 MTP 层（含 MoE）；`:582` 概率门 `min_p` 提前停链、`:586-589` 每步 stream sync；`:288` 明写 "the full layer runs one row at a time"。
- **draft head 子集（A）**：`mtp.cpp:258-274` 读 `draft_vocab.bin`、`gather_rows` 出 `dhead_`、打印 "draft head over %lld tokens"；`:424-432` `nv = sub ? n_dvocab_ : n_vocab_`；日志 `dec-pool2.txt:47` 106,299 tokens / 212.9 MiB。
- **老师 verify 并集（A 读码）**：`E:\strata\src\core\expert_source.cpp:257-444`：`first_of/distinct` `:289-300`；GPU groups 每 distinct 专家 `:307-343`；CPU `job_of[e]` 每 distinct 一个 `ExpertJobMulti`、多 token 激活压进 `jb.act[jb.nt]` `:403-424`；`run_split_multi` `:428-429`；`multi_misses += njobs`（distinct）`:441`；日志 `dec-pool2.txt:71` "CPU experts 0.93 distinct / 1.14 routed per layer"。
- **我们 draft 走 blk.88 MoE（A 读码）**：`PolyStrata\src\poly\engine.cpp:3940-3960`：`draft_input`→`block_attend`→`axpy/rms_norm/quantize`→`moe_router(L,1)`（:3948）→`moe_shared(L,n_layer_,1)`（:3950）→`moe(L,n_layer_,1)`+`moe_add`（:3952）→`moe_latent`（:3953）→`lin(head)`（:3958）；`:3971/:3976` 每个 draft step 前后 `cudaStreamSynchronize`；链式上限 `main.cpp:455-460/480-485`（≤3）。
- **我们 head 全词表（A 读码）**：`engine.hpp:206-208` `draft_rows()`；`:353` `guess_vocab()` 基类 0；仅 `qwen35.hpp:30` 覆盖 131072 ⇒ Nemotron 走全 `n_vocab_=131,072`；`output.weight` Q6_K = 440.402 MB（GGUF 静态）；`--draft-vocab` 在 `main.cpp:954`。
- **我们 CPU 专家并集读（A 读码）**：`cpu_experts.cpp:242-257`（`by=gkey`、`stable_sort`、`run[]`）、`:271-316`（fast 路径 `kq256_rows(... nu ...)` 一次解码权重算 nu 行；注释 R36 :198-202 "one decode of an expert's weights per run of its uses instead of one per use"）、`:343-348`（down 行：每 distinct 专家读一次行权重、内层对 uses 点积）；作用域=一个 `moe()` 调用的 `n_host` 列表（`engine.cpp:2203-2231` 建列表不 dedup，但 `cpu_experts` 内 dedup）；halves 走法见 `engine.cpp:3793-3795/3835-3844/3366-3412`。
- **我们计数 per-use**：`engine.cpp:2357` `stats_.cpu_experts += n_host`；`main.cpp:783-796` 的 decode-only cpu_us/miss 以它为分母。
- **U 树同构**：`upstream-0.1.44.1/src/core/mtp.cpp:308-326` 整块加载 `experts.bin`（512 routed）、`:1052` 注释 "the 512 resident experts"、`:1122/1125` 同样 `moe_group_resident`/`moe_grouped_s2`、`:507/622-690` draft_vocab 子集取舍注释。
- **量级核对**：我们每 draft token ≈ blk.88 dense 82.84 MB（GGUF）+ routed 专家读取（~11.4 cached×3.10MB + 10.6 CPU×3.10MB）+ head 440.4 MB ≈ 590 MB 读 + ~1.5ms CPU 腿 ≈ ~2.3ms/draft；链 3 ≈6.9ms/round ÷ 3.05 token/round ≈ **+2.3ms/token ≈ 75ms 的 +3.1%**，与 R40 **+3.2%** 吻合（R40 数字 B 级）。

## 5. 未验 / 查无
1. **查无**：T/U 两树 `mtp.cpp` 均无"跳过 routed MoE / 只走 shared"路径（`skip`=0、`no_moe`=0、`nomoe`=0、`shared_only`=0、`only_shared`=0）；`draft_vocab` 有（T 2 处、U 4 处）。
2. 我们"跨半窗重复读"的实际次数未实测（读码推断：halves 时每半窗各调一次 `moe()`）。
3. blk.88 层的实际 cache 命中率未单独量（按全局 51.7% 推算 10.6 CPU miss/draft，D 级）。
4. 1.585 GB 专用常驻后主 cache 命中率会掉多少未实测（按槽数比例估算 ~3 个点）。
5. R40 的 tokens/round、兑现、+3.2%/+0.9% 来自 lead（B），未复算；"①+②把 +3.1% 压到 +0.5%"为估算（D，未验）。

## 6. 落盘路径（绝对）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-R-MTP-draft成本结构对比-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IR-20261018-MTP-draft成本结构对比.md`
