# 情报-R · Strata 的 MTP draft 成本结构 vs 我们的（2026-10-18）

- **席位**：`scout-b`｜**零 GPU、零编译、不改码**；只读源码/日志。
- **T 树** = `E:\strata`（老师侧 0.1.40.2；`_staging/dec-pool2.txt:42` `engine 0.1.40.2`）；**U 树** = `upstream-0.1.44.1`（v0.1.44.1）。**我们** = `PolyStrata\src\poly`。
- **一句话结论**：**老师每个 MTP draft token 走完整 MTP 块（attention + routed MoE + shared expert + head），不是跳过 MoE、也不是只看共享专家；差别不在"走不走 MoE"，而在老师把 MTP 块的 512 个 routed 专家一次性放进专用 VRAM 常驻区（675 MiB），draft 的 MoE 是纯 GPU grouped 核、无 cache/CPU/磁盘；我们 blk.88 的专家混在主 cache（5,721 槽、命中 51.7%）里，每个 draft token 的 22 个 routed 专家按普通层走 GPU/CPU，链长 3 时每轮多付 ~4.5–7 ms CPU 腿——这正是 R40 "+3.2%/-0.9%" 的量级。验证并集问题：老师是并集（distinct 专家一次读、算窗口内所有行），我们的 **CPU 读**也是并集（`cpu_experts()` 内按专家指针合并），但**统计口径是 per-use**、且合并作用域只有一个 `moe()` 调用；不过 top-22/512 重叠只有 4.5%，并集最多省 4.5%，**这不是 MTP 不划算的主因**。**

---

## §1 老师（T 树 0.1.40.2）：MTP draft 前向走不走 MoE？

### 1.1 结论：**(a) 走完整 MTP 块**，包括 attention + router + 512 routed experts + shared expert + combine + head

- **MoE 在 draft 前向里，明写在 `MtpDrafter::record_forward`**（`E:\strata\src\core\mtp.cpp`）：
  - `:330-388` attention（K/V append → Q/K norm/rope → `qsa_decode_attn_batch` → o_proj → hyper-connection）；
  - `:389-399` **MoE**：注释原文 `---- MoE: router, the 512 resident experts, the shared expert, the combine, the write`；`:391` router GEMM（`bf16_gemv_fp32_mmvf` → 512 logits）；`:392-393` top-10；`:395` `moe_group_resident(...)`（按 token 的 top-10 分组，指向**常驻 VRAM** 的 `experts_`）；`:398` `moe_grouped_s2(...)`（grouped resident expert 核）；`:400-411` **shared expert**（`mlp.shared_expert.gate/up/down`）；`:412-414` `moe_combine`；`:415` 写回 residual；
  - `:417-432` head：`:424-432` `sub = dhead_ != nullptr`，`nv = n_dvocab_`，`native_mmvq(head…, nv, T)` → **draft head 只用 draft_vocab 子集**（日志 `draft head over 106299 tokens (212.9 MiB)`，`dec-pool2.txt:47`）。
- **不是 (b)/(c)**：`mtp.cpp` 全量搜 `skip` / `no_moe` / `nomoe` / `shared_only` / `only_shared` **均 0 命中**；MoE 调用无分支可跳过（`moe_group_resident`/`moe_grouped_s2` 是必走的）；唯一"廉价化"是 draft head 的词表子集（`draft_vocab.bin`，`:258-274`）。
- **但 routed 专家全部常驻 VRAM**（这是关键）：
  - `:138-151`：注释 `---- the 512 routed experts, one blob each`；`bytes = g.n_expert * BLOB`；`cudaMalloc(&experts_, bytes)`；从 `rt_dir/experts.bin` 整块读入（失败则 "the 512 experts do not fit in VRAM"）；`:239-241` 打印 `draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f)`。
  - 实测日志：`dec-pool2.txt:23` `strata mtp: draft layer loaded, 836 MiB of VRAM (experts 675, dense 111)`；`:76` `mtp 2.360 ms/round drafting (75 rounds), MTP prompt 11.9 ms, 1049 MiB of VRAM`（836+212.9 head≈1049 ✓）。
- **draft 的步法**（每 draft token 一次完整 row forward，非批量）：
  - `MtpDrafter::draft(T, tokens, p, a, drafts, …)`（`mtp.cpp:552-591`）：先跑 `round_exec_[T]`（`capture_round` `:472-495`）——其中 `record_forward(T, -1)`（`:487`）**只写 K/V**（`record_forward` 在 `full=false` 时 `:354 return true`，不过 MoE）；然后 `mtp_select` 出第一个 draft；再 `record_forward(1, row)`（`:492`）对接受位置 a 的**完整层**跑一行；`mtp_select` 出 draft[0]（`:494`）。
  - 链式续 draft：`capture_step(j)`（`:498-511`）每个 j 一个新 capture 图，`record_forward(1, row)` = **单行完整 MTP 层（含 MoE）**；`draft()` 里 `for j … while pj >= min_p`（`:582`），每步 `cudaGraphLaunch` + `cudaStreamSynchronize`（`:586-589`）——**按概率门提前停链**，所以不是固定 5 步。
  - 每步全层的注意力 scratch 只够 1 行：`record_forward` `:288` 明写 "the full layer runs one row at a time (its attention scratch is sized for one)"。

### 1.2 每 draft token 的成本结构（读码 + 日志）

| 读的权重 | 量 | 出处 |
|---|---|---|
| MTP 块 dense（fc_embedding/fc_hidden/QKV/O、hyper-connection、norms） | **111 MiB**（日志） | `mtp.cpp:239-241`；`dec-pool2:23` |
| router | 4096×512 GEMM | `mtp.cpp:391` |
| routed experts | **top-10 / 512**，全部在 **VRAM 常驻**（675 MiB 总池；BLOB≈675MiB/512≈1.32 MiB/专家 ⇒ ~13 MiB/draft） | `mtp.cpp:138-151/389-399`，`dec-pool2:23` |
| shared expert（gate/up/down，Q8_0） | 一份 layer 级 | `mtp.cpp:154-155/400-411` |
| draft head | **106,299 行**（draft_vocab 子集）≈ **212.9 MiB** | `mtp.cpp:258-274/424-432`，`dec-pool2:47` |
| **实测** | **2.360 ms/round ÷ 2.30 draft/round ≈ 1.0 ms/draft**（76 rounds，`drafts accepted 124/175`） | `dec-pool2:67/76` |

⇒ **老师 draft 的 MoE 不产生任何 cache 查询、CPU 专家、页缺失**；成本就是 GPU 上一次 router GEMM + top-10 常驻 grouped GEMM + shared expert + head 子集。

---

## §2 老师 verify（普通 MoE 窗口）：专家权重按并集读一次、算 k 行

- `expert_pool_dispatch_multi`（T 树 `E:\strata\src\core\expert_source.cpp:257-444`）：
  - 先按 **distinct 专家**建表：`first_of[]`/`distinct[]`（`:289-300`）——同一专家在一个窗口里出现多次只记一次；
  - **GPU 侧**：每个 distinct 专家成为一个 group（`P.ptr[groups]=…`，`:307-343`），其所有 entries（token×k 的使用）进 `P.dst/P.tok` ⇒ GPU grouped 核按**一个专家 blob 算所有行**；
  - **CPU 侧**：`job_of[expert]` 把 distinct 专家映射到一个 `ExpertJobMulti`（`:403-418`），同一专家的多个 token 激活压进同一个 job（`jb.act[jb.nt]=…`，`:419-424`），随后 `run_split_multi` 多 token 内核跑（`:428-429`）——**一次读权重、算多行**；
  - 计数器：`d.multi_misses += njobs`（`:441`，**distinct**），`d.multi_entries` 是 entries（uses）；日志 `dec-pool2.txt:71` 的 `CPU experts 0.93 distinct / 1.14 routed per layer` 正是这两个数的比（dedup 比 1.23×）。
- ⇒ **老师 verify 是并集口径**。**但**这只适用于主模型 verify；draft 的 MTP 块专家是各自常驻、不进这套 cache/池（§1）。

---

## §3 我们的实现（poly）对照

### 3.1 我们的 draft 每个 token 走不走 blk.88 的 MoE？

**走，而且是完整 MoE（router + 22 routed 专家 + shared + combine + head），与老师同构 (a)；差别在专家不在专用常驻区。**

- `Engine::draft_step`（`PolyStrata\src\poly\engine.cpp:3940-3960`）：
  ```
  draft_input() → block_attend(attention) → axpy → rms_norm → quantize
  → moe_router(L,1)                // :3948  router
  → moe_shared(L, n_layer_, 1)     // :3950  shared expert
  → if (moe(L, n_layer_, 1)) moe_add(L,1)   // :3952  routed experts（22 of 512）
  → moe_latent(L,1) → axpy → rms_norm(d_head_) → lin(head, …)   // :3953-3959 head
  ```
  - `cudaEventSynchronize(ev_)` 在 `moe()` 里（`engine.cpp:2196-2197`）；每个 draft step 前后各一次 `cudaStreamSynchronize`（`:3971`、`:3976`）；链式上限 `main.cpp:455-460/480-485`（≤3 个 draft，`kBlockWindow=4`）。
- **blk.88 的 512 专家不常驻**：它们和其它 40 个 MoE 层共用同一个主 cache；`fill_expert_cache()` 是全层共享计数器（此前 task-6/情报-B 已量），命中 51.7%（R34）⇒ 每个 draft token 的 22 个 routed 专家里约 **10.6 个走 CPU**（其余 GPU cached），CPU 侧 ~10.6×145 µs ≈ **1.5 ms/draft**；链 3 ⇒ **~4.6 ms/轮**级。
- **head 默认全词表**：`engine.hpp:206-208` `draft_rows() = opt_.draft_vocab >= 0 ? opt_.draft_vocab : guess_vocab()`，`guess_vocab()` 基类返回 0（`:353`；只有 `qwen35.hpp:30` 覆盖为 131072），Nemotron 没覆盖 ⇒ `draft_rows() = n_vocab_ = **131,072**` ⇒ 每 draft token **读全输出头 440.4 MB（Q6_K `output.weight`）**（GGUF 静态字节）；`--draft-vocab` 可改（`main.cpp:954`），默认 -1。
- 估算：每 draft token ≈ dense blk.88（GGUF `mtp_block` 16 张量 = **82.84 MB**）+ routed 专家读取（~11.4×3.10 MB cached + 10.6×3.10 MB CPU）+ head **440.4 MB** ≈ 590 MB 读取 + 1.5 ms CPU 腿 ⇒ ~2.3 ms/draft；3 draft ≈ 6.9 ms/round；若 3.05 token/round ⇒ **约 +2.3 ms/token ≈ 75 ms 的 +3.1%**——和 R40 的 **+3.2%** 对得上（B 级：R40 数字来自 lead）。**MTP 赚的 token 基本被"每 draft 一次完整 MoE 层 + 全词表 head"吃掉。**

### 3.2 我们验证 k 个 token 时，CPU 专家 miss 是按并集还是按 token？

**两层要分开看，答案是"读是并集（作用域=一个 moe() 调用），计数是 per-use"：**

- **构建列表**：`moe()` 在 `engine.cpp:2203-2231` 对每个 token 的每个 routed 专家逐 use 建 `cached[]`/`host[]`——**这里不按 distinct 去重**；
- **CPU 计算**：`cpu_experts()`（`cpu_experts.cpp:187-373`）进来后按**专家指针**排序/分组：
  - `by=gkey(experts[i])`（gate 或 up 指针），`stable_sort`、`run[]` 标出每个专家的 uses 段（`:242-257`）；
  - gate/up：`for i in [i0,i1)` 对该专家的**所有 uses** 逐行 dot（`:293-316`）；fast 路径 `kq256_rows(..., nu, ...)` 一次解码权重算 nu 行（`:271-283`，注释 R36 "one decode of an expert's weights per run of its uses instead of one per use"）；
  - down：`for u in n_run`（每个 distinct 专家）**一次读 down 行权重**，内层 `for i in runp[u]..runp[u+1]` 对所有 uses 点积（`:343-348`）；
  ⇒ **一次 `cpu_experts()` 调用内，专家权重按 distinct 读一次、算多行**（并集）。
- **但作用域不是整个 window**：`window()` 用 `halves_ = pick_way(t)`（`engine.cpp:3793-3795`，`pick_way` `:3835-3844`）决定 whole / halves 两种走法；halves 时 `run()` 把窗口拆两半、每半调一次 `moe(L, il, m)`（`m` = 半窗，`:3366-3412`）；whole 时一次 `moe(L, il, n)`（`:3462-3470`）。⇒ **跨半窗重复的专家会读两次**；同一 `moe()` 内（≤半窗/整窗，最多 8 token）是并集。
- **统计口径**：`stats_.cpu_experts += n_host`（`engine.cpp:2357`）与 `cpu_us/miss` 的 miss 都是 **use** 计数（不是 distinct）；老师打印的 0.93 是 **distinct/层/窗口**。所以**我们的 `miss/token=496.7` 不能直接和老师的 `0.93 distinct/层` 对比**；折算成同口径：我们 ~12.4 use/层，重叠 ~4.5% ⇒ ~11.8 distinct/层；老师 0.93 distinct/层 ⇒ 仍差 ~12.7×（与之前 11× 结论一致，口径修正不改变量级）。

---

## §4 结论：差在哪一步？

1. **不是"我们按 token 读、老师按并集读"的实现缺陷。**
   - 老师 CPU/GPU 都是 window 级 distinct 并集（§2）；我们的 **CPU 读**也在 `cpu_experts()` 内按 distinct 指针合并（§3.2），只是作用域是一个 `moe()` 调用（whole 或半窗）且**计数器按 use**。
   - 就算把并集扩到整个 window，top-22/512 的 miss 重叠只有 ~4.5% ⇒ **最多省 4.5% CPU 读**，不足以让 MTP 翻正。
2. **真正的结构差 = MTP block 的专家驻留方式（这决定 MTP 划不划算）：**
   - 老师：MTP block 的 **512 个 routed 专家专用 VRAM 常驻**（675 MiB，`mtp.cpp:138-151/389-399`），MoE 走 resident grouped 核；draft 无 cache 查询/CPU 腿/页缺失；draft head 用 106,299 行子集（212.9 MiB）。
   - 我们：blk.88 的 512 专家混在主 cache（5,721 槽），每个 draft token 的 22 个 routed 专家按普通层走 GPU cached/CPU（约 10.6 CPU miss/draft），**draft 自身就要付 CPU 专家腿**；且 head 默认全词表 **440.4 MB**（老师 212.9 MiB）。
   - 量级核对：我们 3 draft/轮 ≈ 6.9 ms 额外 ÷ 3.05 token/轮 ≈ **+2.3 ms/token ≈ +3.1%**，与 R40 实测 **+3.2%** 吻合 ⇒ **MTP 的收益被"每 draft 一次完整 MoE 层 + 全词表 head"抵消，主要不是验证并集问题。**
3. **可修方向（给 lead 排序）：**
   - ① **给 blk.88 的 512 专家建专用 VRAM 常驻区**：512 × 3.0966 MB = **1.585 GB**；draft 的 MoE 只走 GPU（现有 `moe()` cached 路径即可，`moe_card` 对非门控返回 false、需小改或直接用 cached 分支）⇒ 去掉 ~1.5 ms/draft 的 CPU 腿；代价是主 cache 让出 ~1.585 GB（5,721 槽 ≈16.5 GiB ⇒ 掉 ~550 槽/约 3 个点命中），需算净账。
   - ② **`--draft-vocab` 缩 head**：440.4 MB → 32k 行约 108 MB / 64k 约 215 MB，每 draft 省 ~0.3–0.5 ms；零改码，需 ids 门（draft 只能出子表内 token，可能降接受率；老师用 106,299≈70% 主词表，可作起点）。
   - ③ **不建议照抄"跳过 routed MoE"**：老师没跳（`skip/no_moe/shared_only` 0 命中，§1），跳了是质量换速度，需 A/B 实验；若要做，作为独立臂（`STRATA_POLY_DRAFT_NO_MOE`）。
   - ①+②合起来可把 +3.1% 的 MTP 开销压到 ~+0.5% 以内（估算，未验），**MTP 才可能从"净亏"变"净赚"**。

---

## §5 未验 / 查无（明写）

- **T/U 两树 `mtp.cpp` 均无"跳过 MoE/只走共享专家"分支**：`skip`=0、`no_moe`=0、`nomoe`=0、`shared_only`=0、`only_shared`=0；`draft_vocab` 有（T 2 处、U 4 处）。
- **U 树同构**：`upstream-0.1.44.1/src/core/mtp.cpp:308-326` 同样整块加载 `experts.bin`（512 routed），`:1052` 注释"the 512 resident experts"，`:1122/1125` 同样 `moe_group_resident`/`moe_grouped_s2`；`:507/622-690` 有 draft_vocab 子集的取舍注释（U 树对"子集 vs 接受率"有专门说明）。
- **我们"跨半窗重复读"的实际次数未实测**（只读码：halves 时每个半窗各调一次 `moe()`；`pick_way` 会在两种走法间自适应）。
- **1.585 GB 常驻后主 cache 命中率下降多少未实测**（只是按槽数比例估算 ~3 个点；需要一次 A/B）。
- 我们 draft 的 CPU miss 数 10.6/draft 是按全局 51.7% 命中与 R34/R28 的 145 µs/miss 推算（D），未单独量 blk.88 层的命中（它可能高于或低于全局）。
- R40 的 +3.2%/+0.9%、tokens/round 3.05/3.20、兑现 93.5%/97.8% 来自 lead（B 级），本件未复算。

## §6 落盘 / 复现

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-R-MTP-draft成本结构对比-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IR-20261018-MTP-draft成本结构对比.md`
- 复核命令：
```powershell
# 老师 MTP：MoE 必走 + 512 专家常驻 VRAM
Select-String -LiteralPath 'E:\strata\src\core\mtp.cpp' -Pattern 'MoE: router, the 512 resident experts','moe_group_resident','moe_grouped_s2','the 512 experts do not fit','draft head over'
# 老师 verify 并集
Select-String -LiteralPath 'E:\strata\src\core\expert_source.cpp' -Pattern 'Distinct experts in routing order','multi_misses','job_of'
# 我们 draft：完整 MoE + 全词表 head
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\engine.cpp' -Pattern 'moe_router(L, 1)','moe_shared(L, n_layer_, 1)','moe(L, n_layer_, 1)','head.n_out = draft_rows()'
# 我们 CPU 专家并集读（cpu_experts 内部）
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\cpu_experts.cpp' -Pattern 'stable_sort(by','gkey','runp'
```
