# 上游适配侦查（ninfer / strata / llama.cpp 参照线）— 只读

- 席位：`scheduler` · 2026-10-16 · **只读、零 GPU**；网络访问（**统一走代理 `http://127.0.0.1:7897`**，按队长令）。
- **证据分级**：外部来源一律 **C**；每条带 **repo + sha/ref + 日期 + 原文标题**（原文照抄，不转述）。
- **手段与合规**：① `git ls-remote`（只读）· ② **只在我们自己的 clone 里 `fetch`**（`…\PolyStrata\`，队长明确批准的**唯一** fetch 范围）· ③ GitHub **Atom feed**（因 `api.github.com` 匿名被限流）· ④ 本机 `git log`（只读）· ⑤ 读本地制品 README。**未碰 `E:\strata`；未在任何他人树里 fetch/pull/reset/checkout；未跑重 CPU/磁盘扫描**（builder 在测量）。

---

## §1 ★★★ 头号发现：**llama.cpp 上游 2026-10-15 刚加了「本靶模型」的支持**

**来源**：`https://github.com/ggml-org/llama.cpp` · path **`src/models/nemotron-h.cpp`** · Atom feed `commits/master/src/models/nemotron-h.cpp.atom` · feed 顶端 `<updated>` = **2026-10-15T13:34:56Z**。

**该文件的近期提交（日期 ← 原文标题）**：

| 日期 | 原文标题（照抄） |
|---|---|
| **2026-10-15** | **`models : add Nemotron Super 3.5 VL and Nemotron Bolt 74B A9B support …`** |
| 2026-10-13 | `graph: skip the LM head for pooled embeddings (#30445)` |
| 2026-10-07 | `llama: share the nextn tensor flags between models (#30097)` |
| 2026-10-06 | `models : consolidate nextn row cropping into shared helpers (#30017)` |
| （更早） | `model : extend Nemotron MTP support (#29018)` |
| （更早） | `models : allow Nemotron-H models to only define layer_norm_epsilon (#…)` |

**★ 对我们意味着什么（三条）**：
1. **★★ 参照线比我们新的多 —— 必须重估数值基线**：我们 pin 的是 **`3cf03257`（本机 `llama-cpp-pin`，最新一条触及该文件的提交日期 = 2026-09-20，标题 `CUDA: enable sparse fa for qwen4 (#28770)`）** ⇒ **上游在同一个月里对该文件改了 5–6 次，含"加上 Nemotron Super 3.5 VL 支持"** ⇒ **`E:\llama-cpp-pin` 这个"同源 pin"很可能已过时**，而我们**全部数值对拍（`evalcb`）都基于它**。
2. **`models : allow Nemotron-H models to only define layer_norm_epsilon`** ⇒ **正是我们 t17 踩过的那个键**（`attention.layer_norm_rms_epsilon` ↔ `attention.layer_norm_epsilon`）⇒ **上游的解法是"允许只定义后者"**，我们的 `meta_alias`（`nemotronhmoe.cpp:31-38`）方向与之**一致** ⇒ **可交叉验证**。
3. **`extend Nemotron MTP support` / `nextn` 两条** ⇒ 与本线**冻结中的 MTP** 同题 ⇒ 若将来解冻 MTP，**上游已有进展可参照**。

---

## §2 strata 上游（同一仓：`VecSzn/PolyStrata`）—— **无本模型适配**，但有两条可搬

**手段**：在我们自己的 clone 里 fetch（队长批准的唯一范围）。
```
fetch: eb5ef11..51dc242  origin/main（+ 新 tag v0.1.44.1-poly.3/4/5）
```
⇒ **我们的 clone（HEAD `eb5ef11`）落后上游 `51dc242`（日期 2026-10-15，本地时 13:40）共 24 条提交**。

**逐条扫过 24 条标题后，与"调度/驻留/重叠/专家"相关的只有 3 条**（其余是 docs / serve / vision / MiniCPM-V）：

| sha | 日期 | 原文标题（措辞） | 对本靶的意义 |
|---|---|---|---|
| **`b9aef4f`** | 2026-10-15 | **`poly: give attention cache rows a long request left back to experts`** | ★★ **正对我们 t16 / `KP-UNI-0226` 的"KV ↔ 专家缓存抢显存"**：上游做的是**把长请求剩下的 attention cache 行"还给"专家**；我们的 `cover()` 是**只切给 KV、不还** ⇒ **这是一条明确的、可搬的优化方向**（实现细节未读 ⇒ 未验） |
| `98d1947` | 2026-10-15 | `poly: opt-in ggml Q8_0 KV cache for strata-poly (--kv q8_0)` | **新的 opt-in KV 档 `--kv q8_0`**（我们目前 KV 是 f16，无 int8/q8 档）⇒ 容量/精度维度的新旋钮 |
| `107f1c5` | 2026-10-15 | `poly: MiniCPM-V 4.7 35B-A3B, text and pictures, on the qwen35moe engine (…)` | 上游**又接了一个 MoE 架构**（MiniCPM-V，走 `qwen35moe` 引擎）⇒ 说明上游在持续扩 MoE 家族，**但恰好不是 `nemotron_h_moe`** |

**★ 关键词核（本地 `git log` grep）**：24 条标题里含 `nemotron|mamba|latent|hybrid|expert|resident|stage|overlap|schedul|moe` 的**只有上面 2 条**（`107f1c5` / `b9aef4f`）⇒ **没有一条提到 nemotron / mamba / latent**。
⇒ **负面结论**：**strata 上游没有针对本靶模型的适配 ⇒ 我们的移植债（latent 几何 / 非门控 / 批路径 / 重叠 …）没有现成补丁可摘。**

**旁证（参照线未漂）**：`3d038c4` 原文 *"the engines stay on Strata's 3cf03257 until Strata moves its pin"* ⇒ **poly 引擎侧 pin 仍是 `3cf03257`**（与我们一致）；vision 侧另起一个 pin（`69f201a`），**与我们无关**。

---

## §3 ninfer 线（`1314521gjy/ninfer-fusion-kvmem`）—— **无本模型适配**

- **URL 定位过程**：本机 `E:/modelscope-publish/github-ninfer-fusion-kvmem/` **有 `.git`**（与队长先前"`git remote -v` 无输出"的判断**不一致** ⇒ 我实测它**有 origin**），`git remote -v` = **`https://github.com/1314521gjy/ninfer-fusion-kvmem.git`**（取自该仓 README 的 issue/releases 链接，两处一致）。
- **本地快照**：HEAD **`61f0257` @ 2026-10-12**（`docs(48): 术语固定 q树/g树 + 基座 A/B 实测对比 …`）。
- **远端**：`git ls-remote` ⇒ **`refs/heads/main = b6e6c20deb93fe5ed84e5715642dad21c0455ef0`**（**比本地新**；**日期未取 ⇒ 未验**）；另有 `refs/tags/engine-v0.11.0-kvmem-20261008` 与若干 PR refs。
- **★ 关键词核（本地 `git log --all` grep `nemotron|mamba|latent`）⇒ 零命中**。
- **它是什么**：README 原文 *"本项目基于 NInfer `0.11.0-rtx3090`，发布融合后的源码、复现记录和验证工具。它是 NInfer 的**下游衍生项目**"*；`refs/upstream-20261009-ninfer-all` 目录名也印证"上游快照"。
⇒ **负面结论**：**ninfer 线（含其上游 NInfer）没有本模型的适配**；且该仓的近期提交集中在 **10-12**、主题是 **KVMem 环 / 基座 A/B / serve**，**与 Mamba-2 / latent MoE 无关**。

---

## §4 结论：对我们意味着什么（含负面结论）

1. **没有现成补丁可摘**（**负面结论，明写**）：**strata 上游 + ninfer 线都没碰过 `nemotron_h_moe` / Mamba-2 / latent MoE** ⇒ 本靶的**移植债（latent 三宽度 / 非门控批路径 / 重叠闸门 / profile 累积 / snap …）仍在我们自己手里**，别指望上游给解。
2. **★★ 但参照线（llama.cpp）昨天刚加了本靶**（`models : add Nemotron Super 3.5 VL and Nemotron Bolt 74B A9B support`，2026-10-15）⇒ **最该做的下一步不是找"移植补丁"，而是去读上游那条提交**：
   - 它会给**权重的规范读法 / 架构常量 / MTP·nextn 处理**的**官方最新口径**；
   - 而**我们的 pin（`3cf03257`，2026-09-20）比它旧一个月、且这一个月里该文件改了 5–6 次** ⇒ **我们所有数值对拍的"参照基线"要重估**（尤其 `layer_norm_epsilon` 兼容、nextn/MTP、pooled LM head 三条）。
3. **两条可搬项（strata 侧，与本靶机制直接对口）**：**`b9aef4f` "把长请求剩下的 attention cache 行还给专家"** —— 正对 t16 的 `KV ↔ 专家缓存` 硬耦合（我们只切不还）⇒ **值得读它的 diff**；`98d1947` 的 **`--kv q8_0`** 是新的 KV 档。
4. **参照线 pin 未漂**（poly 引擎侧仍 `3cf03257`）⇒ 我们的**同源对拍前提**在 strata 侧仍然成立；**变的是 llama.cpp 侧**（§1）。

---

## §5 未验（如实列，全部标 D/C）

- **【C，未取 sha/正文】** llama.cpp §1 六条**只有 Atom 标题与日期**；**每条的具体 sha 未取**（在 entry 的 `<link>` 里，未解），**diff/正文未读** ⇒ "加了 Nemotron Super 3.5 VL 支持"**是标题级结论**。
- **【C，未验】** ninfer 远端 `b6e6c20` 的**日期与内容未取**（只 `ls-remote` 到 sha）⇒ "比本地新"成立，但"新在什么"未知。
- **【B/C，未读 diff】** strata 的 24 条**只读了标题**（`b9aef4f` / `98d1947` 的**实现细节未读**）⇒ "可搬"是**方向判断**，非"已验证可搬"。
- **【D】** `E:\llama-cpp-pin` 是否真的**落后于上游**（我们 pin 的具体 commit 与上游那条 10-15 的关系）—— 只按日期推断，**未比对 sha**。
- **【D，补记】** **NInfer 本体的 URL 未定位**：另跑了一次更广的 grep（`E:/infer-docs` + `E:/视频/_agents` 全域），**ninfer 相关 URL 只出现 `github.com/1314521gjy/ninfer-fusion-kvmem`**（+ 若干无关的 ComfyUI 仓）⇒ 该衍生仓 README 只称"基于 NInfer `0.11.0-rtx3090`"，**未给上游地址** ⇒ **"NInfer 本体是否有本模型适配"这一问：未验**（但可推：**若本体有 nemotron 支持，下游衍生仓不会全历史零命中** ⇒ **弱证据指向"也无"**）。
- **【C，续查】** 同一批量 grep 里另出 **`github.com/Don-Chad/ninfer-3090`** ⇒ 我核了：`ls-remote --symref` 见 **`ref: refs/heads/master`（默认分支 master）**、HEAD = **`d7277e88…`**；分支表含 `release/v0.4.0-rtx3090 … v0.6.1-rtx3090`、`feature/qwen38-rk8v4-paged-sm86` ⇒ **形制与"基于 NInfer 0.11.0-rtx3090"相符 ⇒ 这很可能就是 NInfer 本体（3090 线）**。⚠️ 我对它**只做了 `ls-remote`；`commits/main.atom` 取空（分支是 `master` ⇒ 取错了分支）** ⇒ **"它有无本模型适配"= 未验**（下一刀：`commits/master.atom`）。

---

# §6 ★★ `b9aef4f` 详读（最高价值 —— 正对 t16，且**改了我 t16 的一个前提**）

- **stat**：`src/poly/engine.cpp` **+39/-3** · `engine.hpp` **+8** · `main.cpp` **+7**（共 **+51/-3**）。
- **机制（commit 正文原意，逐句可核）**：
  - **问题**：*"The attention caches grow with the sequence (cover()) and **never shrink**. After one long request every request that follows runs with an expert cache smaller by the rows that request mapped."* ⇒ **一次长请求之后，后续每个请求的专家缓存都少了"那次映射的行"**。
  - **修法**：**在请求开始时**（"once the conversation cache has left the engine where the request goes on from"），**unmap 掉超出「本次 prompt」与「上一个请求到达的最远行」的行**，让 `mind_vram()` **把显存给专家缓存**；判据原文：*"**Rows past that point are no conversation's**"*（停着的会话，其行在主机内存）。**"上一个请求"也计入** ⇒ 防"长会话与短请求轮流时每次请求都换专家"。**全驻留的卡保持整个上下文的行**（如 `load()` 所映射）。
  - `cover()` 改为**按 `kKvGrow` 行为单位分段 map**（使得"一次长读/恢复会话"映射的行**能被归还**），且**不动一个已经持有更多的缓存**。
- **★ 上游自带的数据（C 级，但可反算自洽）**：
  - **GLM 3.0bit、ctx 1,048,576、5090 预留 8 GiB**：一次 **287,474-token** 请求占 **3.59 GiB**（1,672 → 1,192 专家）。**修前**：后续请求保持 **1,192 专家、64–65 tok/s**；**修后**：第二个请求**归还行 ⇒ 1,631 专家、70.5–71.0 tok/s**，命中 **50–52%**（原 **43–45%**）。
  - **Qwen3.6 UD-Q3_K_M @8 GB 笔记本**：31,641-token 之后 **归还 0.52 GiB**，1,719 → **2,042 专家**，命中 **75%**（原 64%）。
  - 触发它的外部读数（issue 10）：24 GB 卡、1M ctx ⇒ **19 tok/s 而非 26.7**。
- **(c) 能否搬到我们引擎 —— 对照我们的 `cover()`（**只切不还**）**：
  | 项 | 上游 | 我们 |
  |---|---|---|
  | 方向 | **KV 缩 ⇒ 行还给专家** | **只有 `cover()` 切给 KV**（`engine.cpp:533-551`）+ `shrink_cache(cut)`；**无反向路径** |
  | 新状态 | `kv_reach_`（最远行，`engine.hpp`） | **没有** |
  | 归还时机 | **请求开始**（`main.cpp` +7） | 无 |
  | 交还对象 | `mind_vram()` | 对应 `resize_cache()` / `held_is_budget()` |
  | 分段 | `cover()` 按 `kKvGrow` 行**分块** | 我们也有 `grain_`/`up_to(...)` 分块（`cover()` 内）⇒ **有基础** |
  - **改动面**：**≈3 处 + 1 个新状态**（与上游同量级 ≈50 行）；**代价**：单卡上"归还"会与后续 KV 增长再抢 ⇒ **必须照搬"计入上一个请求的 reach"这条防抖**；**风险**：我们支持 `--layer-split`（多卡）⇒ **"每卡各自的 reach/归还"要分别算**；与我们**尚未有的档位**（kv-resident/streaming）交互未知。
  - **★ 能省多少（D，反算，且自洽）**：上游 ≈ **13.1 KB/ctx-token**（3.59 GiB ÷ 287,474 ⇒ 与原文 §4 的 ~13.7 KB 吻合 ✅）⇒ **我们（Nemotron）KV = 8 KiB/token**（t16 实测）⇒ **1M 上下文 ⇒ KV ≈ 8 GiB**；一次 **300K 请求映射 ≈ 2.4 GiB** ⇒ **归还后可换成 ≈ +780 个专家槽**（3.1 MB/槽）。
- **★★ 它修正我 t16 的一条结论（重要）**：t16 说「非专家可动项只有 ~0.72 GiB ⇒ 省空间没肉」—— 那是**在 `max_context = 8192` 下**；**一旦上下文开大（≥262K），KV 就是 GiB 量级 ⇒ "KV ↔ 专家"的兑换额从 0.7 GiB 变成 GiB 级** ⇒ **"省空间没肉"只在短上下文成立**（该结论应加此条件）。

# §7 `98d1947` 一句（`--kv q8_0`）

- **旗标名**：**`--kv q8_0`**（**opt-in**）。
- **改什么**：新增 Q8_0 KV 的内核与测试（`kernels.cu` **+309** · `kv_q8_test.cpp` **247 行** · `docs/Q8_KV.md` **157 行** · `main.cpp` +19 · `kernels.hpp` +30）。
- **★ 与现有档的关系**：我们的 KV 目前是 **f16**（无 int8/q8 档）⇒ 这是"**容量/精度**"维度的新旋钮候选；**但接线点只在 `qwen35.cpp`（+105）** ⇒ **不是通用档**，我们（Nemotron 走 `nemotronhmoe.cpp`）**不能直接吃**，要吃就得自己接线（内核 + 接线）。

# §8 llama.cpp 10-15 那条的正文对照（sha **`cd0a5b6a180dff0048e5fd993098b21c0b2c91b8`**）

> 只答"**我们 `nemotronhmoe.cpp` 的做法有没有出入**"。patch 触及：`src/models/nemotron-h.cpp` · `gguf-py/gguf/{constants,tensor_mapping}.py` · `conversion/nemotron.py` · **`tools/mtmd/clip*.{h,cpp}`（视觉）** · `common/speculative.cpp`（投机）。

| 项 | 上游（patch 原文） | 我们 | 出入 |
|---|---|---|---|
| **latent 专家几何** | `ffn_up_exps {moe_n_embd, n_ff_exp_i, n_expert_layer}` · `ffn_down_exps {n_ff_exp_i, moe_n_embd, n_expert_layer}` | `exps()` 校验 `[latent, n_ff_exp, n_expert]` / `[n_ff_exp, latent, n_expert]` | ✅ **一致**（与靶 GGUF 实测 `[1024,2688,512]`/`[2688,1024,512]` 同） |
| **每层专家数** | **新增** `n_expert_layer`（由 router 张量反推 `router_meta->ne[1]`）+ 校验 `n_expert_used(i) ≤ n_expert_layer ≤ n_expert` | 用**全局** `n_expert_` | ⚠️ **上游更严**（本靶 512 全层一致 ⇒ 当前无碍） |
| **每层 `n_ff_exp`** | `hparams.n_ff_exp(i)`，缺则回退 `n_ff(i)/n_expert_used(i)` | 全局 `expert_feed_forward_length` | ⚠️ 同上（上游按层） |
| **★ per-expert scale 张量** | **新增** `ffn_down_exps_s` / `ffn_up_exps_s` / **`ffn_down_exps_in_s`**（`{n_expert}`，`TENSOR_NOT_REQUIRED`，挂 mtp_flags） | **本席未在 poly 见到对应读取** | **★ 待核**：靶 GGUF 若含这些 scale 而我们不读 ⇒ **数值偏差**。（我早先的 GGUF 探针**只筛了固定几个名字、未含 `_s`/`input_scale`** ⇒ **未验**；**一次 GGUF 头读即可定**） |
| 视觉 / VL | `mtmd/clip*` + `conversion/nemotron.py` | 我们未做视觉 | 无出入 |
| 投机 | `common/speculative.cpp` 被改 | 我们 MTP 冻结 | 参照 |
| `layer_norm_epsilon` | **本条 patch 里未命中该字样** ⇒ 它是**另一条** commit | 我们 `meta_alias`（`nemotronhmoe.cpp:31-38`） | **未验**（那条的 sha/正文未取） |

**§8 未验**：`_s`/`input_scale` 是否在本靶 GGUF（**一次头读可定，零 GPU**）· 那条 epsilon commit 的 sha/正文 · patch 我只读了**关键行**（708 行里未逐行读）。
- **未跑 GPU、未改码、未编译**；**未在他人树里做任何写操作**。
