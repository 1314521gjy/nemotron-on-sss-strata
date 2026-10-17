# `--prompt-batch` ↔ 专家 arena ↔ 命中率：定量配比 — 2026-10-16

- 席位：`scheduler`（引擎后端-调度）· **只读**：未改码 / 未编译 / **未跑 GPU**。
- 任务来源：队长（零 GPU、只读）。四问：① 读码定死 batch/max_context/KV 各占多少显存与分配点；② batch=8/64/256/1024 四档 arena 上界与预期命中（D）；③ 张力陈述（批次大=重叠前提 ⇄ 抢驻留）⇒ 两全区间或硬取舍；④ 与 D4 的关系。
- **★ 本件先纠两处既有判断**（证据优先于直觉）：
  - **纠-1（对队长）**：**「`--prompt-batch 1024` 的 scratch/KV 吃掉 ≈17 GiB」不成立**。**A 级实测日志**：当前 G0' 跑法的 arena 就是 **23.1 GiB / 8010 对**（`g0r_1/2/3.log:3`、`g0q_*.log:3`），**与 t16 的 23.6 GiB / 8185 对同量级** ⇒ 没有 17 GiB 被吃掉。缘由见 §2。
  - **纠-2（对我自己）**：`反组设计-20261016.md` §1.3 的「实测斜率只有线性模型的 ≈1/4」**口径用错** ⇒ 见 §4。已在原件加更正横幅。

---

## §0 结论速览（三行）

1. **批次对 arena 的代价很小**：**≈1.4–1.7 MiB/token 的 scratch**（`--prompt-batch 1024` ⇒ ≈1.6 GiB）+ 一个 **1.48 GiB 的 stage 台阶**（仅当 `rows_ ≥ 32` 时分配）。
2. **四档预测（D）**：batch 256/64/32 只让回 **1.2–1.6 GiB**（⇒ **+2.0～+2.6 点容量 ≈ +1.6～+2.1 点命中**，**落在 ±3 点噪声内 ⇒ 量不出来**）；batch 8 让回 **3.2 GiB**（+5.1 点容量 ≈ +4 点命中）**但关掉重叠**。
3. **⇒ 张力是"名义两全、实益近零"**：`--prompt-batch` **不是**本轮"成套配比"的主要旋钮（**reserve 与 profile 才是**）。**真正的二选一只有一处**：`batch < 32`（省 3.2 GiB / ≈+4 点命中 ⇄ **重叠消失**）。

---

## §1 ★ 读码定死：四类显存各占多少、走哪几个分配点（A/B 级）

### §1.1 模型几何（A：GGUF + 代码常量，出处 t16 §1 与 `nemotronhmoe.cpp:56-100`）

| 量 | 值 |
|---|---|
| 层型 | **88 层 = 40 mamba + 8 attn + 40 moe**（`dump` 行 `nemotron_h_moe 88 layers`） |
| `n_embd` / `n_head` / `head_dim` / `n_kv` | 4096 / 32 / 128 / **2**（GQA 16:1） |
| 专家 | `n_expert=512` × 40 = **20480**；`n_used`(top-k)=**22**；非门控 latent，宽 `exp_in=exp_out=1024` |
| Mamba | `d_inner=8192`、`d_state=128`、`d_conv=4`、`n_group=8`、`ssm_heads=128` ⇒ `channels=10240`、`d_in_proj=18560` |
| `q_max` | `max(n_embd, n_ff, n_ff_exp, n_ff_shexp, widest(), 256)` = **`widest()=d_inner=8192`**（`engine.cpp:1636`） |
| 每对专家字节 | **3,096,576 B = 3.0966 MB**（t16） |

### §1.2 分配点（逐条 `file:line`）

| 项 | 分配点 | 随什么变 |
|---|---|---|
| **`rows_`（批次本体）** | `engine.cpp:1591` `rows_ = opt_.dump ? 1 : min(prompt_batch, max_context)` | ← **batch/max_context 的唯一入口** |
| **scratch（架构自算 + 共享）** | `alloc()` = `engine.cpp:1582`：`shared_scratch()` `:1552` · `assign_scratch()` `nemotronhmoe.cpp:215`；两趟计数后一次性 `dmalloc(scratch_at_)` `:1631` | ∝ `rows_`（下 §1.3） |
| **`xq_` / `q8_`** | `engine.cpp:1637-1638` | ∝ `rows_`；`q8_` 按 `q_max=8192` |
| **MMQ 缓冲**（`mq_/mq2_/mqx_/d_*/tab_/iota_`） | `engine.cpp:1648-1663`（`if (mmq_)`） | ∝ `rows_*n_used` |
| **`stage_`（专家暂存/预取）** | `engine.cpp:1668-1670`：`stage_bytes_ = max(min(stage_mib<<20, n_expert_*one_expert), one_expert)` | **与 batch 无关**（按层），**但 `mmq_` 为假时不分配** |
| **KV（attention）** | `nemotronhmoe.cpp:196-204` → `kv_alloc()` `engine.cpp:526-535`；**惰性映射** `cover()` `:541-560` | ∝ **已到达的 ctx 行**（8 KiB/行），非 ∝batch |
| **recurrent state + `snap_`** | `engine.cpp:1594-1605` | ∝层数（常数） |
| **arena（专家缓存）** | `fill_expert_cache()` `engine.cpp:994`，**在 `alloc()` 之后**（`engine.cpp:1487` → `1489`） | = `min(all, free_b − reserve, …)` `:1043` |

★ **关键因果（A）**：`alloc(); fill_expert_cache();`（`engine.cpp:1487-1489`）⇒ **arena 定尺寸时 scratch/stage/KV/state 已经分配** ⇒ **每一 MiB scratch 直接 1:1 从 arena 里扣**（`arena = free_b − reserve`，`:1043`）。这就是"批次抢驻留"的**唯一通路**。

### §1.3 「batch → scratch 字节」算式（B 级，逐项可复算）

记 `n = rows_ = min(prompt_batch, max_context)`。**每 token（每行）的 device 显存 = Σ 下列系数 × 4 B**：

| 来源 | 项 | 系数（float/token） | `file:line` |
|---|---|---|---|
| shared | `r_`（草稿/双调用行） | `max(1, draft?3:1)×(read_rows_>rows_?2:1)×4096` ≈ **24576** | `engine.cpp:1556` |
| shared | `cur_,xn_,ao_,fo_` | `4×4096` = **16384** | `:1557-1560` |
| shared | `g_,u_,h_` | `3×max(n_ff_,n_ff_shexp_,2n_ff_exp_)` = `3×5376` = **16128** | `:1561-1563` |
| shared | `moe_in_,moe_out_,moe_up_` | `3×4096` = **12288** | `:1566-1568` |
| shared | `rl_` | **512** | `:1570` |
| **shared(mmq)** | `gu_,hx_,dx_` = `rr×(2n_ff_exp + n_ff_exp + n_embd)`，`rr=n×22` | **267520** | `:1573-1578` |
| arch | `proj_,cv_,ssm_y_,qraw_,q_,kraw_,k_,v_,att_` | `18560+10240+8192+4096+4096+256+256+256+4096` = **50048** | `nemotronhmoe.cpp:217-225` |
| arch | `awork_` | **96** | `:229` |
| q8 | `xq_`,`q8_` = `native_q8_1_bytes(·)`：`(n_in/32)×36 B` | `(4096/32×36 + 8192/32×36)/4` ≈ **3456** | `engine.cpp:1637-1638`；`mmvq_host.cpp:14-15,56-58` |
| **mmq 缓冲** | `mq_,mq2_,mqx_,d_ids_,d_wts_,iota_` | ≈ **41888**（按 q8 行 ≈1.0625×cols 估） | `:1649-1663` |

- **合计 ≈ 418,512 float/token ≈ 1.674 MB/token**。与代码自述 `main.cpp:764`「about **1.4 MiB** of VRAM each」同量级 ⇒ 取 **1.4–1.7 MiB/token**（**B，±20%**；`mq_/mq2_` 的精确系数依赖 MMQ 库的 `q8_bytes()`，本席未读到其定义 ⇒ 那两项标 **D**）。
- **与 `n` 无关的常数块**（全部 A）：`snap_` **494 MiB**（`:1605`）· `state_` **165 MiB**（`:1596`）· `wlogits_` **4 MiB**（`:1606`）· `logits_/d_` ≈1 MiB · `score_` **64 MiB** + `awork_` 25 MiB（**仅 mmq**，`nemotronhmoe.cpp:226-229`）· **`stage_` 1476 MiB**（**仅 mmq**，`:1668-1670`：`min(2048, 512×3.0966MB=1476MiB)`)。
- **KV 不进 batch 口径**（惰性）：`cover()` 按"已到达的 ctx"映射（`engine.cpp:543`）⇒ 8192 ctx 满值 **64 MiB**（8 KiB/行 × 8 层，t16 §3-2 与 llama.cpp 逐位同）。
- **【D】未读到**：`read_rows_`（`engine.cpp:1592-1593`，取决于 `kReadBytes`）是否 > `rows_` ⇒ `r_` 的 ×1 或 ×2 未定（差 98 KiB/token）。`bf16_max_`（`wf_`）本模型应为 0（无 BF16 矩阵），未核。

---

## §2 ★ 为什么不是 17 GiB：与 A 级日志对表

| 口径 | 值 | 来源 | 级别 |
|---|---|---|---|
| **当前 G0' arena** | **8010 / 20480 = 39.1% · 23.1 GiB** | `P3-20261016\g0r_1.log:3` · `g0q_1.log:3` · `m5_1.log:3` · `loc_base.log:3` | **A** |
| t16 记的 arena | 8185 / 20480 = 40.0% · 23.6 GiB | `dump_t3_fix.log:3`（**dump 跑**） | A（但口径不同） |
| 若真被吃掉 17 GiB | arena 应 ≈ 6 GiB ⇒ ≈ 2000 对 | —— | ❌ **与 A 级日志矛盾** |

- **17 GiB 这个数的来源**：是**本席上一件的 D 级反算**——"命中 66.2% 对上覆盖曲线的 2048 对"⇒ 反推 arena≈6–7 GiB。**它错了**，因为：
  1. **覆盖曲线（静态 profile 分布）在 8010 对处预测 ~94%，而实测命中 66.2%** ⇒ **静态曲线不是"上报命中率"的有效模型**（见 §4）；
  2. 用错的模型反推出的 arena，与 A 级日志直接冲突 ⇒ **日志赢**（证据优先）。
- **推论（A 级）**：`--prompt-batch 1024` 的 scratch+stage ≈ **3.1 GiB**（≈1.6 + 1.48），**不是 17 GiB**。**"降 batch 能把命中拉回 95%"这个预期不成立**（§3）。
- ⚠️ **口径警告**：**arena 绝对值不可跨会话直接比**（dump 跑 23.6 > batch 跑 23.1，但 dump 关掉 mmq/stage 本应**更大** 1.5+ GiB ⇒ 说明两次会话的 `free_b` 不同）⇒ **要证明"batch↑ ⇒ arena↓"必须同一会话内 A/B**（§5-未验-1）。

---

## §3 配比表：四档 batch 的 arena 上界与预期命中（D）

**锚点**：batch 1024 ⇒ arena **23.1 GiB / 8010 对**（A）。**换算律**：
`Δarena = Δ(rows_) × 1.65 MiB`（B）；**1 MiB = 0.338 对 = 0.00165 of 20480**；**命中敏感度取实测斜率 ≈ 1.29 点/GiB**（由 G0'(2500) → GNEG'(256) 的 +2.9 点 / 2244 MiB 得，**A**）。

| batch | 重叠/G2 路径 | Δarena（vs 1024） | 预测 arena | 预测对数（占 20480） | 预测 Δ容量 | **预测 Δ命中（D）** | 可测性 |
|---|---|---|---|---|---|---|---|
| **1024** | ✅（需 `n≥stage_ahead=1024`，刚好） | — | 23.1 GiB | 8010（39.1%） | — | —（基准） | — |
| **256** | ✅（`n=256 < 1024` ⇒ **ahead 关**，见 §4-3） | +1.24 GiB | 24.3 GiB | 8420（41.1%） | +2.0 点 | **+1.6 点** | ❌ 在 ±3 噪声内 |
| **64** | ✅ 同上（ahead 关，除非 `--stage-ahead 64`） | +1.55 GiB | 24.7 GiB | 8521（41.6%） | +2.5 点 | **+2.0 点** | ❌ 噪声内 |
| **32** | ✅ 同上（**临界**：`rows_ ≥ kStageRows`） | +1.60 GiB | 24.7 GiB | 8538（41.7%） | +2.6 点 | **+2.1 点** | ❌ 噪声内 |
| **8** | ❌ **mmq 关 ⇒ 整条批/预取路径关** | **+3.17 GiB** | 26.3 GiB | 9057（44.2%） | +5.1 点 | **+4.1 点** | ⚠️ 约 1 个噪声宽 |

- 各档算式：`1024→256`：(768)×1.65 MiB=1267 MiB；`→64`：(960)=1584 MiB；`→32`：1637 MiB；`→8`：**1016×1.65 + 1476(stage) + 90(score/awork/tab) ≈ 3242 MiB**（mmq 关 ⇒ 同时省掉 stage 与 MMQ 常数块）。
- **口径**：预测 **D**（1.65 MiB/token 是 B，敏感度 1.29 点/GiB 是 A 但对 batch→arena 的**跨会话**外推是 D）；**不要当基线用**。

---

## §4 张力陈述：批次大 = 重叠前提 ⇄ 抢驻留

### §4-1 重叠的前提（A，逐条 `file:line`）
- **批/MMQ 路径**：`batched(L,n) = mmq_ && L.mmq && n ≥ kStageRows(32)`（`engine.cpp:1877`）；而 `mmq_` 的存在要求 `rows_ ≥ 32`（`:1622`）⇒ **`--prompt-batch ≥ 32`**。
- **预取（ahead）**：`moe_rows()` 的 `ahead = stage_ahead>0 && (n ≥ stage_ahead || shared_ahead_)`（`:2197`）；层主序重叠路径要求 `n ≥ opt_.stage_ahead`（`:2855`）。**`--stage-ahead` 量的是"一次调用里的 token 数"**（默认 **1024**）。
- ⇒ **两个门**：①`--prompt-batch ≥ 32`（否则连 mmq 都没有）②**`--prompt-batch ≥ --stage-ahead`**（否则 `ahead=false` ⇒ 预取不发生，退化成逐调用拷贝）。

### §4-2 两全区间：**名义存在、实益近零**
- **`--prompt-batch 64` 且 `--stage-ahead 64`**（**两者同降**）：重叠保留（`n=64≥64` ✓），scratch 让回 **1.55 GiB** ⇒ **+2.5 点容量 ≈ +2.0 点命中** ⇒ **落在 ±3 点噪声内 ⇒ 量不出来**。
- ⇒ **诚实结论**：**在这一轮噪声下，"降 batch 换命中" 是"名义两全、实益近零"**。**别把它当"成套配比"的主旋钮**——主旋钮仍是 **reserve（前件）与 profile（D4）**。

### §4-3 ★ 真正的硬取舍只有一处
- **`--prompt-batch < 32`**（如 8）：省 **3.2 GiB** ⇒ **+5.1 点容量 ≈ +4.1 点命中**（约 1 个噪声宽，**是本件唯一可能量到的批次效应**），**代价 = MMQ 批路径 + 整个预取/重叠链条一起消失**（`:1622` `mmq_` 为假 ⇒ `stage_` 都不分配）。
- ⇒ **这是硬取舍**：**要么批次 ≥32 保重叠（命中几乎不动），要么 <32 拿 ~4 点命中但牺牲重叠**。**没有第三个选项。**
- ⚠️ 叠加项目红线：**`--stage-ahead` 只作用于 prefill**（`moe_rows()` 在 prefill 路径，`engine.cpp:2194`）；decode 只 ~45 token ≈ 3 s（跑法纪律 D7）⇒ **降 batch 的收益也主要在 prefill**。**任何 batch 档的结论都要用 §运前份 D7 的"prefill 主读数"判**。

### §4-4 ★ 为什么静态覆盖曲线不能用来预测命中（纠-2 的根）

> ### ⚠️ 部分更正（2026-10-16 晚，本席加；原文保留不改）
> 本节里"**上报命中率被 1024-token prefill 主导**"这句**错了**（**纠-3**）。
> **真机制**：批（prefill）路径的命中去向 **`stats_.batch_cached/batch_staged`**（`engine.cpp:2366-2368`），**根本不进** `gpu_experts` ⇒ **`experts from VRAM %` 里没有 prefill 的成绩**，它不是"被 prefill 拖低"，而是**把 prefill 排除在外**。
> ⇒ 本节**结论方向**（静态曲线不能预测该比值）**仍成立**；**机制已换**。详见 **`命中率分段取数方案-20261016.md` §2**。
- 覆盖曲线（源=母本 profile）在 2048 对处给 68.0%、8010 对处给 ~94%，而**实测 66.2%**（8010 对）⇒ **曲线系统性高估** ⇒ 说明**上报命中率 ≠ "静态 top-N 的访问覆盖"**：上报值（`main.cpp:722` `gpu_experts/(gpu_experts+cpu_experts)`）是**实例计数**，被 **`--prompt-batch 1024` 的 prefill 主导**——一次 1024 token 的批几乎触到该层全部 512 专家，而其**容量比只有 39%** ⇒ prefill 段命中低、decode 段高，混合成 66.2%。**这条要用 A 级 `--top-probs`/分段计数去验，不能靠曲线推**（§5-未验-2）。
- **⇒ 纠-2**：`反组设计 §1.3` 用覆盖曲线斜率（≈5.2 点/1000 MiB）比实测（1.29 点/1000 MiB）得出「衰减 1/4」是**口径错**：**该比的是"容量比"（+1.65 点/1000 MiB）**，实测 1.29 ⇒ **比值 0.78，不是 0.25**。已在原件加更正横幅。

---

## §5 与 D4 的关系（第 ④ 问）

- **本件所有绝对 arena（23.1/23.6 GiB）都不受 profile 影响**（arena 尺寸只看 `free_b − reserve` 与 RNG-free 的 `used/bytes` 排序 ⇒ **profile 只决定"装谁"，不决定"装多少"**）⇒ **A 级 arena 锚点可跨跑直接比**。
- **但命中率受 profile 强烈影响**（D4：不冻结则 `--top-probs` 全不同、命中集合漂）⇒ **§3 的"预测 Δ命中"必须在冻结 profile 下验证**。
- **⇒ 请写进下次测量设计（同意队长）**：**reserve 三点 `2500 / 6000 / 12000` 全部在 D4 冻结 profile 下取**；**另加两条 batch 点 `1024 / 64`（同会话、冻结 profile）**，只为读 **load 行 `N of 20480 ... (x.x GiB)`**（**不必跑完 decode，加载即打印**，`engine.cpp:897`）⇒ **零额外成本就能把 §3 的 D 级预测变成 A**。
- **红臂（能变红）**：同配置重复一次，**arena 行必须逐字相同**（arena 是确定量）；**若不同 ⇒ 该次 `free_b` 被外部占用 ⇒ 判该跑不合格**，而不是"batch 有影响"。

---

## §6 未验 / 缺口（如实列）

1. **【D】"batch↑ ⇒ arena↓" 目前无 A 级直接证据**：现存变 batch 的日志（`b3_8.log` vs `b3_1024.log`）**都是 dump 跑**（`--dump` 强制 `rows_=1`，`engine.cpp:1591/1408`）⇒ 两者 arena 都是 8185/**同值**，**不含 batch 信息**。⇒ 需 §5 的**同会话 A/B**（只读 load 行）。
2. **【D】命中率的分段构成未验**（prefill 段 vs decode 段各多少）⇒ §4-4 的"prefill 主导"是**机制假说**，需分段计数或 `--top-probs` 验证。
3. **【D】`mq_/mq2_` 的精确系数**依赖 `prefill::mmq::q8_bytes()` 定义（未读到）⇒ §1.3 的 1.674 MB/token 有 ±20% 不确定。
4. **【D】`read_rows_ > rows_` 与否**未核（`kReadBytes` 未读到）⇒ `r_` 系数 ×1 或 ×2。
5. 本件**未改码、未编译、未跑 GPU**；§3 的执行与判读归 builder/verifier。

## §7 复现（只读命令）

```bash
B="E:/视频/_flashnext/experiments/20261015-nemotron-on-strata"
P="$B/PolyStrata/src/poly"
grep -n "routed experts in VRAM" "$B/P3-20261016"/*.log | head    # 各跑 arena（A 锚点）
sed -n '1591p;1622p;1636-1638p;1648-1670p' "$P/engine.cpp"          # rows_/mmq/scratch/stage 分配
sed -n '1552-1578p;1582-1607p' "$P/engine.cpp"                      # shared_scratch/alloc
sed -n '215-230p' "$P/nemotronhmoe.cpp"                             # assign_scratch
sed -n '1877p;2194-2198p;2855p' "$P/engine.cpp"                     # batched / ahead / 重叠门
sed -n '526-560p;994-1004p;1040-1058p' "$P/engine.cpp"              # kv_alloc/cover/fill_expert_cache
grep -n "kStageRows\|kQ81Block" "$P/runtime.hpp" "$B/PolyStrata/metal/src/engine/mmvq_host.cpp"
```
