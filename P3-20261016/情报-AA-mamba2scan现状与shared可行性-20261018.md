# 情报-AA · `mamba2_scan` 现状核实 + ①state→shared 可行性（零 GPU，当前源码）— 2026-10-18

- **席位**：scout-a · **触发**：lead 说 10-17 计划的头号瓶颈 **①state→shared「从未实施」**，要核实现状并评估可行性（10-17 计划原文见 `计划-分环节定位杠杆-20261017.md:25-95`）。
- **状态**：**达成，并纠正一个前提**：**① 不是"从未实施"——它在 2026-10-17 10:0x 就做完并有回执**（`回执\E1-mamba2scan-shared.md`，A 级），**且今天的交付配置已带 `STRATA_M2_SCAN_SMEM=1`**（`C:\sp3\run_delivery_t16.cmd`）。当前源码仍是 `template<bool SMEM>` 双路径、默认关、SMEM 走 opt-in env；**今天没有开 SMEM 后的 nsys**，所以"今天的占比"只能推断（详见 §2）。
- **纪律**：零 GPU、零编译、不改码、未 spawn 子代理；每条 `file:line`；"读码"与"推断"分列。

---

## §0 一句话结论

**① 已经做完并已进交付，不是漏做**：E1 回执（10-17）实测 `mamba2_scan` 单 kernel **38.62→3.36 ms（≈11.5×）**、端到端 prefill **348.2→728.8 tok/s（+109%）**、**ids/`--top-probs` 与旧逐位一致**；代价是 **decode −2.4%**（`rows=1` 摊不开 load/store）。当前源码就是那时留下的 `template<bool SMEM>`（`kernels.cu:2553-2610`），**今天交付跑法带 SMEM=1** ⇒ **现在的 1848–1938 tok/s 里已经含这笔红利**；**mamba2_scan 已不是头号瓶颈**（旧 nsys 里 26.7% 是 **未开 SMEM 的旧 exe**，且那是 **pb=6912/一条 prompt** 的坐标，不是 10-17 计划里 51.4% 的 pb=8 坐标）。**剩余唯一低风险动作 = "按 rows 自动选路"（prefill 用 SMEM、decode 用 global），拿回 decode 那 −2.4%（≈+0.6 tok/s @26–29）；再往下是 barrier 2→1（+4 KiB SMEM，仍 <48 KiB，预估 kernel 再 −20~40%，prefill 上界 +1–3%）与 ② d_state 切线程（占用 6.7%→21–42%，但改求和顺序 ⇒ 需重跑 ids 门，收益上界已被削小）。**

---

## §1 ① 现状确认（当前源码 `PolyStrata\src\poly\kernels.cu`，mtime 2026-10-18 00:35，162,598 B）

| 项 | 现状 | `file:line` |
|---|---|---|
| 定义 | **`template <bool SMEM> __global__ void mamba2_scan_kernel(...)`**（不是单一路径） | `:2553-2554` |
| launch 形状 | **不变**：`<<<n_head, head_dim, smem_bytes, stream>>>`（Nemotron = grid **128** × block **64**）；SMEM 版 `= <true>`，global 版 `= <false>` | `:2925`（true）、`:2941`（false）；grid/block 参数 `:2554-2558`（`n_head`/`head_dim` 传入） |
| SMEM 选择 | **opt-in env `STRATA_M2_SCAN_SMEM=1`**（默认关）；`need = head_dim*(d_state+1)*4` | `:2781`（getenv）、`:2917-2923`（need/optin 判定）、`:2925`（launch） |
| SMEM 大小 | **dynamic = 64×(128+1)×4 = 33,024 B（32.25 KiB）**；`cudaFuncSetAttribute` 只在 **need>48 KiB** 时调用 | `:2912-2917`（`need` 计算）· `:2918`（attr） |
| 现存的 static shared | `sdt, sdA` 8 B + `sB[512]`+`sC[512]` 各 2,048 B = **4,104 B** | `:2570-2571` |
| 总 shared/block | **33,024 + 4,104 = 37,128 B ≈ 36.25 KiB < 48 KiB 默认上限 ⇒ 本靶不需要 opt-in**（E1 也实测 33.02 KB 走 fast path，未触发回退） | 计算；E1 §1/§2 |
| `for t` 串行 | **仍是跨 token 串行**：`for (int t = 0; t < rows; ++t)`（一个 block 一个 head，64 线程只按 head_dim 切） | `:2572` |
| barrier 次数 | **仍每 token 2 次 `__syncthreads()`**（sB/sC 装载后一次、token 尾一次）；**SMEM 版没有新增 barrier**（state 是每线程私有行，不需要） | `:2582`（第一次）、`:2606`（第二次） |
| state 位置 | **默认（`SMEM=false`）仍在全局**：`float* sp = s + (size_t) ii * d_state;`，每 token 内 `sp[st]` 读+写 | `:2565`、`:2592-2597` |
| SMEM 版 state | 每线程 `d_state+1` 行进 dynamic shared：`ss = smem_state + i1*(d_state+1)`；进入时 load 一次、结束时 store 一次；token 循环内只读写 `ss[st]` | `:2566-2569`（声明/装载）、`:2585-2590`（循环内）、`:2609`（写回） |
| y / 快照 | `y[t*dim+ii] = sumf + xbc[...]*D[h]`；快照 `snap_of(...)` 写 `m += ii*d_state`（与 state 同坐标系） | `:2605`、`:2598-2604` |
| warp 变体 | 已实现但**禁用**（`#if 0`，R14b 的 tree/grouped-reduction kernel 在本卡**输给 SMEM**） | `:2612`（kernel `#if 0`）、`:2784`（launcher `#if 0`） |
| 今天交付是否开 | **是**：`run_delivery_t16.cmd` 第 5 行 `set STRATA_M2_SCAN_SMEM=1`，跑 `sp_borrow.exe`（R60 加 `STRATA_PROMPT_BORROW=1` 后 decode 25.90→28.83，ids 128/128 同，prefill −3.8%） | `C:\sp3\run_delivery_t16.cmd`；R60 驱动日志 |

**⇒ 结论（问题 1）**：launch 形状、`for t` 串行、每 token 两次 barrier、默认 state 在全局——**与 10-17 描述一致，未被改**；唯一变化是**多了一条已实现的 SMEM 快路径（默认关、交付已开）**。

---

## §2 ② 今天的占比（有/没有 nsys；没就推断）

### §2.1 没有 post-SMEM 的 nsys（明写）
- 盘上只有 **`C:\sp3\nsysR10.sqlite`（10-17 23:54）** 与 **`nsysR13.sqlite`（10-18 00:09）**，两次都用**旧 exe**（`sp_ring2.exe`，SMEM 源码 mtime **00:35**，晚于两次 nsys）⇒ **没有开 SMEM 后的 trace**。今天 R50–R60 的跑法日志里 **`mamba2_scan`/`M2_SCAN` 字样 0 命中**（只打印阶段账，不打 kernel）。

### §2.2 旧 nsys 的真实读数（A 级，pre-SMEM、pb=6912）
`nsysR13.sqlite`（`CUPTI_ACTIVITY_KIND_KERNEL` + `StringIds` 联接）：
| kernel | ms | 占比 | 次数 |
|---|---|---|---|
| `mul_mat_q` | 1,262.2 | 38.73% | 3,070 |
| **`mamba2_scan_kernel`** | **871.8** | **26.75%** | **40**（一层一次，prefill 6791 token 一次调用） |
| `quantize_mmq_q8_1` | 233.7 | 7.17% | 1,924 |
| `relu2_kernel` | 205.4 | 6.30% | 4,218 |
| `mamba2_conv_kernel` | 144.7 | 4.44% | 40 |
| `native_mmvq_slots_kernel` | 113.0 | 3.47% | 6,232 |
（R10 同构：845.2 ms / 26.73%；单次调用 **~21 ms**，40 次全是 prefill 大调用，无 decode 小调用。）
- **⇒ 旧坐标**：pb=6912、6791-token prompt、**旧 exe（未开 SMEM）**：`mamba2_scan` = **kernel 时间的 26.7%**、单次 ~21 ms。
- **⚠️ 10-17 计划的 "51.4% / 1.237 s / 1840 次" 是另一个坐标**：pb=8、700-token prompt、348 tok/s 的路径（calls 数是 ~1840，不是 40）⇒ 两个数不可混用（计划原文 `计划-分环节定位杠杆-20261017.md:25-37`）。

### §2.3 今天的占比（推断，D 级）
两条独立推断，互相夹逼：
1. **E1 的 kernel 份额转换**：pb1024/700-token 配置下 `mamba2_scan` 占 kernel 时间 **67.9% → 8.7%**（E1 §5 "kernel 份额 67.9%→≈8.7%"）。按同样比例套到 pb6912 的旧 26.7% ⇒ **今天 ≈3.5% 的 kernel 时间**。
2. **端到端夹逼**：旧 845 ms/次 40 调用；即使只按 E1 的 e2e 增益 2.09× 折 ⇒ **≤404 ms**；今天 prefill 6792 token / 1848 tok/s = **3.67 s** ⇒ **≤11% 的 prefill 墙钟**；若按 kernel 份额那条 ⇒ **~1–3% 墙钟**。
- **⇒ 今天 `mamba2_scan` ≈ 3–8% 的 prefill 墙（保守上界 ≤11%），已不是头号瓶颈**（旧 26.7% 是 pre-SMEM 旧 exe 的读数）。
- **要拿到真数**：在**今天 exe** 上跑一次 nsys（同 `run_delivery_t16.cmd` 配置，`--max-new 1` 缩短），按 §2.2 的 sqlite 联接法算 `mamba2_scan` 份额 ⇒ 这是"验证 E1 在 pb=6912 上是否同样 11×"的唯一硬办法。

---

## §3 ③ ①state→shared 的可行性（逐项）

### §3.1 shared 预算：放得下，且**不需要 opt-in**（本靶）
- dynamic：`head_dim × (d_state+1) × 4 B = 64 × 129 × 4 = 33,024 B = 32.25 KiB`（`+1` 是每行 padding 消 32 路 bank 冲突，见 `:2567` 与 E1 §1）。
- 现有 static：`sdt+sdA (8 B) + sB[512]+sC[512] (4,096 B) = 4,104 B`（`:2570-2571`）。
- **合计 37,128 B = 36.25 KiB < 48 KiB 默认上限 ⇒ 走 fast path，不触发 `cudaFuncSetAttribute`**（launcher 只在 `need > 48 KiB` 才 opt-in：`:2917-2923`；超过设备 optin 上限还有**响亮回退**：`:2930-2938`，E1 已实测未触发）。
- 若换更大 d_state/head_dim 的模型：`need` 会超 48 KiB，launcher 自动 opt-in（最多到 device optin 上限）。

### §3.2 布局与访问模式（`s + ii*d_state` → shared 的改法，位精确）
- 全局坐标：`ii = h*head_dim + i1`（h=blockIdx，i1=threadIdx），`sp = s + ii*d_state`（`:2565`）。
- shared 坐标：`ss = smem_state + i1*(d_state+1)`（`:2567`）——**每线程一行、行内 st 连续**；`st` 仍是**全局 st 索引**（0..d_state−1），**不改快照坐标**（见 §3.4）。
- 位精确：两条路径的内层**表达式与求值顺序逐字相同**——`state = ss[st]*sdA + sB[st]*x_dt; sumf += state*sC[st]`（SMEM `:2587-2588`）vs `state = sp[st]*sdA + sB[st]*x_dt; sumf += state*sC[st]`（global `:2593-2594`）；**st 升序、同一 FMA 形式、同一累加顺序** ⇒ float 结果逐位相同（E1 已实测：pb=8 下 ids `1531,6330,1307,23716,1395,1605,1261,6330` 三臂相同、`--top-probs` md5 三臂相同；N-A 负控把 `st<d_state-1` 故意少加 1/128 ⇒ ids/概率都变，判据有鉴别力）。
- 不新增 barrier：每线程只读写自己那条 `ss`，**不需要** `__syncthreads`（E1 §1 明写；当前代码 token 循环内确实只在 `:2582/:2606` 两处 barrier，与 state 无关）。

### §3.3 对 decode 的影响（同一个 kernel 双用）
- 同一函数、`rows` 参数化：decode 调用 `rows=1`。
- **E1 实测 decode −2.4%**（56.36→55.00 tok/s）：`rows=1` 时 shared 版每 call 多一次 128 load + 128 store，且摊不开（全局版在 rows=1 时也只读/写一次，收益为 0、shared 拷贝是纯开销）。
- **今天交付把 SMEM 开在所有调用上**（`run_delivery_t16.cmd` 全局 env）⇒ **现在 decode 一直背着这 −2.4%**。

### §3.4 快照/state 坐标约束（10-17 计划 §Q4 的地雷，① 天然合规）
- state 布局 `[h][i1][st]` 与快照坐标系写死：`snap_at_ + (B.ssm_state - state_)`、`stride = rec_n_`（`nemotronhmoe.cpp:286/289` + 计划 `:87-92`）。**凡改 `(ii,st)` 映射必须同步改快照写法**，否则 `keep()` 静默回滚到错位置。
- **① 不改映射**（`sp[st]` 的全局索引不变、只是存放位置搬进 shared）⇒ **快照代码一行不用改**（E1 §1 确认没碰 engine）。⇒ 这是 ① 相对 ② 的最大优势。

### §3.5 改动行数 / 风险
- **已实现**（E1）：`kernels.cu` 里 `template<bool SMEM>` + 4 处 `if (SMEM)` 分支 + launcher env/回退（约 40–60 行改动）。
- **剩余可做**（不含 ① 本身）：
  - **(i) 按 rows 自动选路**：launcher 加 `smem_state && rows > 32`（3–5 行）⇒ prefill 走 SMEM、decode 走 global ⇒ **拿回 decode 的 +2.4%**（26–29 tok/s 时约 +0.6–0.7 tok/s），零数值风险（两条路算术逐位同）。
  - **(ii) barrier 2→1**：把 `sB/sC` 双缓冲（各 2 份，+4 KiB shared ⇒ 45,256 B 仍 <48 KiB），下一 token 的 sB/sC 在计算当前 token 时预取，省一次 block-wide barrier（`:2582` 或 `:2606` 二选一）⇒ 约 20–40 行；**预估该 kernel 再 −20~40%**（SMEM 后 ncu Compute 仅 40.5%、占用 6.7%，仍同步/延迟受限）⇒ 若 kernel 份额按 §2.3 的 3–8% 算，prefill 收益 **+1~3%**。
  - **(iii) ② d_state 切线程**（计划 "甲"，`计划-分环节定位杠杆-20261017.md:83`）：`<<<n_head, head_dim*S>>>` 改一个数 + 跨段归约（warp shuffle 或与现有 barrier 合并）⇒ 并行度 8192→32,768/65,536、占用 6.7%→~21%/42%；**改求和结合顺序** ⇒ 相对误差 ~1e-5、**ids 可能要重跑门**；且 E1 后 kernel 已只 ~3.36 ms/8.7% 份额 ⇒ 收益上界缩小。

---

## §4 ④ 若 ① 太挤：更小的第一步

**先纠正**：① 不挤（37.1 KiB < 48 KiB，本靶不需要 opt-in），且已经落地。真正更小的下一步按风险排：
1. **按 rows 自动选路（最便宜）**：3–5 行、1 编译；prefill 保留 SMEM、decode 回 global ⇒ 拿回 −2.4%（≈+0.6 tok/s）；ids 不受影响（两路逐位同）。
2. **barrier 2→1（sB/sC 双缓冲）**：+4 KiB shared（45.3 KiB 仍 <48 KiB）、20–40 行；预估 kernel −20~40%，prefill +1~3%。
3. **② d_state 切线程（甲）**：只改 launch 一个数 + 归约；占用 6.7%→21–42%；但改数值顺序、需 ids/`--top-probs` 门，且 E1 后收益上界已缩小 ⇒ 放在 (2) 之后，先看 nsys/ ncu 是否仍把 `mamba2_scan` 放关键路径。
4. **warp 变体不必再试**：R14b 已试（tree/grouped-reduction），在本卡**输给 SMEM**，代码以 `#if 0` 留档（`:2612`/`:2784`）。

---

## §5 ⑤ 一句话结论（值不值得做/能省多少/是否位精确）
- **① 已经做完、且今天的交付已开**（E1：kernel **38.62→3.36 ms ≈11.5×**、prefill **348→729 +109%**、**ids/概率逐位一致**；代价 decode **−2.4%**）。**它不再是头号瓶颈**：旧 nsys 的 **26.7%** 是**未开 SMEM 的旧 exe**、pb=6912 坐标（**不是** 10-17 计划的 51.4%/pb=8 坐标）；今天没有 post-SMEM nsys ⇒ 推断 **3–8% 的 prefill 墙（上界 ≤11%）**。
- **还能拿的**：**按 rows 自动选路**（prefill SMEM / decode global）⇒ **decode 回 +2.4%（≈+0.6 tok/s），零数值风险**；**barrier 2→1**（+4 KiB）⇒ prefill **+1~3%**；**② 并行度** ⇒ 占用 6.7%→21–42%，但改求和顺序、收益上界已缩小。
- **是否位精确**：**① 是**（E1 实测 ids + `--top-probs` 与旧逐位同，N-A 负控能变红）；**(2)/(3) 不是**（(2) 不改算术顺序、仍应逐位同；(3) 改结合顺序 ⇒ 需重新过 ids 门）。

---

## §6 未验 / 边界
- **U1**：**没有 post-SMEM 的 nsys**（R10/R13 都早于 SMEM exe 00:35）⇒ §2.3 是推断（D）；要硬数须在今天 exe 上跑一次 nsys。
- **U2**：E1 的 ncu "38.62→3.36 ms" 与旧 nsys 在 pb=6912 的 "~21 ms/次" 是**两条不同路径/行数**（pb1024 vs pb6912），不能直接相乘外推；§2.3 已用两种独立夹逼并标 D。
- **U3**：E1 的性能臂是 **pb=1024 ⊆ 700-token** 的旧配置（当时 pb≥10 还有 NaN，后来由 `moe_rows()` latent 几何修复；今天的 1848 是修复后的 pb=6912 路径）⇒ "今天 SMEM 的增益比例" 不能直接引用 +109% 的绝对值。
- **U4**：今天 decode 的实际构成（26–29 tok/s、带 `STRATA_PROMPT_BORROW=1`）与 E1 时（56 tok/s、threads24）不同 ⇒ decode −2.4% 的绝对值需在当前交付配置上重测（但机制清楚、方向不会变）。
- **U5**：barrier 2→1 的 "+20~40% kernel" 是**推断**（ncu 显示 Compute 40.5%、占用 6.7%、原实现的 No-Eligible 98.67% 每 75 cycle 一条 ⇒ 同步/延迟受限；未实测）。
- **U6**：② 的 `d_state×S` 并行是否需额外 barrier/归约成本，取决于 S 与 barrier 合并方式（计划 §甲 已论证"净增 ≈0–1 次"，未实测）。

---

## §7 复算命令（只读）
```powershell
# ① 当前源码状态
Select-String 'PolyStrata\src\poly\kernels.cu' -Pattern 'template <bool SMEM>|__syncthreads|smem_state|STRATA_M2_SCAN_SMEM|mamba2_scan_kernel<true>|mamba2_scan_kernel<false>'
# ② E1 回执（① 的实测）
Get-Content '…\P3-20261016\回执\E1-mamba2scan-shared.md'
# ③ 旧 nsys 份额（python sqlite3，联接 StringIds）
#    SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
#    本件已算：mamba2_scan 871.8 ms / 26.75% / n=40（R13）
# ④ 今天跑法是否带 SMEM
Select-String 'C:\sp3\run_delivery_t16.cmd' -Pattern 'STRATA_M2_SCAN_SMEM'
```
**判据（能变红）**：① 当前 `kernels.cu` 里没有 `template <bool SMEM>` ⇒ §1 错；② 今天 `run_delivery_t16.cmd` 没有 `STRATA_M2_SCAN_SMEM=1` ⇒ "①已进交付"错；③ 用今天 exe 的 nsys 算出的 `mamba2_scan` 份额 >15% ⇒ §2.3 推断错（该重开 ②）；④ 在 `rows>32` 走 SMEM 的 A/B 里 ids 不逐位同 ⇒ §3.2 位精确结论错。

---

## §8 落盘
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AA-mamba2scan现状与shared可行性-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-mamba2scan现状与shared可行性.md`
