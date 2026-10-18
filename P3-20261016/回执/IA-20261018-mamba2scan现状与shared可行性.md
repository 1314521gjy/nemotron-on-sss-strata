# 回执 · 情报-AA：`mamba2_scan` 现状核实 + ①state→shared 可行性（2026-10-18）

## 1. 任务
lead 说 10-17 计划判定的头号瓶颈 `mamba2_scan_kernel`（51.4%）的 **①state→shared「从未实施」**；要求核实当前源码现状（launch/for t/barrier/state 位置）、今天的占比（今天 prefill 已 1848 tok/s）、① 的可行性（shared 预算/布局/位精确/decode 影响/行数/风险）、若太挤的更小第一步、一句话结论。零 GPU、零编译、不改码。

## 2. 状态
**达成（并纠正一个前提）**：**① 不是"从未实施"——它 2026-10-17 10:0x 做完并有回执**（`回执\E1-mamba2scan-shared.md`，A 级），**且今天的交付配置已带 `STRATA_M2_SCAN_SMEM=1`**（`C:\sp3\run_delivery_t16.cmd`）。当前源码正是那时留下的 `template<bool SMEM>` 双路径。零 GPU、零编译、不改码、未 spawn 子代理；只写本回执与详细件。

## 3. 结论（一句话）
**① 已完成并已进交付**：E1 实测 kernel **38.62→3.36 ms（≈11.5×）**、端到端 prefill **348.2→728.8 tok/s（+109%）**、**ids/`--top-probs` 与旧逐位一致**；代价 **decode −2.4%**（rows=1 摊不开 load/store）。**今天它已不是头号瓶颈**：旧 nsys（未开 SMEM 的旧 exe）里 `mamba2_scan` 是 **26.7% 的 kernel 时间（871.8 ms / 40 次）**，但那是 **pb=6912 坐标**；10-17 的 51.4% 是 **pb=8/700-token 的另一坐标（~1840 次调用）**，两者不可混用。今天**没有 post-SMEM nsys** ⇒ 推断 **3–8% 的 prefill 墙（保守上界 ≤11%）**。**剩余最划算动作 = 按 rows 自动选路**（prefill 用 SMEM、decode 用 global）⇒ **拿回 decode +2.4%（≈+0.6 tok/s）、零数值风险**；其次 **barrier 2→1**（+4 KiB SMEM ⇒ 45.3 KiB 仍 <48 KiB，预估 kernel 再 −20~40%、prefill +1~3%）；**② d_state 切线程** 收益上界已被削小且改求和顺序需重跑 ids 门。

## 4. 证据（关键读数 + 出处 + 级别）

| # | 结论 | 证据 | 级 |
|---|---|---|---|
| E1 | **① 已于 10-17 实现并验证**：ncu 单 kernel **38.62/37.08/35.64 ms → 3.36/2.50/3.36 ms（≈11.5–15×）**；Compute(SM)% 2.98→40.50；L2% 84.81→2.68；DRAM% 0.84→3.47；**dynamic shared 33.02 KB**；占用 6.87→6.66%；端到端 prefill **348.2→728.8（+109%，2.09×）**；**decode 56.36→55.00（−2.4%）**；pb=8 下 ids 三臂相同、`--top-probs` md5 三臂相同；N-A 负控（`st<d_state-1`）ids/概率都变 ⇒ 判据有鉴别力 | `回执\E1-mamba2scan-shared.md` §2–§5 | **A（实测）** |
| E2 | 当前源码仍是双路径：`template <bool SMEM>`（`:2553-2554`）、launch 形状不变 grid 128×block 64（`:2925` SMEM / `:2941` global）、`for t=0..rows` 串行（`:2572`）、**每 token 2 次 `__syncthreads`**（`:2582`/`:2606`，SMEM 未改）、默认 state 全局（`:2565`、`:2592-2597`）、SMEM 行布局 `i1*(d_state+1)`（`:2567`）、进/出各一次 load/store（`:2569`/`:2609`） | `PolyStrata\src\poly\kernels.cu`（mtime 10-18 00:35） | A |
| E3 | **① 已进今天交付**：`run_delivery_t16.cmd:5` `set STRATA_M2_SCAN_SMEM=1`，跑 `sp_borrow.exe`（R60 加 `STRATA_PROMPT_BORROW=1`：decode 25.90→28.83、ids 128/128 同、prefill −3.8%） | `C:\sp3\run_delivery_t16.cmd`；`C:\sp3\R60-delivery-driver-20261018.log` | A |
| E4 | **旧 nsys（pre-SMEM 旧 exe）**：`nsysR13.sqlite` 联接 `StringIds` 后——`mul_mat_q` 1,262.2 ms（38.73%）、**`mamba2_scan_kernel` 871.8 ms（26.75%，n=40，单次 ~21 ms）**、`quantize_mmq_q8_1` 233.7 ms（7.17%）、`relu2` 205.4 ms（6.30%）、`mamba2_conv` 144.7 ms（4.44%）、`native_mmvq_slots` 113.0 ms（3.47%）；R10 同构（845.2 ms / 26.73%） | `C:\sp3\nsysR13.sqlite`、`nsysR10.sqlite`（`CUPTI_ACTIVITY_KIND_KERNEL`+`StringIds`） | **A** |
| E5 | **无 post-SMEM nsys**：R10（23:54）/R13（00:09）都早于 SMEM 源码 mtime（00:35）；今天日志无 kernel 级打印（`mamba2_scan|M2_SCAN` 0 命中） | 文件时间戳 + 今日日志 grep | A |
| E6 | **shared 预算（本靶）**：dynamic `64×129×4 = 33,024 B` + static `8 + 2×2048 = 4,104 B` = **37,128 B ≈36.25 KiB < 48 KiB 默认 ⇒ 无需 opt-in**；launcher 仅 `need>48 KiB` 才 `cudaFuncSetAttribute`（`:2917-2923`），超设备上限有响亮回退（`:2930-2938`） | 计算 + `kernels.cu:2912-2938` + E1 §1/§2 | A |
| E7 | **位精确**：SMEM/global 两条内层**同一 st 升序、同一表达式**（`:2587-2588` vs `:2593-2594`）⇒ float 结果逐位同；E1 实测 ids + `--top-probs` md5 三臂相同（pb=8），负控能变红；且 **① 不改 `(ii,st)` 映射 ⇒ 快照代码（`nemotronhmoe.cpp:286/289`）一行不用改** | `kernels.cu:2587-2594`；E1 §4；计划 `:87-92` | A |
| E8 | **decode −2.4% 的机制与现状**：`rows=1` 时 shared 版每 call 多一次 128 load+store、摊不开（全局版在 rows=1 同样只一次读写）；今天交付把 SMEM 开在**所有**调用上 ⇒ 现在 decode 仍背这 −2.4% | E1 §5；`run_delivery_t16.cmd` | A |
| E9 | 今天推断（D）：按 E1 "kernel 份额 67.9%→8.7%" 折算到旧 26.7% ⇒ **≈3.5% kernel 时间**；按端到端 2.09× 夹 ⇒ **≤11% 墙**（今天 prefill 6792/1848 = 3.67 s）⇒ **3–8% 墙、非头号瓶颈** | 推断（输入 E1/E4/交接 §8.1 的 1848） | **D** |
| E10 | warp/tree 变体已试并禁用（`#if 0`）：R14b 在本卡**输给 SMEM** | `kernels.cu:2612`/`:2784`；`回执\R14-mamba2scan-warp-20261018.md`、`回执\R14b-mamba2scan-树形-20261018.md` | A |

**能变红的判据**：① 当前 `kernels.cu` 无 `template <bool SMEM>` ⇒ §3 错；② `run_delivery_t16.cmd` 无 `STRATA_M2_SCAN_SMEM=1` ⇒ "已进交付"错；③ 今天 exe 的 nsys 算出 `mamba2_scan` >15% 墙 ⇒ E9 推断错（要重开 ②）；④ rows>32 走 SMEM 的 A/B 里 ids 不逐位同 ⇒ E7 错。

## 5. 可行性要点（问题 3 逐条）
1. **每 block shared**：dynamic **33,024 B**（`head_dim × (d_state+1) × 4`，+1 为每行 padding），加现有 static **4,104 B** = **37,128 B = 36.25 KiB**；**< 48 KiB 默认上限 ⇒ 本靶不需要 opt-in**（launcher 只在 need>48KiB 时 set attribute；超设备 optin 上限则响亮回退）。
2. **布局/访问**：全局 `sp = s + ii*d_state`（`:2565`，`ii=h*head_dim+i1`）→ shared `ss = smem_state + i1*(d_state+1)`（`:2567`）；st 仍是全局索引、升序循环；**同一表达式、同一累加顺序** ⇒ 位精确（E1 实测 ids/概率逐位同）。
3. **对 decode**：同一 kernel 双用（rows 参数化）；rows=1 时 shared 收益为 0、拷贝是纯开销 ⇒ **E1 实测 −2.4%**；今天交付全调用开 SMEM ⇒ 这 −2.4% 仍在。
4. **快照约束**：改 `(ii,st)` 映射必须同步改快照（计划 §Q4 `:87-92`）；**① 不改映射 ⇒ 天然合规**。
5. **行数/风险**：① 本身约 40–60 行（已实现）；剩余 (i) rows 自动选路 3–5 行（零数值风险）；(ii) barrier 2→1 双缓冲 +4 KiB、20–40 行（预估 kernel −20~40%，prefill +1~3%，推断）；(iii) ② d_state 切线程（launch 改一个数 + 归约，占用 6.7%→21–42%，但改求和顺序 ⇒ ids 门）。

## 6. 更小第一步（问题 4）
1. **按 rows 自动选路**（最便宜、最稳）：`smem_state && rows > 32` 才走 SMEM，decode 回 global ⇒ **decode 拿回 +2.4%**（当前 26–29 tok/s 约 +0.6–0.7），prefill 不变；1 编译、无 ids 风险。
2. **barrier 2→1（sB/sC 双缓冲）**：+4 KiB（45.3 KiB 仍 <48 KiB），20–40 行；预估该 kernel 再 −20~40% ⇒ prefill +1~3%（D 级推断）。
3. **② d_state 切线程**（计划"甲"）：放在 (2) 之后、先用今天 exe 的 nsys 确认 `mamba2_scan` 仍在关键路径（E9 推断它只剩 3–8%）。
4. **warp 变体不必再试**（R14b 已判负）。

## 7. 未验 / 边界
- **U1**：**无 post-SMEM nsys**（R10/R13 都早于 SMEM exe）⇒ "今天的占比"是 D 级推断；要硬数须在**今天 exe**上跑一次 nsys（同 `run_delivery_t16.cmd`，`--max-new 1` 缩短），按 §4 的 sqlite 联接法算。
- **U2**：E1 的 ncu 38.62→3.36 ms 是 **pb≈1024/700-token** 路径；旧 nsys 的 ~21 ms/次是 **pb=6912/6791-token** 路径 ⇒ 两条路径不可直接相乘外推（§4-E9 已用两种独立夹逼）。
- **U3**：E1 性能臂处于 **pb≥10 还有 NaN 的旧代码**（后来由 `moe_rows()` latent 几何修复）；今天的 1848 是修复后的 pb=6912 路径 ⇒ 不能直接引用 +109% 的绝对值当"今天开 SMEM 的增益"。
- **U4**：今天 decode 26–29（带 `STRATA_PROMPT_BORROW=1`、threads16）与 E1 的 56 tok/s（threads24）不同 ⇒ −2.4% 的绝对值需在当前配置重测（方向/机制清楚）。
- **U5**：barrier 2→1 的 "+20~40%" 是推断（ncu：Compute 40.5%、占用 6.7%、旧实现 No-Eligible 98.67%、每 75 cycle 一条 ⇒ 同步/延迟受限；未实测）。
- **U6**：② 的跨段归约成本（S=4/8 的 barrier 合并）计划已论证"净增 0–1 次"，未实测。

## 8. 落盘（绝对路径）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AA-mamba2scan现状与shared可行性-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-mamba2scan现状与shared可行性.md`
- 关键旁证：`…\回执\E1-mamba2scan-shared.md`（① 的实测）· `C:\sp3\nsysR13.sqlite`（旧 nsys）· `C:\sp3\run_delivery_t16.cmd`（今天开 SMEM）· `…\计划-分环节定位杠杆-20261017.md:25-95`（51.4% 的原始坐标）
