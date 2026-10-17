# 回执 · P0-b（`mmvq_il_parity --bench --emit-table` 实测 sm_89 IL 表）

- **任务**：P0-b —— 用 `mmvq_il_parity --bench --emit-table` 实测 sm_89 的 IL 表（→ `kIlRows_89`），判是否加表。
- **状态**：**达成（结论 = 不改码）**
- **结论一句**：**逐位对拍全过（表可用），但表对我们用到的类型是"全 0" ⇒ 加表对本靶无收益 ⇒ 建议不改码。**
- **证据（A 级本机实测；口径注明）**：
  1. **构建**：`cmake --build build-p3 --target mmvq_il_parity`（只编该 target；**编后 `strata-poly.exe` sha 仍 = `978d64bc…`** ⇒ 读数绑定未被破坏）→ 产物 `PolyStrata\build-p3\mmvq_il_parity.exe` ✓。
  2. **校验行**：`mmvq_il_parity: OK (every case, T 1-4, rows 1/2/4 and the table, bitwise)`（`il_bench.log` 末）⇒ **逐位对拍全过** ⇒ 表可用（scheduler 的 §1.2 前提满足）。
  3. **emit 块（`il_bench.log` 尾）**：
     ```
     {23, {0, 1, 0, 1, 0}},   // IQ4_XS
     {12, {0, 0, 0, 0, 0}},   // Q4_K   ← 我们的主类型
     {13, {0, 0, 0, 0, 0}},   // Q5_K
     {14, {0, 0, 0, 0, 0}},   // Q6_K
     {11, {0, 0, 1, 1, 1}},   // Q3_K
     ```
  4. **★ 对我们 = 无收益**：**`ffn_up_exps`/`latent_*`/`sh_up` = Q4_K(12) → 表给 `{0,0,0,0,0}`（走 plain kernel）**；**`ffn_down_exps` = IQ4_NL(ggml type 20) → 根本不在表内**（scheduler §1.3，A）⇒ **三个投影里没有一个能吃到 IL**。
  5. **性质**：IL 表**只影响速度、不改位**（`docs:3-5`）⇒ 即使有非 0 项也不动正确性；本条只判"值不值得加"。
- **未验 / 边界**：
  - **未做端到端 prefill 验证**（因表对我们是 0，**改码本身没意义** ⇒ 未做）。
  - **"每个投影是否真的走 `native_mmvq_il`" 未逐处核**（scheduler §3.3 亦标未验）；且**新 canonical 的 prefill 走 `moe_rows()`（`Mmq::mul`），不过 `native_mmvq`** ⇒ **表的作用面本就只剩 decode**（我方推论，未验）。
  - **`emit` 只覆盖 type 11/12/13/14/23**；换量化类型时本结论需重做。
- **建议**：**不改码**（加表 = 加一段对本靶恒为 0 的死表）。若将来把 `ffn_down_exps` 换成表内类型（如 IQ4_XS/Q4_K），**再重跑本 bench**。
- **落盘（绝对路径）**：
  - `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\il_bench.log`（全量 BENCH + emit 块）
  - `…\P3-20261016\build_il.cmd` / `build_il.log`（构建命令与日志）
  - `…\P3-20261016\il_win.log`（窗证据：无 strata）
  - `…\P3-20261016\回执\P0-b.md`（本件）
