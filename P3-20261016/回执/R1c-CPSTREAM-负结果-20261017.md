# 回执 · R1c：A3 生产形态判绿 + CP_STREAM 负结果 — 2026-10-17

## 任务
按 lead 指令：A3/A3b（零改码、旧 exe）验生产形态 → 绿了才编译 → **一次编译只做一件**：`STRATA_MOE_ROWS_CP_STREAM`（轮内拷贝第二流）；旗标关/开两臂；记账三件。lead 追加口径：call 只许写"推算"；参照系 A0 组 4 跑中位（DMA 89,646 / 墙 10.501 / prefill 646.8）；`--top-probs` 跑不计 decode；A3 与 CP 开臂各 3 个非-probs 跑。

## 状态
**达成；按 lead 最新指令已停（CP_STREAM 不再调）。**
- A3/A3b 判绿（零改码，旧 exe sha `cc2a9851…`）：**生产形态 = `--prompt-cache-every 8192`，不需要改码保 checkpoint**。
- CP_STREAM 编译两版（`bab819eb…` 第一版；`6f71aca6…` 第二版 = 第一版 + 旗标自证打印），**两版都是零效果 = 负结果**；**第二版没有加双缓冲**（只是同一"单 stage_ + 一条新流"设计 + 自证行），故按 lead 规则**驻留/DMA 两列不适用**，并已按指令停手。
- 12 跑全部非-probs、一次一跑；每跑前打印 exe sha、断言无引擎进程、`copy /Y p_master.txt p_run.txt` 冻结 profile、报 nvidia-smi 显存；45 ids 全量落盘；全部跑后无残留进程、GPU 942 MiB / 3%。

## 结论（一句话）
**A3 判绿**：新 exe + `--prompt-cache-every 8192` + 旗标关 = prefill **1354.4 / 1359.3 / 1330.5**（3 跑）、DMA **28,792**、体积 **44.58 GB**、chunk **111**、驻留 **5,890**、ids **45/45** 逐位同（`call≈1` 是推算，非读数）。**CP_STREAM 判负**：同 sha（`6f71aca6…`）旗标关→开，墙 **5.0148→4.9915 s（−0.46%）**、prefill **1354.4→1360.7（+0.47%）** ⇒ 噪声内、零效果。**结论：只有一张 `stage_` 缓冲 ⇒ 下一层专家无处预装，加一条流不可能产生重叠；真重叠必须双缓冲，而双缓冲要付驻留槽 + DMA 体积两笔账（本轮未 build、未测）。**

## 证据（读数 + 口径）

**A3 / A3b（旧 exe `cc2a985128379b49b7c5682474d3fd54b3f92a14eea90518740eac7158b88f89`，日志见落盘）**

| 臂 | 配置 | prefill tok/s | 墙 s | decode | DMA | 体积 GB | chunk | 驻留 | ids |
|---|---|---|---|---|---|---|---|---|---|
| A3 | every=8192 | 1358.9 / 1354.3 / 1358.7（中位 **1358.7**） | 4.9982/5.0151/4.9989 | 12.89/12.29/12.17 | 28,792 ×3 | 44.58 | 111 ×3 | 5,890 | 45/45 |
| A3b | every=6912 | 1352.4（1 跑） | 5.022 | 11.08 | 28,792 | 44.58 | 111 | 5,890 | 45/45 |

- A3 相对 lead 参照（646.8）= **+110.06%**；DMA 相对 89,646 = **−67.9%**。判据（prefill ≥1300、DMA≈28,792、ids 45/45）全绿。
- A3/A3b 均与 A1（every=0）同 DMA/chunk ⇒ `every=8192/6912` 与 `0` 在本 prompt（6792 < 8192/6912）上给同一"1 个 call"（**call≈1，推算**；`marks` 行不存在于日志，无直接读数）。

**CP_STREAM（两版 exe；主对照 = 同 sha `6f71aca6…` 的旗标关 vs 开）**

| 臂 | exe sha | prefill（3 跑） | 墙（3 跑） | decode（3 跑） | DMA | 体积 | 驻留 | ids |
|---|---|---|---|---|---|---|---|---|
| 旗标关 NC | `6f71aca6…` | 1359.3 / 1354.4 / 1330.5 | 4.9967 / 5.0148 / 5.1048 | 12.12 / 12.10 / 11.56 | 28,792 | 44.58 GB | 5,890 | 45/45 |
| **旗标开 CPf** | `6f71aca6…` | 1359.2 / 1360.7 / 1363.0 | **4.9971 / 4.9915 / 4.9831** | 10.89 / 11.71 / 11.78 | 28,792 | 44.58 GB | 5,890 | 45/45 |
| 旗标开 CP（第一版） | `bab819eb…` | 1362.4 / 1363.5 / 1354.2 | 4.9853 / 4.9813 / 5.0155 | 12.08 / 11.75 / 12.23 | 28,792 | 44.58 GB | 5,890 | 45/45 |

- **同 sha 配对**：NC 中位墙 **5.0148 s** → CPf 中位墙 **4.9915 s** = **−0.46%**；prefill 中位 1354.4 → 1360.7 = **+0.47%**；两组区间（NC 4.9967–5.1048 / CP 4.9813–4.9971）重叠 ⇒ **零效果（负结果）**。CP vs A1（旧 exe 5.010 s）= **−0.37%**。
- **开关生效自证**：`6f71aca6…` 的 3 个旗标开日志各有一行 `strata-poly: STRATA_MOE_ROWS_CP_STREAM=1: a prompt batch's round copies run on a stream of their own`；旗标关日志无此行 ⇒ 不是"开关没生效"的假零。
- **两列核查（lead 铁律）**：若 build2 加了双缓冲，必须报驻留↓与 DMA↑。**本轮没有双缓冲**——证据：12 跑全部 **驻留 5,890 / DMA 28,792 / 44.58 GB 逐位不变**（双缓冲要让出 1.586 GB ⇒ 驻留 ≈5,378、DMA 会上涨）。⇒ **两列不适用；按 lead 指令停手，不再调。**
- **结构性负结果（file:line）**：每层拷贝 = ahead（`up_` 流，`engine.cpp:2403-2409` 早拷贝 + `:2737` st_ 等 ev_ahead_）≈341 专家，与 round 拷贝（`:2741`，`stage_` 内）≈24 专家 ⇒ **93–95% 的搬运字节在 `up_` 上、已在"自己一条流"里**；第二流只能重叠 round 拷贝 ≈2.3–3.0 GB/44.58 GB（5–7%），按 17.7 GB/s 上界 ≈0.13–0.17 s（**2.6–3.4%**），实测 0.46%（噪声）。而 `stage_` 只有一张（`engine.cpp:1768`；`ev_free_` 在 `:2785`、`up_` 在 `:2403` 等它）⇒ 下一层的 ahead 拷贝不能早于上一层 kernel 完成 ⇒ **加流不可能重叠；要重叠必须 N≥2 个 stage 槽（双缓冲/ring），且要改 `ev_free_`/`ev_ahead_` 那套配对（红线），代价 = 驻留 −512 槽（1.586 GB）+ DMA 体积上升（按 lead 粗算 ~+2.2 GB，约 46.8 GB）。**
- **decode**：CPf 中位 11.71 vs NC 中位 12.10 = **−3.22%**（略超 3% 线，但本会话单跑散度 11.3–12.8；CP 不采纳，未据此下任何 decode 结论）。
- 所有跑 `realWarn=0`（pin 的 "no error (0)" 是正常自印，非告警）。

**记账三件（新 exe）**
1. **sha256**：`6f71aca6d76393b33d3e83a16f927d2ba9371c1fa0841a3ff1527ef86165f421`（size 65,413,632；build 时间 2026-10-17 22:32:04；第一版为 `bab819eb363eee7587cbf75c7a7221cf0b00a45ae39d060deb9a0c07fb1c9de9`）。
2. **源码改动清单**（只动这两个文件，旗标默认 0）：
   - `PolyStrata\src\poly\engine.hpp:630-636`：成员 `cp_ / ev_krn_ / ev_copy_`（默认 null）。
   - `PolyStrata\src\poly\engine.cpp:1781-1800`：`STRATA_MOE_ROWS_CP_STREAM=1` 时建 `cp_` 流 + `ev_krn_/ev_copy_`；开关状态自证行（`:1789-1790`）。
   - `PolyStrata\src\poly\engine.cpp:2723-2751`：round 拷贝改走 `cp_` + `ev_krn_`（等上一轮 kernel 读完 stage_）/`ev_copy_`（kernel 等拷贝）；**旗标关时走原 `st_` 路径，逐字节同旧**。
   - `PolyStrata\src\poly\engine.cpp:2780-2783`：每轮 kernel 后记 `ev_krn_`。
   - 析构：`engine.cpp:389`（同步 cp_）、`:397-399`（销毁两事件）、`:409`（销毁 cp_ 流）。
   - **红线四件套未动**：`engine.cpp:2407`（`up_` 等 `ev_free_`）、`:2409`（记 `ev_ahead_`）、`:2737`（`st_` 等 `ev_ahead_`）、`:2785`（记 `ev_free_`）——逐字仍为原文。
3. **完整构建命令**（cwd = `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata`）：
   `call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"` → `"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build-p3 --target strata-poly -j 24`；脚本 `C:\sp3\build_cpstream3.cmd`；日志 `C:\sp3\build_cpstream.log`（第一版）/`C:\sp3\build_cpstream2.log`（第二版）；两次均 `EXIT=0`、无编译/链接错误。

## 未验 / caveat
1. 未跑 nsys：CP_STREAM 的收益上界（2.6–3.4%）与"单 stage_ 不可重叠"是**结构推导（A 源码 + B 计数器）**，无 H2D 时间线实测。
2. 双缓冲（N≥2 stage 槽）**未 build、未测**：其驻留 −512 槽 / DMA +~2.2 GB / 净收益"墙 <4.2 s 才算赢"的三项账未验。
3. decode 单次 45 token 散度 11.3–12.8，3 跑中位仍有 ±3% 量级不确定；CPf 的 −3.22% 不能判为真效应。
4. `call≈1` 为推算（marks 不出现在日志），本轮只把 `pin_chunks / direct DMAs / prefill / decode / 驻留` 当读数。
5. 本轮零改码之外的源码改动 = 上述 CP_STREAM 一处（旗标默认 0）；未碰 `E:\strata`、`E:\strata-v01431`、`E:\rlab`。

## 落盘（绝对路径）
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R1c-CPSTREAM-负结果-20261017.md`
- 日志（12 跑）：`C:\sp3\R1a-A3-20261017.log` · `C:\sp3\R1a-A3-2-20261017.log` · `C:\sp3\R1a-A3-3-20261017.log` · `C:\sp3\R1a-A3b-20261017.log` · `C:\sp3\R1c-NC1-20261017.log` · `R1c-NC2-…` · `R1c-NC3-…` · `R1b-CP1-…` · `R1b-CP2-…` · `R1b-CP3-…` · `R1c-CP1-…` · `R1c-CP2-…` · `R1c-CP3-…`（均 `C:\sp3\`）
- 运行脚本：`C:\sp3\run_R1a_A3{,-2,-3}_20261017.cmd` · `run_R1a_A3b_20261017.cmd` · `run_R1c_{NC1,NC2,NC3,CP1,CP2,CP3}_20261017.cmd` · `run_R1b_CP{1,2,3}_20261017.cmd`
- 构建脚本/日志：`C:\sp3\build_cpstream3.cmd` · `C:\sp3\build_cpstream.log` · `C:\sp3\build_cpstream2.log`
- exe：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\build-p3\strata-poly.exe`（`6f71aca6…`）· 运行副本 `C:\sp3\sp_cp.exe`（同 sha）· 旧基线仍备于 `C:\sp3\sp_tile.exe` 与 `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\_backups\R1-baseline-20261017\strata-poly.exe`（`cc2a9851…`）
- 方案件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\施工-根因方案-20261017.md`（§10 结果回填）
