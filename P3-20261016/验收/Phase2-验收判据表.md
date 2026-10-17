# Phase 2 验收判据表 — Nemotron pb=6912（P3-20261016）

> ★ **2026-10-17 22:2x 口径变更（lead 裁定）：历史 652.6 单点绝对门作废，改用"同日同 exe 同 config 多跑中位"；先读文末 §8。**
> **席位**：验收（reviewer）· **任务**：`task-4` · **建立**：2026-10-17
> **本表是"读数 → 红/绿"的唯一判据页**；所有基线数值的**唯一归属处** = 下面的基线文件（本表只引用与派生，两者冲突以基线文件为准）。
> **用法**：施工交来新日志后，按 §2 三条命令跑一遍；每条读数的红/绿/琥珀按 §3 表判；红/琥珀的处置按 §4。

---

## §0 基线（先确认你没拿错基线）

| 项 | 值 / 指针 |
|---|---|
| **基线文件（机器可读 · 唯一归属处）** | `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\baseline-pairD-pb6912-tile1.txt`（sha256 `d8cd92f269b0f6f1…`） |
| **基线来源（逐字抽取）** | `C:\sp3\pair_D_pb6912.log` 行 `1`(命令行) · `84`(tile 自证) · `86`(驻留) · `87`(读数) · `88`(DMA) · `89`(ids) · `90`(EXIT=0) |
| **D 臂读数**（pb=6912 + `STRATA_SCRATCH_TILE=1` + PIN=59.1 + SMEM=1） | prefill **652.6 tok/s** · decode **12.30 tok/s** · 驻留 **5890 of 20480**（17.0 GiB）· DMA **91,076 次** avg 1.48 MiB（校正体积 **141.0 GB**；旧口径 134.8 GB）· ids **45 个** · EXIT=0 |
| **§1.3 前 12 个 ids 对上？** | **对上**（A 级，验收自测）：`10041,1278,4206,1636,9956,1317,6347,1261,19940,1044,3147,1278`；`compare-ids.ps1 -VerifyBaseline` 输出 `verify_first12_vs_handoff1.3=ok`、`verify_ids_sha256=ok`、`ids_count=45`。基线 ids sha256 = `7d1e67f027b6f888b47798dfeb74387e97c826f7bf089dc1037fe902ed5add2f` |
| **exe 绑定（证据等级 B）** | 跑法脚本 `C:\sp3\run_pb6912_D.cmd:12` → 实跑 `C:\sp3\sp_tile.exe`，sha256 `cc2a985128379b49b7c5682474d3fd54b3f92a14eea90518740eac7158b88f89`，mtime `2026-10-17 12:54:02`；同 sha 副本 = `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\build-p3\strata-poly.exe`；构建 `C:\sp3\build_tile3.log`（`[79/79] Linking CXX executable strata-poly.exe`，EXIT=0，12:54:01）；日志 mtime 12:55:57 > exe mtime ⇒ 读数在该 sha 之后产生。**日志本身不打印 sha** ⇒ 只能算 B 级（Phase 2 要求 A 级：把 sha 写进日志头，见 R2） |
| **参考负控臂（同日同 exe）** | C 臂（tile=0/pb=6912）= `C:\sp3\pair_C_pb6912.log`，**573.9 / 10.46 / 4244 / 112,450**，ids 逐位同上，可绑 sha；B2 臂（tile=1/pb=1024）= `C:\sp3\pair_B2_pb1024.log`，**651.2 / 13.62 / 7135 / 76,792** |
| **probs（逐位概率）** | **未提供**：pair 四跑均无 `--top-probs`；全实验区也搜不到该 prompt 的 `top` 行（0 命中）⇒ **probs 门当前 = 未验，不能判绿**（补齐条件见 R13） |

> 基线口径更正（照 22:00 横幅，不改原文只留指针）：`avg=1.48` 的单位是 **MiB**（`main.cpp:726-729` 除 `(1<<20)`）⇒ 校正体积 **141.0 GB**（= 91,076 × 1.548288 MB，`情报-A-字节账-20261017.md` §1.3）；①b 回执里的 134.8 GB 是旧口径。

---

## §0.6 GPU 开跑前置（硬闸门 · 决策记录 2026-10-17）

> **要求出处**：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\计划-P0派单-20261017.md:3` 与 `...\P3-20261016\计划-R2R3-槽偏斜与显存纪律-20261017.md` §纪律-1：**GPU 开跑前必须先报「几臂 / 几分钟 / 占多少显存与内存」，得到同意才跑；不许连轴跑。**

任何要占 GPU 的批（含 P1 槽配额、P2 显存纪律、此后所有 GPU 臂）**开跑前必须给 lead 这五项，缺一项不许起跑**：

1. **几臂**：L 条臂 × 每条 R 跑（旗标关/开、负控、top-probs 跑分开列）。示例：`关×3 + 开×3 + 开关各1 top-probs = 8 跑`。
2. **几分钟**：每跑预估 × 臂数给总区间（现基线：加载 ~12.5 s + prefill ~5 s + decode ~4 s ≈ **25–40 s/跑**）。
3. **占多少显存与内存**：**预估峰值 VRAM（MiB）** + RAM 峰值（GiB）；带 ring 时最紧（见 R21/R24）。给"预估峰值"，不是"还剩多少"。
4. **跑序 + profile 冻结 + 每跑前断言**：每跑前 `copy /Y p_master.txt p_run.txt`；断言无引擎进程、显存空闲、exe sha 与声明一致。
5. **能不能停**：说明从哪一步可中断、中断后哪几跑要重跑。

**判据（怎么变红）**：① **没报就起跑 ⇒ 该批读数整批作废**（决策记录，不是建议）；② 实测峰值 VRAM 超出所报预估 **>20%**，或实际时长超出所报 **>50%** ⇒ 该批降级为"未按预算执行"，需重报后重跑；③ 连轴跑（批间不停/不报）⇒ 后一批作废。

---

## §1 三个文件与三条命令

| 文件（绝对路径） | 作用 |
|---|---|
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\compare-ids.ps1` | ids 硬门：逐位相同？首个分歧位置（1 基 token 位）· 两侧 ids · `max_abs_delta_p` · 输出 `VERDICT`，退出码 0 绿 / 2 红 / 3 解析错 |
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\extract-metrics.ps1` | 读数抽取 + 臂判据：prefill/decode · 驻留槽 · DMA 次数/avg/体积 · ids 数+sha · EXIT/WARNING · 引擎自证行 · exe sha 核对；`-Arm` 出 `VERDICT`，退出码 0 绿 / 1 红 / 2 琥珀 / 3 解析错 |
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\负控\` | 负控/自测件（见回执 §红臂）：`neg_shift1.ids.txt`（错位 1 token）· `neg_mid_sub.ids.txt`（第 13 位改 1 个）· `neg_drop_first.ids.txt` · `pos_identical.ids.txt` · `synth_probs_*`（概率路径自测）· `synth_Aarm_*`（判据方向自测，**合成件，不是引擎证据**） |

```powershell
# 在 PowerShell 会话里（本机 PowerShell 5.1 已实测；脚本 ASCII-only + [IO.File]/[regex]，中文路径安全）
$acc = 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收'

# (0) 基线自检 —— 每次复核前先跑，防拿错基线 / 防基线文件被改
& "$acc\compare-ids.ps1" -VerifyBaseline

# (1) ids 硬门 —— 每一臂都跑
& "$acc\compare-ids.ps1" -New 'C:\sp3\<新日志>.log'

# (2) 读数 + 臂判据 —— 多跑（≥3）时把日志都传给 -Log
& "$acc\extract-metrics.ps1" -Log 'C:\sp3\<新日志1>.log','C:\sp3\<新日志2>.log','C:\sp3\<新日志3>.log' `
    -Arm A -ExePath 'C:\sp3\<新exe>.exe' -ExpectExeSha '<声明sha>' -RequireTile 1 `
    -RequireFlagWitness '<新机制启动自证串，如 STRATA_MQ_TILE>'
# 臂：A=新机制ON/tile=1/pb=6912；REF-D 或 NEG1=旗标OFF/tile=1/pb=6912（回 652.6/12.30/5890/91076）；NEG0=tile=0/pb=6912（回 573.9/10.46/4244/112450）
```

---

## §2 判据总表（每条都写了"怎么变红"）

> 约定：**A**=本机本轮实测可自证 ｜ **B**=同机旁证（推断/脚本 echo）｜ **D**=外推。tok/s 类读数一律带噪声上限。

| # | 判据（读数） | 绿 | 红（怎么变红） | 琥珀 | 等级 | 判据有效性（已实测的负控） |
|---|---|---|---|---|---|---|
| **R1** | **ids 45 个逐位** | `identical=true`、`mismatch_count=0`、双侧 45 | 任一位置不同 ⇒ 报 `first_diff_pos` + 两侧 id | — | A（确定性） | 合成错位 1 token ⇒ RED@pos1（`old=10041 new=1278`）；真实坏跑 `pair_B_pb1024.log` ⇒ RED@pos1；`tile_pb1024.log` ⇒ RED@pos1 |
| **R2** | **新读数绑定新 exe sha** | 日志头含 64-hex sha 且 == 声明 sha；`-ExePath` 实算 sha == 声明；日志 mtime > exe mtime | sha 缺失 / 不等 / exe mtime 晚于日志（陈旧日志冒充新读数）⇒ RED | 日志无 sha ⇒ 绑定只能算 B 级（`exe_sha_in_log=AMBER`） | B（现状）/ A（补打 sha 后） | 故意声明全 0 sha ⇒ `gate_exe_sha=RED`（已实测）。★ 真实教训：①b 的 `pair_A`/`pair_B` 两跑（12:51）所用 `C:\sp3\sp_tile.exe` 已于 **12:54:02 被 build_tile3 覆盖**，中间版 sha 盘上不可追 ⇒ 这两跑不能当硬门证据 |
| **R3** | **A 臂 prefill** | 3 跑中位 > **652.6** | 中位 ≤ 652.6；或 n<3 就判死 | n=1（暂绿，待复现；`gate_replication=AMBER`）；或单跑落在 652.6×(1±1%) 内（噪声带） | A 读数 / 噪声 D | 把 A 判据套在基线自己身上 ⇒ `gate_prefill=RED`（==652.6 不算变快，已实测）。同配置 p56 三跑 prefill 606.7/608.6/607.2（极差 0.31%） |
| **R4** | **A 臂驻留槽**（结构证据） | ≥ **6104**（=5890+214，只分块 `mq_` 的下限）；到 **6294**（+404，加 `dx_`）为 A 项字面上限 | **仍 == 5890**（VRAM 没腾出来 ⇒ prefill 增益无结构解释）；或 `Δ槽×20.1 MB` 与施工给出的"源码缩掉字节数"对不上（对账失败） | 5891–6103（低于 mq_ 分块下限） | 计数器 A / 期望窗口 **D**（`情报-A-字节账-20261017.md` §3.4：A 项能到 6900 是**不成立**的，字面上限 +404 槽） | 用 NEG1 判据套 tile=0 日志 ⇒ `gate_resident=RED`（got 4244，已实测） |
| **R5** | **A 臂 DMA 次数** | 中位 ≤ **88,344**（=91,076×0.97，降 >3% 才出噪声） | 中位 ≥ **91,076**（完全没降） | 88,345–91,075（在 ±3% 噪声内） | 计数器 A / 期望值 **D**：按 C→D 斜率 12.99 DMA/槽外推，+214 槽 ⇒ −2,780 次；+404 槽 ⇒ −5,250 次（仅 2 个点，属外推） | 同上（NEG0/NEG1 的 DMA 窗彼此能互打红） |
| **R6** | **decode 12.30 不许掉**（`KP-UNI-0218`：预填借槽让 prompt 快 2–5% 但 decode 慢 10–23%；`KP-UNI-0250`：prefill/decode 机制不可互替，判据=单变量拆臂+decode 中位） | 3 跑中位 ≥ **12.30** | 3 跑中位 < **11.69**（−5%），或任一跑 < **11.32**（−8%，跌出单跑噪声带） | 中位 11.69–12.29；或 n=1 且 11.32–12.29（必须复跑）；n=1 且 ≥12.30 只算**暂绿** | A 读数 / 噪声上界 B | 这是本表**最容易被假绿蒙过去**的一条（见 §5）；单跑噪声上界证据：`C:\sp3\p56_{1,2,3,warm}.log` decode 12.23/12.80/11.30/12.50（同配置，**max-new=8 口径**，45-token 跑的噪声应更小，但未测） |
| **R7** | **负控 REF-D / NEG1**（新旗标 **关** + tile=1 + pb=6912，"逐字回旧值"） | prefill 652.6±2% · decode 12.30±8%（单跑）/±6%（3 跑中位）· 驻留 **恰好 5890** · DMA 91,076±3% · ids 逐位 · `witness_tile` 在场 | 任一项越界 ⇒ **新代码在默认路径上改变了行为**（或这条读数不是声明的 exe/flags 出来的）⇒ 该臂无效 | — | A（计数器/ids）+ B（tok/s） | `extract-metrics.ps1 -Arm NEG1 -Log C:\sp3\pair_D_pb6912.log -RequireTile 1` ⇒ 全绿（已实测）；换成 `pair_C_pb6912.log` ⇒ `tile_witness`/`prefill`/`resident` 三项 RED（已实测） |
| **R8** | **负控 NEG0**（`STRATA_SCRATCH_TILE=0` + pb=6912） | prefill 573.9±2% · decode 10.46±8% · 驻留 **恰好 4244** · DMA 112,450±3% · ids 逐位 · `witness_tile` **缺席** | 任一项越界 ⇒ tile=0 路径被新代码污染 ⇒ RED | — | A（计数器/ids） | 已实测 pair_C 全绿 |
| **R9** | **新机制启动自证行** | 日志存在 `strata-poly: <新旗标>: …` 自证行（像 ①b 的 `STRATA_SCRATCH_TILE: 8192 rows a batch (0.27 GiB …)`） | **无自证行 ⇒ 整次判据无效**（旗标可能根本没生效；假绿形态 3："没报错"≠"按我以为的路跑了"） | — | A（自证设计） | 反例已在盘上：`STRATA_M2_SCAN_SMEM` 在正常路径**零自证**（`PolyStrata\src\poly\kernels.cu:2715` 读 env，只有超限才 WARNING）⇒ 漏掉它时 prefill 从 624.8 静默掉到 336.9，日志里没有任何提示 |
| **R10** | **混杂臂闸门**（同日同机同 flags） | header 含 `--prompt-batch 6912 --max-new 45 --no-draft --lookup 0`，env 含 `PIN=59.1 SMEM=1 TILE=1`；且 `--top-probs` 状态与其他臂一致 | 缺任一 ⇒ 该臂不是"一次一变量"对照，读数作废（交接件 §3.1 三条混杂臂：`--dump-layers`/`--no-gpu-experts`/`tile_class`） | header 是脚本 echo（B 级），环境变量未打印 ⇒ 只能算 B | B | — |
| **R11** | **EXIT / WARNING** | `EXIT=0` 且无 `WARNING`/`REFUSED`/`ERR` 行 | 否则 RED（兜底路径被执行 = 声称覆盖不到那条路） | — | A | ①b 的 `maxc>tile` 兜底会打印 `WARNING: STRATA_SCRATCH_TILE: …`（`engine.cpp:2522`）—— 在场即说明不是 tiled 路径 |
| **R12** | **profile 冻结** | 每跑前 `copy /Y p_master.txt p_run.txt`（`C:\sp3\run_pb6912_D.cmd:11`），跑序可查 | 未冻结 ⇒ 驻留/DMA/decode 与历史不可比，ids 门仍有效 | 日志无自证 ⇒ 需施工把该行写进回执 | B/C | — |
| **R13** | **probs 逐位**（`--top-probs`） | 基线 `probs=` 行回填 45 个逐位概率后，按 `max_abs_delta_p` 判 | **当前不能判绿**：基线无 probs ⇒ 任何 probs 声称 = 未验 | 补齐后：`max_abs_delta_p ≤ 1e-6` 绿；> 1e-6 按交接件 §1.3 **先报首个分歧 + max\|Δp\| 给队长判，不许自动判失败** | 未验 | 概率路径已用合成件自测：`max_abs_delta_p=0.0001`、`probs_flag=DRIFT_GT_1E-6`（脚本能力已验证，**不是引擎证据**） |

---

## §3 收读数时的判读顺序（照这个顺序，前一步红就别看后面）

1. **基线自检**：`compare-ids.ps1 -VerifyBaseline` 必须 `VERDICT=GREEN`。
2. **R2 sha 绑定**：`-ExePath` + `-ExpectExeSha`；日志 mtime 必须晚于 exe mtime；日志里没有 sha ⇒ 只能给 B 级结论。
3. **R1 ids 硬门**：`compare-ids.ps1 -New <日志>`；红 ⇒ **该跑直接作废**，不用看 tok/s（真实反例：`pair_B_pb1024.log` 709.7 tok/s 但 ids 是退化循环）。
4. **R10 臂隔离 + R9 自证行**：缺自证/混杂 ⇒ 该臂无效。
5. **R4/R5 确定性计数器**（驻留、DMA）：这是"结构上是否真省了"的硬证据；与施工的"源码缩掉字节数"对账。
6. **R3/R6 tok/s**：n≥3 才判硬绿/硬红；n=1 ±噪声带内 ⇒ 琥珀，要求复跑（GPU 窗口由队长批）。
7. **R7/R8 负控**：每批读数必须至少有一条"旗标关"的负控日志与 A 臂同日同 exe；**负控不过 ⇒ 整批不可信**。
8. 出四选一结论（达成/部分/未达成/阻塞），每条读数标 A/B/C/D，未验单列。

---

## §4 红/琥珀怎么处置

| 状态 | 处置 |
|---|---|
| **红（确定性项）** ids / 驻留 / DMA / EXIT / sha | 直接退回施工：附 `first_diff_pos` 或计数器读数 + 对应基线行号；**不替施工改产物** |
| **红（tok/s 项）** prefill / decode | 先看是否 n=1：n=1 的 tok/s 红**不能单独判死**（噪声未锁）⇒ 要求 ≥3 跑中位（队长批 GPU 窗口）；n≥3 中位越界 ⇒ 判红 |
| **琥珀** | 一律"待复现"：写清缺哪一跑 / 缺哪条自证行；不许当绿 |
| **负控不过** | 整批读数无效，先查"旗标是否真的关/开"（R9 自证行）、"exe 是不是声明的那版"（R2），再查跑法（R10） |

---

## §5 ★ 最容易假绿的一条（本席裁定）

**R6「decode 12.30 不许掉」用单跑判 = 最容易被假绿蒙过去。**

- **为什么**：decode 的单跑噪声实测上界 **−8.6%（11.30）～+4.1%（12.80）**（`C:\sp3\p56_1.log` 等 4 跑同配置，且还是 max-new=8 的小样本口径）。⇒ 一次跑出 **12.4 的"绿"完全可能与真实 −10% 的伤害共存**；反过来 11.6 也可能只是噪声。而 `KP-UNI-0218` 恰好记录过：预填借槽这类"动 prefill 的机制"让 prompt 快 2–5% 的同时 **decode 慢 10–23%** —— 正好落在噪声带里，**单跑必假绿**。
- **怎么防**：(a) 3 跑中位（R6 红/琥珀门）；（b) **同日同 exe 的旗标关负控**做同窗口对照（R7 decode 12.30±8%）；（c) 同时看驻留/DMA：若驻留涨了 decode 反而掉，机制伤 decode 的嫌疑最大（`KP-UNI-0250`：prefill 与 decode 机制不可互替）。
- **亚军**：**"负控臂旗标关 ⇒ 逐字回旧值"用旧 exe 顶包**。日志不打印 sha 时，拿上一版 exe 跑"旗标关"会让负控**恒绿**且毫无信息；本机已有实物教训——①b 的 `pair_A`/`pair_B` 用的 `C:\sp3\sp_tile.exe` 在 12:54:02 被新构建覆盖，中间版 sha 已不可追。防法 = R2（sha 进日志头 + `-ExpectExeSha` + mtime 序）。

---

## §6 本席自测证据（判据能红能绿，不是"我看了一眼"）

| 自测 | 命令（在 `$acc` 下） | 实际输出（摘） | 结论 |
|---|---|---|---|
| 基线完整性 | `compare-ids.ps1 -VerifyBaseline` | `ids_count=45` · `verify_ids_sha256=ok` · `verify_first12_vs_handoff1.3=ok` · `VERDICT=GREEN`（exit 0） | 基线对上 §1.3 |
| **负控（绿）** | `compare-ids.ps1 -New 负控\pos_identical.ids.txt` | `identical=true` `mismatch_count=0` `VERDICT=GREEN`（exit 0） | 相同 ⇒ 绿 |
| **负控（红 · 错位 1 token）** | `compare-ids.ps1 -New 负控\neg_shift1.ids.txt` | `first_diff_pos=1` `old_id=10041` `new_id=1278` `mismatch_count=45` `VERDICT=RED`（exit 2） | 错位 ⇒ 红并指出位置 |
| **负控（红 · 单点替换）** | `compare-ids.ps1 -New 负控\neg_mid_sub.ids.txt` | `first_diff_pos=13` `old_id=19940` `new_id=19941` `mismatch_count=1` `VERDICT=RED`（exit 2） | 能定位到第 13 位 |
| 真实坏跑（红） | `compare-ids.ps1 -New C:\sp3\tile_pb1024.log` | `first_diff_pos=1` `mismatch_count=42` `VERDICT=RED`（exit 2） | 真 bug 日志能被抓 |
| 真实好跑（绿） | `compare-ids.ps1 -New C:\sp3\pair_C_pb6912.log` | `identical=true`、`new_ids_sha256=7d1e67f0…`、`VERDICT=GREEN` | tile=0 臂 ids 与基线逐位同（§1.3 结论复现） |
| 概率路径 | `compare-ids.ps1 -New 负控\synth_probs_new.log -Baseline 负控\synth_probs_baseline.txt` | `max_abs_delta_p=0.0001` `probs_flag=DRIFT_GT_1E-6` | 概率比对代码路径可用（合成件） |
| 读数抽取 | `extract-metrics.ps1 -Log C:\sp3\pair_D_pb6912.log` | `prefill_tok_s=652.6` `decode_tok_s=12.3` `resident=5890 of 20480` `dma_count=91076` `dma_vol_corrected_from_avg_GB=141.3` `ids_count=45` | 不靠人眼抄数 |
| 臂判据（负控绿） | `… -Arm NEG1 -RequireTile 1 -ExePath C:\sp3\sp_tile.exe -ExpectExeSha cc2a…` | 全绿，仅 `exe_sha_in_log=AMBER` ⇒ `VERDICT=AMBER` | 旧日志绑定只能 B 级（如实） |
| 臂判据（A 对基线自己 ⇒ 红） | `… -Arm A -Log pair_D` | `gate_prefill=RED`（==652.6）`gate_resident=RED`（仍 5890）`gate_dma=RED`（≥91076）`VERDICT=RED` | A 判据不会自证绿 |
| A 判据方向（合成件） | `synth_Aarm_good` ×3 | `gate_replication=GREEN` `gate_decode=GREEN`（中位 13.0）`VERDICT=GREEN`（exit 0） | 好读数能绿 |
| A 判据方向（合成件） | `synth_Aarm_decode_harm` | `gate_decode=RED`（10.8 < 11.32）`VERDICT=RED` | 伤 decode 能红 |
| 负控的负控 | `-Arm NEG1 -Log pair_C -RequireTile 1` | `gate_tile_witness=RED` `gate_prefill=RED` `gate_resident=RED` `VERDICT=RED` | 拿错臂必红 |
| sha 打假 | `-Arm REF-D -ExpectExeSha 000…0` | `gate_exe_sha=RED` `VERDICT=RED` | 错 sha 必红 |

---

## §7 未验 / 边界（本席明写）

1. **A 臂真实读数未到**：本表所有"绿/红"方向只用**合成件**与**①b 旧数据**自测过；真实 Phase 2 读数到达前，判据表本身**未经真实数据检验**（已验的只是"判据会红会绿"）。
2. **probs 门 = 未验**：没有旧 exe 的 `--top-probs` 基线；补齐需在旧 exe 上跑一版（占 GPU 窗口，队长批），并回填基线文件 `probs=` 行。**在补齐前，任何"概率逐位相同"的说法不能判绿。**
3. **decode 噪声未锁（n=1）**：D 配置（pb=6912+tile=1）只有 `pair_D` 一跑；噪声上界借自 p56（max-new=8）⇒ 45-token 口径的真实噪声**未测**，这也是 R6 要求 3 跑的原因。
4. **A 项期望窗口是 D 级外推**：+214/+404 槽、20.1 MB/槽、12.99 DMA/槽来自 `情报-A-字节账-20261017.md` 的字节账与 C/D 两点斜率；**不是实测收益**。
5. **零 GPU / 零编译 / 零进程**：本席未跑引擎、未改 `PolyStrata\src`、未碰 `E:\strata`/`E:\strata-v01431`/`E:\rlab`。
6. **验收模型厂商**：本会话模型 provider = deepseek（deepseek-flash）。施工 provider 若同为 deepseek，按 `E:\视频\_agents\角色卡\验收.md:12-16` 的厂商多样性硬约束，整次验收只能算"降级自检"，需改由 mimo-cn / Qoder Qwen3.8-Flash 复核——**这一条由队长裁定，本席不自行豁免**。

---

## §8 ★ R1a 轮口径变更与追加判据（2026-10-17 22:2x · lead 裁定，覆盖 §0/§3 中的绝对历史阈值；旧结论保留不删）

### 8.1 口径变更（★ 用历史 652.6 判绿/红一律作废）

- 旧基线 652.6 / 12.30 / 91,076 是**单点**（`pair_D` 一跑）；本轮同 config 四跑 **prefill 627.4 / 646.5 / 647.1 / 661.7**（−3.0%..+2.3%），证明跑间漂移真实存在。
- **新参照系 = 同日、同 exe、同 config 的多跑中位**；跨 exe / 跨日读数一律作废（exe sha 一变全作废）。
- 本席独立复算的本轮基线（A0 组，exe `cc2a9851…`，pb=6912、默认 every=2048）：
  - prefill 中位 **646.8**（4 跑：627.4/646.5/647.1/661.7）
  - DMA 中位 **89,646**（4 跑：95,102/91,692/87,600/86,226）；仅 3 跑（A0/A0b/A0p）= 91,692；仅非 probs 2 跑 = 93,397
  - 墙中位 **10.501 s**（4 跑推算 = prompt_tokens/prefill；3 跑 = 10.506）
  - decode 中位 **11.65**（仅非 probs 2 跑：11.38/11.92；probs 跑 decode 因逐 token 打印变慢，不可用）
- ★ 与 lead 口头基线对账：prefill 646.8 一致；**DMA 91,692 与墙 10.506 恰是"3 跑集(A0,A0b,A0p)"的中位，而 prefill 646.8 是"4 跑集"的中位 ⇒ 口径混用，已写进回执请 lead 定口径（本表按 4 跑/如实并列两套数）**。

### 8.2 追加判据 R14 · 旧 exe 必须**实跑**出旧值（lead 2026-10-17 22:2x）

- 负控"回旧值"不许用"我相信旧 exe 会回旧值"，也不许拿 `C:\sp3\sp_tile.exe` 当旧 exe（它可被下次副本覆盖）；**必须用干净备份副本实跑同一命令**：
  `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\_backups\R1-baseline-20261017\strata-poly.exe`
  sha256 `cc2a985128379b49b7c5682474d3fd54b3f92a14eea90518740eac7158b88f89`（本席 22:1x 复核：65,411,584 B，mtime 2026-10-17 12:54:01）。
- 判据：`extract-metrics.ps1 -ExePath <备份副本> -ExpectExeSha <sha>`；**日志头 sha == 声明 == 物理文件 sha** 三方一致才算 A 级绑定（R1a 日志已自带 sha256 字段，`gate_exe_sha_log` 已实现）。

### 8.3 追加判据 R15 · 读数是读数、推算是推算（★ 本轮新发现）

- **call 数（marks+1）不是日志读数**：R1a 全部 9 个日志里**没有** marks / call / chunk-call 行（已全量 grep）；`marks` 只在 `PolyStrata\src\poly\main.cpp:365-378` 内部构造，`n_calls` 在 `engine.cpp:3140-3148` 内部算，**均不打印**。
- call 数只能**推算**：`calls = floor((n_prompt−1)/every)+1`，every 取 `PolyStrata\src\poly\conversations.hpp:31`（默认 2048）或 `--prompt-cache-every` 显式值 ⇒ 6792 token：2048→4、1024→7、0→1。**任何把 call 数写成"日志读数"的表述 = 假绿形态 3**。
- 能当读数的只有：`pin chunks direct=…` / `direct DMAs=…` / prefill / decode / 驻留槽（墙是 `prompt_tokens/prefill` 的**推算**，日志不打印 prompt_ms）。

### 8.4 追加判据 R16 · probs 门必须配"同 config 对照"

- `max|Δp|` 单独一个数不能判"漂移来自机制"：必须同时算**同 config 两个 probs 跑互比**作对照。
- 本轮：A1p vs A0p（跨 call 对）= `1.405e-3 @pos1`（0.9897→0.991105）；A0p2 vs A0p（同 config 对照）= `3.4e-5 @pos23` ⇒ 跨 call 对是对照的 **41×**，可归因于 call 切分改变求和顺序（`engine.cpp:3115-3128` 注释），不是随机噪声；且在 lead 阈值 ≤2e-3 内。
- probs 跑的 decode **不可用**（逐 token stderr 打印拖慢：A0p 10.48 vs A0 11.38/11.92）；ids 仍可用。

### 8.5 追加判据 R17 · decode 只认 ≥3 非-probs 跑中位

- n<3 ⇒ **未验（不许绿）**；单跑绿/两跑绿都不收（lead 标准）。

### 8.6 本轮臂定义（从 run script 逐字核，一次一变量成立）

| 臂 | flags 差异（唯一变量） | run script |
|---|---|---|
| BASE / A0 | 默认 every=2048（无该参数） | `C:\sp3\run_R1a_A0_20261017.cmd:12`（与 `pair_D` 同 config） |
| A1 | `--prompt-cache-every 0` | `C:\sp3\run_R1a_A1_20261017.cmd:12`（仅多此参数） |
| A2 | `--prompt-cache-every 1024` | `C:\sp3\run_R1a_A2_20261017.cmd:12`（仅多此参数） |

- 全部 9 个 R1a 日志的 exe sha 均 = `cc2a985128379b49b7c5682474d3fd54b3f92a14eea90518740eac7158b88f89`（**与 BASE 同一二进制**；本轮"新机制"是旗标，不是新 exe——CP_STREAM 新 exe 未到，到时读数全作废重判）。

---

## §9 R1c 轮追加判据（2026-10-17 22:3x · A3 生产形态 + CP_STREAM 负结果；新 exe `6f71aca6…`）

### 9.1 A3 生产形态定义与读数（唯一读数来源 = `C:\sp3\R1c-NC{1,2,3}-20261017.log`）

- A3 = **零改码**生产形态：`--prompt-cache-every 8192`（同 exe `6f71aca6d76393b3…`，65,413,632 B，mtime 22:32:04；构建 `C:\sp3\build_cpstream2.log` `[56/56] Linking` + `EXIT=0`）。
- n_prompt=6792 < 8192 ⇒ 无周期 checkpoint ⇒ **1 call（推算，非读数）**。
- 3 跑读数：prefill **1359.3 / 1354.4 / 1330.5**（中位 **1354.4**）· DMA **28,792 ×3 逐位同** · 体积 **44.6 GB** · 驻留 **5,890 ×3** · ids **45/45 ×3（GREEN）** · decode **12.12 / 12.10 / 11.56**（中位 **12.10**，3 非-probs 跑）· 墙 **4.997 / 5.015 / 5.105**（中位 5.015，推算）· `WARNING` 0 · tile 自证在场。
- 参照（跨 sha，旁证）：prefill 中位 = old-exe A1 的 **+0.95%**、old-exe A0(4-call 组) 的 **+109.4%**；DMA = old A0 中位 89,646 的 **−67.9%**；decode 中位 ≥ old A0 11.65 / old A1 11.635（+3.9%/+4.0%）。确定性项（ids/DMA/驻留）与 old-exe 单-call 完全一致 ⇒ 机制可复现。

### 9.2 追加判据 R18 · 开关生效必须有引擎自证（防"开关没生效"假绿）

- 开臂日志必须含 A 级自证行：`strata-poly: STRATA_MOE_ROWS_CP_STREAM=1: a prompt batch's round copies run on a stream of their own`（`engine.cpp:1789-1791`）；关臂必须无此行。
- ★ **陷阱**：引擎在**变量被设置**时就会打印该行，`: off` 后缀表示"设了但没生效"（例如 `=0`）⇒ **只 grep 关键字不够，必须读后缀**。
- 本轮：R1c-CP **3/3** 为 `…run on a stream of their own`；R1c-NC **0/3**；全 `C:\sp3` 205 个日志仅这 3 份命中。旧 exe `bab819eb…` 的 R1b-CP 3/3 **无**自证行 ⇒ 旧开臂"生效"不可证（跨 sha 旁证，不混判；且该 exe 已不在盘上，只剩日志自报 sha）。

### 9.3 追加判据 R19 · "没加双缓冲"的反证口径

- 若 CP 引入双缓冲/多一份拷贝：DMA 次数或驻留必须变化 ⇒ 反证 = 所有单-call 日志的**确定性计数器完全一致**。
- 本轮 12 跑（R1c-NC1-3 + R1c-CP1-3 + R1b-NC1-3 + R1b-CP1-3）：驻留 **全 5,890**、DMA **全 28,792**（deviation=0）；另 old-exe 单-call 4 跑（R1a-A3/A3-2/A3-3/A3b）同为 5,890/28,792。任一份偏离 ⇒ 直接红。

### 9.4 追加判据 R20 · 负结果判"零效果"的口径

- 必须**同 sha 同 config** 的 开/关两组各 ≥3 跑；判零效果 = prefill/墙 两组**区间重叠（或相邻/相接）且中位差 <1%**；decode 中位差需在历史单跑噪声带内且区间重叠，否则只能写"未定/需复跑"。
- **禁止**拿跨 sha 的旧读数当本轮臂的参照（本轮 lead 给的 A3 decode 12.89/12.29/12.17 实为 old-exe `R1a-A3/A3-2/A3-3`，按 §8.1 作废）。
- 本轮结果（同 sha `6f71aca6`）：prefill NC 1330.5–1359.3（中位 1354.4）vs CP 1359.2–1363.0（中位 1360.7）⇒ 区间相邻（重叠 0.1 tok/s≈0.007%）、中位 **+0.47%**；墙 5.105–4.997 vs 4.997–4.983 ⇒ **边界相接**、中位 **−0.46%**；decode NC 12.10 vs CP 11.71（−3.2%，两区间重叠、CP1 10.89 为单跑低尾）⇒ 在噪声带内，不作伤害证据 ⇒ **判"零效果"成立**。

### 9.5 机制解释核对（本席只核代码在不在、语义是否如述）

- `engine.cpp:2407` `cudaStreamWaitEvent(cs(up_), ev_free_, 0)`（施工引用 :2403，现为 **:2407**，+4 行因 `engine.cpp` 22:30:58 有改动）· `:2409` `cudaEventRecord(ev_ahead_, cs(up_))` · `:2737` `cudaStreamWaitEvent(cs(st_), ev_ahead_, 0)`（与施工引用一致）· `:2785` `cudaEventRecord(ev_free_, cs(st_))` ⇒ **"`up_` 等 `ev_free_`、`st_` 等 `ev_ahead_`"在且语义如述**，四条配对未被 CP 改动破坏。
- CP 机制（`engine.cpp:2723-2751`）：只新增 `cp_` 流与 `ev_krn_/ev_copy_` 事件，注释自述 "The buffer is still one, so this overlaps a copy with the round before it and nothing more" ⇒ 与 DMA 不变、驻留不变一致。

---

## §10 P0-3/P0-5 显存纪律与命中率口径（2026-10-18 00:xx · task-10 + task-12；**零 GPU 建立**）

> 适用范围：R21–R25 只对**占用 GPU 的跑**生效（P1 槽配额、P2 显存纪律、此后所有 GPU 臂）。全部由既有采样 CSV / 日志 / profile / 源码复算得到；本席未跑、未编、未改码。

### 10.1 R21 桌面余量闸门（硬要求）

- **判据**：每跑 `32,760 − 峰值 VRAM ≥ 1,024 MiB`（即峰值 ≤ **31,736 MiB**）。**不满足 = RED**（明确要求，不是建议）。
- **峰值取法（固定方法，唯一）**：
  1) 外部 1 Hz 采样器，从引擎启动前（≥3 个 idle 采样）到退出后 ≥2 s：`nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits`，每行一个 MiB 落 CSV（与 `C:\sp3\R2a-vram-run{1..6}-20261017.csv` 同格式）。
  2) 峰值 = 窗口内 `memory.used` 的 **max**；总量必须 = **32,760 MiB**（否则 ERROR）；余量 = 总量 − 峰值。
  3) 覆盖率：采样数 ≥ 0.9×wall_s（1 Hz），首采样在启动前、末采样在退出后；不满足 ⇒ **未验（不得判绿）**。
  4) 临界带 `[1,024, 1,536)` MiB ⇒ **AMBER**（1 Hz 可能漏短峰），必须 ≥4 Hz 复采；复采仍 <1,024 ⇒ RED。
- **禁用替身**：引擎日志末行的 `… MiB of VRAM free at the end`（`main.cpp:680-682`）是**末态**不是峰值 ⇒ 不能代替采样（只作 sanity）。用末态 free 判 R21 = 假绿。
- 已实测（R6 JSON，`C:\sp3\R6-arms-summary-20261017.json`）：OFF 峰值 30,092–30,308 ⇒ 余量 2,452–2,668 = GREEN；**ring ON 峰值 31,691–31,814 ⇒ 余量 1,040/1,069/1,058/946** ⇒ **ONtop 31,814 = RED**，其余三跑仅高出阈 16–45 MiB（临界，按第 4 条必须复采）。R5 ON1 32,109（作废 block，旁证）⇒ 余量 651 = RED。
- 怎么变红：① 余量 <1,024 ⇒ RED；② 无采样/覆盖不足 ⇒ 未验；③ 用末态 free 或"开机 idle"代替峰值 ⇒ 判据无效/退回。

### 10.2 R22 槽配额生效自证（P1 `STRATA_EXPERT_PER_LAYER`）

- 背景（A 级读码）：现行 `fill_expert_cache()` 是**全层共享一个计数器**按 `(prior+use)/slot_bytes` 全局降序取到 arena 满（`engine.cpp:1095-1145`；`常驻名单流水线-判据-20261016.md:18` 明写"名单不是每层 top-N"）。P1 改为 per-layer 配额（`计划-P0派单-20261017.md` L23）。
- **必要证据（缺 = 未证，不许用性能代替）**：
  - N1 旗标自证行（A 级）：`strata-poly: STRATA_EXPERT_PER_LAYER=…`，按 R18 **读后缀**（`: off` 也打印）。
  - **路径 A（首选，A 级）**：引擎启动打印每层槽数（40 个值，一行 `slots_per_layer=…sum=R` 或 40 行 `layer=L slots=Q`），且 **sum == 同一日志 `loaded … R of 20480` 的 R**。RED：打印缺席且无合格路径 B / 和≠R / 与声明 Q_l 不符。
  - **路径 B（次选，B 级，仅当预测效应 > 噪声）**：`profile 重算 + experts from VRAM 对照`（per-token 口径，同口径比值消绝对偏移）：`ρ = Σ_{新集} c_e / Σ_{老集} c_e`，`H_pred = H_off(中位)×ρ`；`|H_on(中位) − H_pred| ≤ 2.0 pt` 绿（H 场 12 跑离散 52.1–53.1 = 1.0 pt）。
  - **无鉴别力规则（本席用 profile 复算，D）：等配额 147/层 vs 全局 top-5890 ⇒ A 口径预测 −0.66% rel / 去重口径 −0.17% rel，都 <1 pt 噪声 ⇒ 路径 B 对这类方案无效，必须路径 A**；只有声明方案预测变化 >2 pt 才允许路径 B。
- **"只有性能变快不算生效"（写死）**：prefill/DMA 变快一律不是配额生效证据；无 N1+（A 或合格 B）⇒ R22 = 未验/退回。
- 改前偏斜（A 级，本席从 profile 复算）：全局 top-5890 分到 40 层 = **97–189 槽/层**（max/min 1.95×，20/40 层 <147）。

### 10.3 R23 命中率三口径（**唯一可用 = C 去重字节**；A/B 禁用范围见下）

| 口径 | 算式（分子/分母） | 单位 | 作用域 | 数据来源字段 | 本机值 | 复算状态 |
|---|---|---|---|---|---|---|
| **A** profile 计数份额 | `Σ_{e∈R} c_e / Σ_e c_e`（×帧字节则等价） | profile 路由事件（×3,096,576 B） | **冻结 profile、全历史**（20,275 非零专家，Σc=53,957,200） | `p_run.txt` 计数 + 常驻集 R | **68.2985%** = 36,851,956/53,957,200 = 114.115/167.083 TB | **A 级复现**（本席逐行重算，一致） |
| **B** 日志每 token | `gpu_experts/(gpu_experts+cpu_experts)`（`main.cpp:715-724`） | 专家-事件（token×层×专家） | **本次运行的 decode 段**（计数只在 eval `engine.cpp:2122-2123`；prompt 批路径不计数） | 日志 `experts from VRAM X%`（绝对计数器不打印） | 5,890 槽时代 **52.0–53.4%**；7,388 槽时代 **77.7%**（`pf_off_*`/`dlc_pb32` 实测） | **字段 A 级；换算不能独立复算**（缺 `gpu_experts`/`cpu_experts` 绝对值与逐 eval 路由） |
| **C** 去重字节（**唯一准用**） | `H = Σ_{(l,e)∈D∩R} b / Σ_{(l,e)∈D} b`；等大帧= `|D∩R|/|D|` | **去重专家**（×3,096,576 B） | **本次运行 prompt 的去重路由集 D** | D 需引擎去重计数/dump（**当前日志未打印 ⇒ 缺字段**）；R 取 `loaded … R of 20480` | **38.5%** = 7,388/19,198（7,388 槽/700 token 时代） | **D 级模型复现 38.42%**（T=700, p=22·c/Σc, 19,230 触达）；**单 call 时可从 DMA 反推**（见 R25）：A3 跑 DMA 28,792 ⇒ 14,396 份 ⇒ ≈28.2–29.7%，模型 29.38% |

- **唯一该用**：**C（去重字节口径）** —— 一切"命中率/字节账/DMA 效率"门；单 call 下的可操作形式 = **计数器（R25）**：`streamed 去重份数 = DMA/2`，`bytes = 份数×3,096,576`。
- **禁用**：① **B 不得用于字节/streamed/DMA 预测，也不得与 C 混算**（B 只允许解读 decode 的单 token 搬运占比/延迟）；② **A 不得当"本次运行命中率"**（它是 profile 历史事件的加权份额，可高估约 2×：同 5,890 槽 A=68.3% vs C 模型=30.6%）；③ A/B 都不得写"命中率↑"门。
- 混算反面数（本席算）：7,388 槽/700 token 时，C 给 streamed 11,810 专家 = **36.6 GB**；若用 B=77.7% 算 = 4,281 专家 = **13.3 GB（少算 23.3 GB）**；用 A=75.74% 算 = **14.4 GB**。⇒ 混用会让"DMA 体积"差 2.5×。
- 怎么变红：字节账用 B/A ⇒ RED；给的命中率无法用 C 算式（D∩R 或 DMA/2）复算 ⇒ RED；同一跑两口径被报成同一个数 ⇒ RED。
- 完整推导/三口径冲突场景/复算脚本见：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\命中率-三口径与DMA投影对账-20261017.md`。

### 10.4 R24 reserve 对账（AMBER 条款）

- 输入：`--vram-reserve-mib` 请求值；实测空闲 at peak = 32,760 −（R21 同一次采样的峰值）；可选用引擎末态自印 `X MiB of VRAM free at the end`（`main.cpp:680-682`）。
- 算式：`dev = (实测空闲_peak − reserve_asked)/reserve_asked`；`|dev| ≤ 25%` GREEN；**>25% ⇒ AMBER 并点名"reserve 未守住"**（实测<asked=被后分配吃掉；实测>asked=reserve 未兑现/过保守）。末态 free < asked ⇒ 也 AMBER（B 级）。
- 机制（A 级读码）：reserve 只在 arena 定尺寸时减一次（`engine.cpp:1108`）；fill **之后**的分配（ring 第二缓冲 **1,512 MiB**、CUDA graph、blas 工作区）**不再询问 reserve** ⇒ 峰值空闲可远低于 asked。
- 已实测两方向：OFF A3（asked 2,500）实测 2,637–2,756 ⇒ dev +5.5..+10.2% = GREEN（本席从 CSV 复算）；ring ON v2（asked 2,500）实测 946–1,069 / 651 ⇒ dev −57%..−74% = **AMBER"reserve 未守住"**。
- 怎么变红：|dev|>25% ⇒ AMBER；与 R21 同报（R21 判硬门，R24 解释原因）。

### 10.5 R25 涨驻留的 prefill 收益：**只许计数器口径（lead 2026-10-18 裁定，本席已复核）**

- **判据（甲，唯一准许）**：单 call 下 `DMA 次数 = (20,480 − N_zero − R) × 2`；`streamed 份数 = DMA/2`；`体积 = 份数 × 3,096,576 B`。其中 `N_zero = 从未被路由的专家数 = 205`（`p_run.txt`：20,275 非零 / 205 零）。
- **复核"每个 (层,专家) 单 call 只搬一次"（本席 66 个单 call 日志全量核）**：`residual = (DMA/2 − (20,480−205−R)) / (20,480−205−R)` ⇒ **最大 |residual| = 0.17%**（R=5,890:+0.08% · 5,721:+0.08% · 5,551:+0.03% · 5,213:−0.02% · 5,377:+0.11% · 5,653:+0.12% · 5,889:+0.17%）；残差恒为正且 ≤0.17%，与 decode 期换入少量副本一致 ⇒ **无额外重复，甲成立（A 级）**。
- ★ **对 lead 投影的算术更正**：驻留 6,900 ⇒ `20,480−205−6,900 = 13,375 份` ⇒ **DMA 26,750 次 / 体积 41.42 GB（较 44.58 GB = −7.1%）**；lead 公式 `(20,480−6,900)×2=27,160 / 42.05 GB（−5.7%）` 漏扣 205 零路由专家 ⇒ **偏 0.63 GB**。
  - ★ **prefill 增益目前不能写死**：R8 实测边际（DMA +2.3% ⇒ prefill −2.15%；DMA +4.6% ⇒ −4.1%）约 **0.9 prefill% / DMA%** ⇒ 按 −7.1% 推 **+6~7%**，与 lead 的 +3~4% 不一致（R8 为单跑且含窗口漂移风险，未定）⇒ **prefill 增益 = 未定，不许进承诺**；只许写"体积 −7.1%（A 级计数器口径）"。
- **口径乙（禁用）**：用频率加权命中率（A：68.30%→73.47%）乘基线体积 = (1−0.7347)×~141 = **37.31 GB（−16%）** ⇒ 禁用理由：① 141 GB 是 4-call 重复搬运的旧体积，单 call 基线是 44.58 GB；② 去重已吃掉频率收益，频率加权命中率再乘一遍 = 双重计数。
- 怎么变红：单 call 日志 residual >0.5% ⇒ RED（重复搬运被引入）；用 A/B 命中率预测 prefill 体积收益 ⇒ RED（禁用乙）；只给"性能变快"不给计数器 ⇒ 未验。

