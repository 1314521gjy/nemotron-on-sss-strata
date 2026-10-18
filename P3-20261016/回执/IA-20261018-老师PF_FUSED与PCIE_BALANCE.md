# 回执 · 情报-AE：老师 PF_FUSED 与 PCIE_BALANCE 掏清楚（2026-10-18）

## 1. 任务
用户"为何不模仿老师？"lead 复盘发现两项停在"算过不划算/没测"：**A `PF_FUSED`**（backlog 仅写"上游实测仅 +5%，未接，暂缓"，从没查过它是什么/能不能接）、**B `PCIE_BALANCE`/`STRATA_PCIE_FRAC_DEFAULT`**（只测过固定 `--pcie-frac`，自动份额从未实现/测试）。要求：A 是什么/融什么/在哪条路径/+5% 出处/我们有没有/抄过来改哪多少行/前提；B 读 `04be241a`、`aa7b0cb1` 与 0.1.44.1 实现，搞清怎么定 frac/更新频率，抄哪多少行；C 给预期收益与等级，明确"值不值得抄、先抄哪个"。

## 2. 状态
**达成**。零 GPU、零编译、不改码、未 spawn 子代理；只写本回执与详细件。老师侧读 `E:\strata`（0.1.20，含 `STRATA_PF_FUSED` 消费点）、`E:\strata-v01431`（0.1.43.1）、`upstream-0.1.44.1`（0.1.44.1）与 bare clone `C:\sp3\upstream-hist\polystrata.git`（两个 commit）；我们侧读当前 `PolyStrata\src\poly` 与 `回执\R39-pcie-direct-20261018.md`。

## 3. 结论（一句话）
**先抄的不是这两个——是零改码的"固定 `--pcie-frac 0.2~0.35` + `--pcie-mode direct` 重测"**（R39 已测 **+5.6% 绝对值**，A 级，但今天配置已变需重测；当前交付**根本没开 pcie**）。
- **PF_FUSED = 预填（prompt）路径的 fused int8 专家**（`STRATA_PF_FUSED=1`；不是 decode）。它把整条 prompt 专家链（native 激活量化→GGUF 权重解码→int8 `mma.sync`→SwiGLU→down 到 per-slot）融进自有内核（`moe_fused_iq.cu` 1,189 行 + `moe_fused.cu`）。上游实测 prefill **+4~12%**（C），**+5% 是 5090 32K 那一格**。**现在不能抄**：K 硬编码 2560/640（我们 1024/2688）、CUDA 侧不支持我们 up 的 Q4_K（KQ 只在 HIP）、poly 没接；抄=内核 300–600 行 + 接线 150–300 行 + class-E，**只换端到端 +0.6~1.3%**。
- **PCIE_BALANCE 实为两件**：① **PcieAuto 全局份额**（`aa7b0cb1`，0.1.42+）= 探针 L + 池速度 R ⇒ `f = L/(L+1.85R)`，3 次更新、0.05 步、夹 [0.10,0.60]；老师 3060 0.55→0.20 ⇒ **decode ×1.300**，A4000 ×0.999。我们按公式应得 **0.30**（L≈26 GB/s、R≈32.2 GB/s），与 R39 实测最优 0.2–0.35 重合 ⇒ **相对手调只 +0~2%（D）**，~80–120 行。② **`STRATA_PCIE_BALANCE=1` 每层动态**（`04be241a`，仅 v0.1.45-poly.1）= 每层用带遗忘 LSQ 拟合 CPU/GPU 两侧成本、选使长边最小的 m；老师 16-worker 实测 **12.19→12.17（≈0%，池 DRAM-bound）** ⇒ 我们同型 ⇒ **不抄**。

## 4. 证据（关键读数 + 出处 + 级别）

| # | 结论 | 证据 | 级 |
|---|---|---|---|
| E1 | **PF_FUSED 定义/路径**：`STRATA_PF_FUSED=1` 开、`=0` 回 MMQ；**只 prefill**；`experts_native()` 一次启动一层一批流式专家；文件头注释逐条写算术（激活量化/权重解码/int8 mma/cp.async/ldmatrix/SwiGLU/down） | `upstream-0.1.44.1\src\prefill\moe_fused_iq.cu:1-32,1160-1189`；`src\prefill\moe_fused.cu:654-660`；`bench\results\2026-10-03-dev-5090\README.md:43`（decode 在噪声内） | C |
| E2 | **PF_FUSED 实测**：5090 IQ3_S prefill **+11%/+5%/+4%**（4K/32K/128K）；5070Ti **+5.5%@104K、+7.8%@24K**；4080S **+8.9%** cold；IQ2_XS **+12%@4K、+3%@32K**；R9700 gfx12 默认开 **4K +15.7%、16K +13..17%**；质量"rounds differently from MMQ"⇒class-E | `bench\results\2026-10-03-dev-5090\README.md:43`；`2026-10-04-community-rtx-5070ti-windows\README.md:102`；`2026-10-04-community-rtx-4080s-iq3s\README.md:233`；`docs\DETAILS.md:26-27,2028`；`docs\AMD_HIP.md:360` | C |
| E3 | **PF_FUSED 前提/不匹配**：`gu_covered`=IQ2/3/4_XS 系列（KQ 的 Q4_K/Q5_K 仅 `__HIPCC__`）、`d_covered`=Q2_0/IQ4_NL（KQ 加 Q5_1/Q8_0）；CUDA `native_supported` 里 `if (env<0) return false`；需流式 chunk ≥ `STRATA_PREFILL_STREAM_MIN`（1024）；**K 常量 `GU_ROWS_K=2560, D_ROWS_K=640`** | `moe_fused_iq.cu:67,1071-1090,1118-1146`；`docs\STRIX_HALO.md:119` | C |
| E4 | **我们侧**：`moe_fused_iq.cu` 源码在树里并编进 `strata_mmq`（`CMakeLists.txt:1427-1429`），但 **`src/poly` grep `moe_fused`=0**；poly prefill 走 MMQ（`engine.cpp:13` include mmq；`:2660-2673` `L.mmq=…Mmq::fits`；`:2545/:2425`）；我们无 gate、up 一个矩阵 + relu2（`:1129`） | `PolyStrata\CMakeLists.txt:1427-1429`；`engine.cpp:13,1129,2545,2660-2673` | A |
| E5 | **PcieAuto 算法**：启动 `probe_pcie_h2d_gbps()`（256 MiB×4 burst 取 max）；初始份额 `base×min(1,gbps/20)`（base native 0.55/NVFP4 0.25/canonical 0.2）；窗口内 `f = L/(L+1.85R)`（R=专家字节/池 wall per miss），0.05 步、夹 [0.10,0.60]、最多 3 次（48/96/96 窗）、Δ<0.025 不动；显式 `--pcie-frac` 优先，`STRATA_PCIE_FRAC_DEFAULT=old` 退回 | `upstream-0.1.44.1\src\program\generate.cpp:593-594,1664-1723,1736-1780,2850-2895,8680-8712`；commit `aa7b0cb13eab1d7f1e97ece8907591b545a70a2e`（2026-10-10） | C |
| E6 | **PcieAuto 实测（老师）**：RTX3060 0.55→0.20 **decode 1.300（5/5 对）**；P100 1.011；A4000 0.999；质量：显式份额逐字节同 0.1.41，新默认回答略异（KL 0.0015/0.0011）⇒class-E；Windows canonical 需 `STRATA_PCIE_AUTO_CANON=1`（5070：code +2.5%、短答案 −3.3%） | `docs\DETAILS.md:1708-1716` | C |
| E7 | **`STRATA_PCIE_BALANCE=1` 每层动态**（commit `04be241a`，2026-10-14，port PR #1548 `a513394c4`）：`wait_flag_ge_stamped` 写 GPU clock；host 每 step 计时；每窗口后带遗忘 LSQ 拟合 CPU `a+c·n_cpu+d·n_pcie`、GPU `g0+g·n_pcie`；每层选使长边最小的 m；每 64 层强制换一个；**off = 与 v0.1.39 逐位同**。实测：单 AVX2 worker 16.77/17.17→**12.98/12.82**；**16 workers 12.19→12.17（≈0%）**；只进 `v0.1.45-poly.1`（0.1.44.1 里 grep `PCIE_BALANCE`=0） | commit `04be241a5a7ec016716cb50aab99be82a801e4fc`（body 原文）；`git tag --contains` | C |
| E8 | **我们侧（空壳 + 已测）**：`--pcie-auto` 是空壳（`engine.cpp:2262-2264` 打印 "the share fitting is not in this build; only --pcie-frac is …"）；`--pcie-mode dma/direct/auto` 已实现（`:2594-2651` direct 设备别名；`:2751-2764` 建槽；`:3157-3189` 选 miss）；**交付脚本没开 pcie**（`run_delivery_t16.cmd`/`run_delivery_borrow.cmd` 无 pcie 参数）；R39 实测：E0 frac0 decode 12.26、**E2 0.2 direct 13.06（重跑 12.84，中位 +5.6%）**、E1 0.2 dma 11.17（direct vs dma +16.9%）、E3 0.35 direct 12.26（0%） | `engine.cpp:2262-2264,2594-2651,2751-2764,3157-3189`；`回执\R39-pcie-direct-20261018.md` | A |

**能变红的判据**：① `gu_covered` 在 CUDA 侧含 Q4_K ⇒ E3 的"接不了"错；② `GU_ROWS_K` ≠2560 ⇒ K 不匹配结论错；③ PcieAuto 公式 ≠ `L/(L+1.85R)` ⇒ E5 错；④ R39 E2 中位不是 +5.6% ⇒ E8 预期错；⑤ `04be241a` 的 16-worker 不是 12.19→12.17 ⇒ E7 错。

## 5. 两个的"若实现，预期变化量 + 等级"
- **固定份额 + direct（零改码）**：**+3~6% per-token**（R39 旧测 +5.6% 绝对值，**A**；需 1–2 跑重测因配置已变）；class-E（E2 ids 43/64、E3 1/64）。
- **PcieAuto 全局（~80–120 行）**：对我们机器 **相对手调 +0~2%（D）**；换慢 CPU 机器 **×1.30**（C，老师 3060）；class-E；需 1 跑。
- **`STRATA_PCIE_BALANCE` 每层（~150–250 行）**：**≈0%（D）**（我们池 32.2 GB/s DRAM-bound；老师 16-worker 12.19→12.17，C）⇒ 不抄。
- **PF_FUSED（内核 300–600 行 + 接线 150–300 行）**：prefill **+4~9%（C）**、decode 0% ⇒ 每答案 **+1.6~3.8%**、端到端（含 ~18 s 加载）**+0.6~1.3%（D）**；class-E ⇒ 现在不抄。

## 6. 未验 / 边界
- **U1**：R39 的 +5.6% 是旧配置（alt prompt、threads16、`--expert-swaps 0`、旧 exe）⇒ **必须重测**（今天 `sp_borrow.exe` + `STRATA_PROMPT_BORROW=1`）。
- **U2**：PcieAuto 的 R 用 R49 的 96 µs/miss 推 32.2 GB/s；交付形态若 miss 结构变（borrow/表），R 会变，f 在 0.25–0.35 间（D）。
- **U3**：PF_FUSED 收益全来自老师/社区机器（C）；我们模型（Q4_K_M + relu2、K=1024/2688）没有同类内核，只剩"prefill 时间 × 外部百分比"（D）；我们 prefill 3.5 s 里 MMQ 占多少**没有 post-1938 的 nsys**（未测）。
- **U4**：`04be241a` 只在 `v0.1.45-poly.1`；0.1.44.1 树无 `PCIE_BALANCE`（A）；PcieAuto 0.1.42+ 才有（老师跑的 0.1.40.2 没有）。
- **U5**：class-E：任何份额变化都会改 CPU/GPU 分工 ⇒ ids 漂（R39 已测），只能 opt-in/性能模式。
- **U6**：我们 `--pcie-mode auto` 在 direct 可用时 = direct（R39 §5 边界），不是份额自适应；不要与 PcieAuto 混名。

## 7. 落盘（绝对路径）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AE-老师PF_FUSED与PCIE_BALANCE-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-老师PF_FUSED与PCIE_BALANCE.md`
- 关键旁证：`C:\sp3\upstream-hist\polystrata.git`（`aa7b0cb1`、`04be241a`）· `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\prefill\moe_fused_iq.cu` · `…\回执\R39-pcie-direct-20261018.md`
