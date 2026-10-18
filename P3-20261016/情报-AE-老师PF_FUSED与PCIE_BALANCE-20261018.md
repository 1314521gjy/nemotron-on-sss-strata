# 情报-AE · 老师两个"没测就暂缓"的组件：PF_FUSED 与 PCIE_BALANCE（零 GPU，读码 + 已有实测）— 2026-10-18

- **席位**：scout-a · **触发**：用户"为何不模仿老师？" lead 复盘发现两项停在"算过不划算/没测"：**A `PF_FUSED`**（backlog 只写"上游实测仅 +5%，未接，暂缓"）与 **B `PCIE_BALANCE`**（只测过**固定** `--pcie-frac`，**自动份额从未实现/测试**）。本件要回答：它们各是什么、算法、我们能不能接、**值不值得抄、先抄哪个**。
- **状态**：**达成**。零 GPU、零编译、不改码、未 spawn 子代理；老师侧取本机 `E:\strata`（0.1.20 树，含 `STRATA_PF_FUSED` 消费点）、`E:\strata-v01431`（0.1.43.1）、`upstream-0.1.44.1`（0.1.44.1）与 bare clone（`C:\sp3\upstream-hist\polystrata.git`）的 `04be241a`/`aa7b0cb1`；我们侧取当前 `PolyStrata\src\poly` 与 `C:\sp3\R39-*.log`/`回执\R39-pcie-direct-20261018.md`。
- **口径**：每条给 `file:line`/commit/sha；"读到的"与"推算"分开；查不到的写"已查 X/Y/Z 无命中 + 计数"。

---

## §0 一句话回答（先抄哪个）

**先抄的不是这两个，而是"固定 `--pcie-frac 0.2~0.35` + `--pcie-mode direct` 的重测"**（零改码，1–2 跑）——R39 已实测 **+5.6% 绝对值**（A 级，但今天配置已变，需重测）；这两个组件里：
- **PF_FUSED：现在不抄。** 它是**预填（prompt）路径**的 fused int8 专家内核，**不碰 decode**；上游实测 prefill **+4~12%**（C 级），折到"每答案"只有 **+1.6~3.8%**（不含加载）。而且它**硬编码 K=2560/640（Qwen3.8-Flash 几何）**，我们是 **1024/2688**，且 CUDA 侧不支持我们的 **Q4_K up**（KQ 只在 HIP 编进来）⇒ 不是"接个开关"，是**重写内核级**（~300–600 行内核 + 150–300 行接线），**不划算**。
- **PCIE_BALANCE：分两半。** ① **PcieAuto 全局份额自动**（`aa7b0cb1`，0.1.42+）**有条件可抄**（~80–120 行）：公式 `L/(L+1.85R)`；但在**我们机器**按公式应得 **0.30**（L≈26 GB/s、R≈32.2 GB/s），与我们实测最优点 0.2–0.35 重合 ⇒ **相对手调只有 +0~2%（D）**；它的价值在**换机器自适应**（老师 RTX 3060 慢 CPU：0.55→0.20 ⇒ decode **×1.300**，C 级）。② **`STRATA_PCIE_BALANCE=1` 每层动态**（`04be241a`，只进 `v0.1.45-poly.1`）**不抄**：我们 CPU 池是 **DRAM-bound（32.2 GB/s）**，老师自己的 16-worker 实测就是 **12.19→12.17 ms/round（≈0%）**（C 级）。

---

## §1 A · `PF_FUSED` 究竟是什么

### §1.1 定义与机制（老师侧，`file:line`）
| 项 | 内容 | 出处 |
|---|---|---|
| 名字 | **`STRATA_PF_FUSED`**（PF = prompt/prefill，非 decode）；`=1` 开，`=0` 回退到 MMQ prompt 内核；`STRATA_PF_FUSED_NATIVE=0` 是它的 A/B 臂 | `upstream-0.1.44.1\src\prefill\moe_fused.cu:654-660`、`src\prefill\moe_fused_iq.cu:1119-1121` |
| 融的是什么 | **一条 prompt 专家链全融**：per-64 值的 native 激活量化 → 权重的 GGUF 块**按需解码成 int8**（codebook + scales，双缓冲 shared）→ `mma.sync m16n8k32`（scale-per-16 的格式用 m16n8k16）→ SwiGLU 后 H 再量化成 int8 → down 直接写 per-slot 行。流水线是 cp.async 四段 + ldmatrix fragments | `moe_fused_iq.cu:1-32`（文件头注释逐条写了算术/load stage/epilogue） |
| 走哪条路径 | **只 prefill**：`experts_native(Batch...)` 一次启动一层的一批流式专家；**decode 不受影响**（老师 bench 明写 decode 在噪声内，不做声明） | `moe_fused_iq.cu:1160-1189`（`experts_native`）；`bench\results\2026-10-03-dev-5090\README.md:43` |
| 前提条件 | ① **native pack 的格式组合**：`gu_covered` = IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS（KQ 只在 `__HIPCC__` 下加 Q4_K/Q5_K）；`d_covered` = Q2_0/IQ4_NL（KQ 加 Q5_1/Q8_0）② CUDA 上**默认关**：`native_supported()` 里 `if (env<0) return false`（非 HIP 的默认路径）③ prompt 要按**流式 chunk ≥ `STRATA_PREFILL_STREAM_MIN`（默认 1,024 token）**才进它 ④ 需要一层一批的 `Batch/NativeGeom` 启动 | `moe_fused_iq.cu:1071-1090`（covered）、`:1118-1146`（`native_supported`）、`:67`（`GU_ROWS_K=2560, D_ROWS_K=640`）；`docs\STRIX_HALO.md:119`（1,024 token 门槛） |
| 为什么接不了我们 | **K 硬编码**：`GU_ROWS_K = 2560`（gate/up 的 K）、`D_ROWS_K = 640`（down 的 K）是 Qwen3.8-Flash 的几何；我们 Nemotron：**up K = 1024（latent）、down K = 2688**。另：我们 up 是 **Q4_K**，CUDA 的 `gu_covered` **不含 Q4_K**（只有 HIP KQ 路径有），我们的 down `IQ4_NL` 在 covered 内——**组合也不成立** | `moe_fused_iq.cu:67,1083-1090`；我们几何见 `PolyStrata\src\poly\engine.cpp:1129`（"no gate and no fused gate_up: one up per expert, activated by relu2"）、专家帧 3,096,576 B |

### §1.2 +5% 的出处（老师/社区实测，C 级）
- **RTX 5090 + IQ3_S**：prefill **+11% / +5% / +4%** at 4K/32K/128K（相对 0.1.38 默认）——`upstream-0.1.44.1\bench\results\2026-10-03-dev-5090\README.md:43`。**backlog 的"仅 +5%"就是 32K 这一格**（还有 5070Ti 的 +5.5% 104K）。
- **RTX 5070Ti Windows**：prefill **+5.5% @104K（0.1.36）/ +7.8% @24K（0.1.38）**——`bench\results\2026-10-04-community-rtx-5070ti-windows\README.md:102`。
- **RTX 4080S + IQ3_S**：**+8.9%** cold prefill（2,195.2 vs 2,014.9）——`bench\results\2026-10-04-community-rtx-4080s-iq3s\README.md:233`。
- **IQ2_XS**：+12% at 4K、+3% at 32K——`docs\DETAILS.md:26-27`；**R9700 gfx12（IQ3_S/XXS 默认开）**：4K +15.7%、16K +13..+17%——`docs\DETAILS.md:2028`。
- 质量：老师明说 **"rounds differently from MMQ"**（`docs\AMD_HIP.md:360`），KQ 变体"output differs from the FP16 expert path's at rounding level; KL-gated"（`moe_fused_iq.cu:1067-1069`）⇒ **class-E（非位精确）**。

### §1.3 我们 poly 线有没有对应物
- **源码在树里但没接线**：`PolyStrata\src\prefill\moe_fused_iq.cu`（1189 行）+ `moe_fused.cu` + `include/strata/prefill/moe_fused_iq.hpp` **存在**，CMake 把 `src/prefill/moe_fused.cu`、`moe_fused_iq.cu` 编进 `strata_mmq` 静态库（`PolyStrata\CMakeLists.txt:1427-1429`）。
- **我们的 prompt 路径不用它**：`src\poly` 全目录 grep `moe_fused` = **0 命中**；我们的 prefill 走 **MMQ**：`engine.cpp:13`（`#include "mmq.hpp"`）、`:2660-2673`（`L.mmq = … Mmq::fits(...)`）、`:2545`（`if (mmq_)` 批路径）、`:2425`；即"按 rows 走 `moe_rows()` 的 MMQ 批路径"。
- **结论**：不是"没有"，是"**有内核源码、没接进 poly 的预填**"，而且直接接也接不上（§1.1 K/格式）。

### §1.4 若抄：改哪里、多少行、预期
| 子项 | 改动 | 行数（估） |
|---|---|---|
| ① 内核适配我们的 K/几何 | 重写/重新模板化 `moe_fused_iq.cu` 的 K 常量、tile 选择（`pick_ww`）、act 量化块（native 的 64 值/80B 块是按 K 的） | **300–600 行**（含新测试） |
| ② CUDA 侧支持 Q4_K up + IQ4_NL down | `gu_covered/d_covered` 只在 HIP 开 Q4_K；CUDA 的 `setup_one<T,…>` 表也没有 Q4_K 的 native kernel 实例 | **50–150 行 + 编实例** |
| ③ 接进 poly 预填 | 把我们的 `moe_rows()` 的 staging 环（`stage_mib 2048`）映射成 `Batch/NativeGeom`，替换 MMQ 分支，落 per-slot 输出 | **150–300 行** |
| ④ 质量门 | prefill 舍入变化 ⇒ `--top-probs`/ids 门 + class-E 记录 | 1 跑 |
- **预期收益（我们 regime）**：prefill **+4~9%**（C；取 5090/5070Ti/4080S 区间）⇒ 我们 prefill 6792 token / 1938.8 tok/s = **3.50 s → 3.21–3.36 s**（−0.14~0.29 s）；**decode 0%**；每答案（prefill+decode ≈ 3.5+5.0 s）**+1.6~3.8%**；算上 ~18 s 加载的端到端 **+0.6~1.3%**。**等级：C（外部实测折我们时长）/ D（折算）**。
- **判断**：**现在不抄**。理由：只 prefill、几何不匹配、CUDA K-quant 不覆盖、工作量最大（400–900 行）、class-E，换端到端 ≤1.3%。

---

## §2 B · `PCIE_BALANCE` / `STRATA_PCIE_FRAC_DEFAULT` 的自动平衡算法

### §2.1 机制一：`PcieAuto` 全局份额自动（commit `aa7b0cb1`，2026-10-10，进 0.1.42+）
- **commit**：`aa7b0cb13eab1d7f1e97ece8907591b545a70a2e`（`C:\sp3\upstream-hist\polystrata.git`，作者 Niko1221，2026-10-10）："CUDA: the default `--pcie-frac` is refined in the first decode windows from the CPU pool's measured speed (single GPU, no --pcie-frac); `STRATA_PCIE_FRAC_DEFAULT=old` keeps the start-up rule"（`git show -s aa7b0cb1`）。
- **它怎么定 frac**（`upstream-0.1.44.1\src\program\generate.cpp`）：
  1. **启动探针** `probe_pcie_h2d_gbps()`（`:1664-1719`）：256 MiB ×4 次 H2D burst，取最大 GB/s；初始份额 = `base × min(1, gbps/20)`（`pcie_frac_for_gbps` `:1720-1723`；base = native 0.55 / NVFP4 0.25 / canonical 0.2）。
  2. **解码窗口内细化**（`struct PcieAuto` `:1736-1780`）：用引擎**已有的计时**（CPU 池的 wall time / miss 数）算 **`R` = 每 miss 专家字节数 ÷ 每个 miss 的池 wall（GB/s）**、`L` = 探针链路 GB/s；**新份额 `f = L / (L + 1.85·R)`**，取整到 0.05，夹 [0.10, 0.60]；**最多改 3 次**（第 48、96、144 个窗口），每次必须跨一整步（|Δ|≥0.025）才动；miss 数 <300 就再等；窗口太多了（2000）还测不到就关掉。**显式 `--pcie-frac` 永远优先**；`STRATA_PCIE_FRAC_DEFAULT=old` 退回 0.1.41 规则。
  3. 调用点：启动 `:2850-2895`（`o.pcie_frac_auto = …`）、窗口后 `auto pcie_auto_step = …` `:8680-8712`；选项字段 `:593-594`。
- **老师实测（C 级，`docs\DETAILS.md:1708-1716`）**：交错整跑 A/B，新默认 vs old（median，route tail skip 开）：
  | 卡 | 池速度 | 份额 old→new | decode 新/旧 | 更快对数 |
  |---|---|---|---|---|
  | RTX 3060 12G IQ3_XXS | 32–37 µs/expert | 0.55→0.20 | **1.300** | 5/5 |
  | Tesla P100 IQ3_XXS | 84–97 µs | 0.29→0.20–0.25 | 1.011 | 4/5 |
  | RTX A4000 Q2_0 | 97–115 µs | 0.30→0.30 | 0.999 | — |
  - 质量：显式份额与 0.1.41 **逐字节同**；新默认 vs 旧默认回答略有差异（`KL 0.0015`/`0.0011`，与 run-to-run floor 同量级）⇒ **class-E**。
  - Windows canonical 包：opt-in `STRATA_PCIE_AUTO_CANON=1`（RTX 5070：code +2.5%，短答案 −3.3%）；HIP 另有 `STRATA_PCIE_AUTO_SPLIT`。
- **我们能不能用/值不值**：
  - 我们 `--pcie-auto` **是空壳**：`PolyStrata\src\poly\engine.cpp:2262-2264` 明确打印 *"--pcie-auto: the share fitting is not in this build; only --pcie-frac is …"*；`--pcie-frac` 固定值走 staging/direct 路径（`:2751-2764` 建槽、`:3157-3189` 选 miss 走 PCIe、`:2646` direct 设备别名）。
  - **按老师公式推我们**：R = 专家帧 3,096,576 B ÷ 每 miss 池 wall ≈96 µs = **32.2 GB/s**（R49 账，A）；L ≈ PCIe 4 x16 实测 26–28 GB/s ⇒ **f = 26/(26+1.85×32.2) = 0.304**（D）。我们 R39 实测最优点在 **0.2（direct +6.5%/−1.7%）与 0.35（0%）** 之间 ⇒ auto 会选 **0.30**，**与手调最优重合** ⇒ **相对手调只有 +0~2%（D）**。
  - 抄的代价：**~80–120 行**（probe 1 处 + PcieAuto 结构 1 个 + 解码窗口钩子 1 处；我们 `Stats` 已有 `cpu_ms`/miss 数/pcie 统计可喂它，见 `main.cpp:870-912` 的 ledger）；**质量是 class-E**（R39 同表：E2 43/64、E3 1/64 ids 变）。
  - **结论**：**有条件抄**——但对**单机交付**属于"自动调参"，不如先用固定 0.2–0.35 direct 拿 R39 那 +5.6%；若产品化给别家用，则 auto 值 +30%（慢 CPU 机，C）。

### §2.2 机制二：`STRATA_PCIE_BALANCE=1` 每层动态（commit `04be241a`，2026-10-14，只进 `v0.1.45-poly.1`）
- **commit**：`04be241a5a7ec016716cb50aab99be82a801e4fc`（2026-10-14，Niko1221，port of PR #1548 `a513394c4`）："decode: STRATA_PCIE_BALANCE=1 picks each layer's PCIe count from measured costs"。
- **算法**（commit body 原文）：
  1. 开 `wait_flag_ge_stamped`：GPU 侧就绪内核在三个时刻**写 GPU clock**（plan 到达、PCIe 部分开始、GPU 等 CPU 行）——**不加 launch**，只在开关开时捕获；
  2. host 每 step 计池时间；**每窗口后**用**带遗忘的最小二乘**拟合：CPU 侧 `a + c·n_cpu + d·n_pcie`（d 是"链路的 DRAM 读拖慢池"项——DRAM-bound 池把专家挪到 PCIe **不省**）与 GPU 侧 `g0 + g·n_pcie`；
  3. **每层**选它 miss 里走 PCIe 的 `m`，使**更长的那一侧最小**；**每第 64 层强制把一个专家换侧**，让两侧拟合都被持续测到；
  4. **关闭（默认）时 logits 与 32 token 与 v0.1.39 逐位同**。
- **老师实测（C 级，commit body）**：IQ2_XS，RTX 5090 + 9950X3D，聊到 turn 末：
  - **一个 AVX2 池 worker（模拟慢 CPU）**：16.77/17.17 → **12.98/12.82 ms/round**（拟合：CPU 42–60 µs/专家、GPU 28 µs/PCIe 专家；**0.8 的 miss 被挪走**）；
  - **9950X3D 的 16 workers**：3 对 **12.19 → 12.17**（**池 DRAM-bound，几乎不挪**）。
- **我们能不能用/值不值**：我们 CPU 池 = **1.53 GB/token ÷ 31.1 GB/s**（R49）⇒ **DRAM-bound**；老师 16-worker 同型实测 **≈0%**（12.19→12.17）⇒ **不抄**（且要改 kernel 加 clock 戳 + LSQ + 每层派发记账，约 150–250 行，收益 ≈0）。

---

## §3 预期收益与"先抄哪个"（我们 regime）

| 机制 | 是什么/怎么做 | 我们现状 | 预期变化量（等级） | 成本/风险 | 建议 |
|---|---|---|---|---|---|
| **固定 pcie 份额 + direct**（不是新组件，但 R39 已测） | `--pcie-frac 0.2~0.35 --pcie-mode direct` | 交付**当前没开 pcie**（`run_delivery_t16.cmd` 无 pcie 参数） | **+5.6% 绝对值**（R39 E2 中位 12.95 vs E0 12.26；A，旧配置需重测） | 0 行（只改命令行）；**class-E**（E2 ids 43/64；E3 1/64） | **先做（1–2 跑，零改码）** |
| **PcieAuto 全局份额** | 老师 `L/(L+1.85R)`，3 次更新（`aa7b0cb1`） | `--pcie-auto` 空壳（`engine.cpp:2262-2264`） | 对我们机器 ≈选 0.30，**相对手调 +0~2%**（D）；跨机器慢 CPU 可达 **×1.30**（C，老师 3060） | ~80–120 行；class-E；需重测 | **有条件：单机不抄；产品化再抄** |
| **`STRATA_PCIE_BALANCE` 每层** | 每层 m 由两侧拟合定（`04be241a`，v0.1.45-poly.1） | 无 | **≈0%**（D；我们 DRAM-bound，老师 16-worker 12.19→12.17，C） | ~150–250 行（kernel clock 戳+LSQ） | **不抄** |
| **PF_FUSED** | prefill fused int8 专家（`moe_fused_iq.cu`，`STRATA_PF_FUSED=1`） | 源码在树里但 poly 未接；K/格式不匹配 | prefill **+4~9%**（C）⇒ 每答案 **+1.6~3.8%**、端到端 **+0.6~1.3%**；decode 0% | 内核 300–600 + 接线 150–300 行；class-E | **现在不抄** |

**⇒ 先抄哪个（一句话）**：**先抄"固定份额 + direct 重测"（零码）；然后（若产品化/换机）抄 PcieAuto 全局（~100 行）；`STRATA_PCIE_BALANCE` 每层与 PF_FUSED 都不抄**（前者我们池 DRAM-bound，后者只 prefill 且几何不匹配）。

---

## §4 未验 / 边界
- **U1**：R39 的 +5.6% 是 10-18 早些时候 alt prompt、threads16、`--expert-swaps 0` 下的读数（A 级但**同批**）；今天交付配置已变（`sp_borrow.exe` + `STRATA_PROMPT_BORROW=1` + 交付表）⇒ **必须重测**（1–2 跑）才能写进交付。
- **U2**：PcieAuto 的 `R` 我们用 **96 µs/miss**（R49 交付形态的数字）推得 32.2 GB/s；交付形态若 miss 结构变化（borrow/表变了），`R` 会变，`f` 会在 0.25–0.35 间移动（D）。
- **U3**：PF_FUSED 的收益全是**外部/老师机器**（C）：5090/5070Ti/4080S/R9700 的 IQ3_S/IQ2_XS 数据；**我们 Q4_K_M + relu2（无 SwiGLU）的 Nemotron 没有同类内核**，收益只能算"prefill 时间 × 外部百分比"（D）。若真要做，先量我们 prefill 3.5 s 里 MMQ 占多少（现在没有 post-1938 的 nsys）。
- **U4**：`04be241a` 只在 **v0.1.45-poly.1** 的 tag 里（`git tag --contains`），0.1.44.1 树里 grep `PCIE_BALANCE` = **0 命中**（A）；所以"老师那一半"严格说只有 PcieAuto（0.1.42+）在老师实际跑的 0.1.40.2 之后才出现（0.1.40.2 无此机制）。
- **U5**：class-E 判定：R39 已证份额变化会改 ids（E2 43/64、E3 1/64），任何自动份额都应作 opt-in/性能模式，不进无损交付。
- **U6**：我们 `--pcie-mode auto` 在 direct 可用时就等于 direct（R39 §5 边界），不是份额自适应；别把两者混名。

---

## §5 复算命令（只读）
```powershell
# A. PF_FUSED 定义/前提/实测
Get-Content 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\prefill\moe_fused_iq.cu' | Select-Object -Skip 1059 -First 90
Select-String 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\bench\results' -Pattern 'PF_FUSED' -Recurse | Select-String '5.5|7.8|8.9|11%|12%'
# B. PcieAuto
Get-Content 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\upstream-0.1.44.1\src\program\generate.cpp' | Select-Object -Skip 1659 -First 122
git --no-pager --git-dir=C:\sp3\upstream-hist\polystrata.git show -s --format='%H|%ad|%s%n%b' aa7b0cb1
git --no-pager --git-dir=C:\sp3\upstream-hist\polystrata.git show -s --format='%H|%ad|%s%n%b' 04be241a
# 我们：空壳与 pcie 路径
Select-String 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\engine.cpp' -Pattern 'pcie_auto|pcie_frac_|pcie_slots_|pcie_direct_'
# R39 表
Get-Content 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R39-pcie-direct-20261018.md'
```
**判据（能变红）**：① `gu_covered` 里 CUDA 侧包含 Q4_K ⇒ §1.1 的"接不了"错；② `GU_ROWS_K` 不是 2560 ⇒ §1.1 K 不匹配错；③ PcieAuto 的公式不是 `L/(L+1.85R)` ⇒ §2.1 错；④ R39 的 E2 中位不是 +5.6% ⇒ §3 首行预期错；⑤ `04be241a` 里 16-worker 不是 12.19→12.17 ⇒ §2.2 错。

---

## §6 落盘
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-AE-老师PF_FUSED与PCIE_BALANCE-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-老师PF_FUSED与PCIE_BALANCE.md`
