# 计划 · Stage D：GPU 侧 T 布局（消除 GGUF row-major down）— 2026-10-18 立项提案

> 提案人：builder3（施工席 C）· 状态：**立项待批** · 依据：R53（Stage A sidecar + 几何/可逆）、R56（Stage B loader/自证/索引 + A2/A3/A4）、
> R57（Stage C 位精确 T 内核 + GPU ids 逐位同 + 方案 A/B 的 RAM 死路论证）· 零 GPU、不改码、不编译。

## 1. 目标（一句话）

**让 GPU 侧也读 neuron-major（T）布局**，从而 GGUF 里的 row-major down 不再是必需副本 ⇒ 模型文件从 **70.1 GB 降到 38.4 GB**，
pin 预算闭合为 **up 29.53 + sidecar 29.53 = 59.06 GiB**（与今天完全相同），decode down 相 26.9 → **5–10 ms**，
墙 65.9 → **~46 ms ⇒ +35~40%**（且按 R57 位精确路线可保持 ids 逐位同）。

**为什么这是唯一自洽解**（R57 §7 已证）：
- 现状三份 down 数据：GGUF row-major down 29.53 GiB（prefill/GPU 全量需要）+ sidecar T down 29.53 GiB（decode CPU 需要）+ up 29.53 GiB（两面都要）> 96 GB RAM ⇒ 无论怎么 pin 都有一个方向崩（A：sidecar 不驻留 ⇒ decode 112 ms/token；B：GGUF down 不驻留 ⇒ prefill −90%，且 avail RAM 240 MB）。
- 若 GPU 也吃 T：GGUF 不再需要 row-major down ⇒ 只有 **up（29.53）+ T down（29.53）** 两份，pin 预算与今天相同（59.06 GiB），prefill staging 的 down 也来自 pin 住的 sidecar（同布局直接 DMA，无反变换、无 gather）。

## 2. 改动清单（展开 R57 §7 的论证）

| # | 件 | 内容 | 估计 |
|---|---|---|---|
| D1 | **小批 GPU down kernel** | `native_mmvq_ids`/`native_mmvq` 的 IQ4_NL down 分支：从"按行主序读 1512 B/行"改为"按神经元读 512 B 连续块 + nibble 取行"；活神经元跳过的粒度按 32 值块（与 CPU T 内核同一可跳集） | ~150–250 行 CUDA |
| D2 | **prompt-batch MMQ down** | `mmq_->mul(sbuf + down_at, d_type, out_w, ff, L.d_bytes, ...)` 的 down 内核改为 T 布局（或专用 T 版 mul；**不做 T→row 的 GPU 转置**——1.55 MB/专家 × 29k staging = 45 GB 转置流量，不可取） | ~100–200 行 CUDA |
| D3 | **槽布局** | `Layer::slot` 内容 = `up | downT`（down 段从行主序换 T，尺寸不变 1,548,288 B）；`slot_down()` 偏移不变，`expert_slot_bytes()` 不变 | ~10 行 + 注释 |
| D4 | **staging 布局** | `stage()`：up 从 GGUF pinned 直接 DMA（不变）；down 从 **sidecar pinned** 直接 DMA（T 块，不再从 GGUF gather）；`pin_expert_arena` 只 pin up；`relu2_down_open` pin sidecar（B 的三处改动直接复用，R57 已实现） | ~40 行（已在树里，改数据源） |
| D5 | **fill / swap** | `fill_expert_cache` 与 swap 的 down 拷贝改从 sidecar DMA（同 T）；up 仍从 GGUF pin | ~30 行 |
| D6 | **加载器/模型文件** | 方案 D6a（推荐，零大文件复制）：保留现 GGUF 文件，但**加载器不再映射/不需要其 down 张量**（内存上等价于"无 down"；盘上仍多 31.7 GB，但 E: 剩 468 GB 够）。方案 D6b（省盘）：另做 `tools/strip_down.py` 输出 38.4 GB 无 down GGUF，需 70 GB 级重写 + 逐张量校验 | D6a ~30 行；D6b ~150 行 + 一次 70 GB 重写 |
| D7 | **CPU 侧** | 全部复用 R57：`relu2_down_t.hpp` + `CpuExpert.downT` + `Relu2Active` 索引 + 自证行 | 0 行 |
| D8 | **验证工具** | ① CPU parity（已有 `C:\sp3\stageC-parity.cpp`，0 差）② **GPU-T 内核 vs GPU-row 内核逐位 parity**（同一专家双布局喂两内核，CUDA test exe，零 GPU 数据准备 + 1 跑） | ~120 行 test |
| D9 | **开关** | 全链只由 `STRATA_RELU2_DOWN=0|1` 控制（默认 0 = 今天的 row 布局逐位等价）；模型/槽/staging 的 T 态只在 =1 时出现 | ~10 行 |

**复用清单（不需要重做）**：Stage A sidecar `E:\sp3-sidecar\downT.bin`（31.71 GB，20,480 专家）+ manifest；Stage B `relu2_down_open`/自证行/活索引；Stage C `relu2_down_t.hpp` 位精确 CPU 内核 + parity 工具 + GPU 实测 ids 逐位同的证据。

## 3. 风险

1. **位精确（最大风险）**：**GPU 的 float 累加结构与 CPU 不同**——现有 ids 是"GPU 驻留专家（mmvq/MMQ）+ CPU 漏算专家（ggml vec_dot）"两种结果按固定顺序加权求和得到的。R57 证明了 **CPU 侧**可以逐位复现 ggml vec_dot；但 **GPU-T 内核必须逐位复现 GPU-row 内核**（mmvq/MMQ 自己的累加顺序/树），否则 ids 会漂。⇒ **D8② 必须先零 GPU/1 跑做到 0 差再动主路径**；做不到就退回"类 E 接受"或放弃。
2. **模型文件迁移**：D6b 要重写 70 GB 级文件（低风险但耗时、需逐张量校验 + 版本号/标记）；D6a 不动文件但盘上多 31.7 GB（E: 现剩 468 GB，可接受）。
3. **prefill 不倒退是硬门**：staging 的 down 从 pinned sidecar DMA 的前提是 sidecar 与 up 同时 pin 住（59.06 GiB）；若 pin 失败（RAM/驱动）⇒ 响亮拒绝并回 row 路径。
4. **v3 全套门**：ids（≥2 异构 prompt）+ 全位置 Δp + 确定性 + 同批 CTRL + 3 跑中位；peak ≤31,700。
5. **寄存器/性能**：GPU-T 的 down 读 512 B/神经元而非连续行，gather 效率可能把 +35~40% 压到 +25~30%（上限仍在，见 §5）。
6. **回滚**：`STRATA_RELU2_DOWN=0` 必须逐位回到今天的路径（Stage D 的全部改动默认关）。

## 4. 验收判据（预注册）

| # | 判据 | 绿 | 红（怎么变红） |
|---|---|---|---|
| G1 | **零 GPU parity（两层）** | ① CPU T 内核 vs ggml vec_dot 0 差（已有）② **GPU-T 内核 vs GPU-row 内核逐位 0 差**（D8②） | 任一用例有 1 位差 ⇒ 停手，不得上主路径 |
| G2 | **ids 逐位** | 与旧 exe（`sp_sidecar3`/`sp_donly` 同配置）**≥2 异构 prompt 逐位同**（alt/qa/code 各 64 token） | 任一位置不同 ⇒ RED（不接受"类 E"漂；位精确是立项前提） |
| G3 | **down 相** | ≤ **10 ms/token**（现 26.9；R57 内核计算下界 5.0 ms） | >20 ms ⇒ 停手查 I/O |
| G4 | **prefill** | ≥ **1700 tok/s**（交付 1938.8 的 −12% 内；D0 基线 127.3 是 alt 293，需用同口径） | 掉出 1700 ⇒ RED |
| G5 | **RAM 三条** | 进程 WS 增量 ≤1 GiB（除 pin）· avail RAM 最低 ≥8 GiB · down 相带宽 ≥25 GB/s | 任一条 ⇒ RED |
| G6 | **峰值** | ≤31,700 MiB | 超 ⇒ RED |
| G7 | **默认关** | `STRATA_RELU2_DOWN` 不设 ⇒ 与 D0 基线 ids/日志/staging 全同 | 任一同 ⇒ RED |
| G8 | **确定性** | 同配置 2 跑 ids 逐位同 | 不同 ⇒ RED |

## 5. 预估

| 项 | 估计 |
|---|---|
| **文件数** | GPU 内核 2–3 个（`native_mmvq.cu`、`s2_expert_grouped.cu`、MMQ 相关）+ `engine.cpp/.hpp` + `cpu_experts.cpp`（复用）+ 新 `tools/`（D6b 可选）+ 新 CUDA parity test |
| **行数** | CUDA ~250–450 行（D1+D2+D8②）· C++ ~100–150 行（D3–D5 改数据源）· 工具 ~150 行（可选） |
| **编译次数** | **2–3 次**（一次全改 + 1–2 次修 parity） |
| **GPU 跑数** | **8–12 跑**：parity 验证 1–2 跑；主路径 A/B（OFF/ON ×2）+ CTRL ≈ 5–6 跑；质量门 Δp/确定性 2–3 跑；总计 ~3–5 分钟纯 GPU |
| **工期** | 一轮专注施工（含 parity 调试） |
| **收益** | 位精确前提下 **+35~40%**（down 26.9→5–10；墙 65.9→~46 ms）；若 GPU gather 效率差则 +25~30% |
| **盘/RAM** | D6a：盘 +0（沿用现有 GGUF+sidecar）；D6b：盘 −31.7 GB。RAM：pin 59.06 GiB（与今天同） |

## 6. 建议的施工顺序（先零 GPU 可判死的三步）

1. **D8② GPU-T vs GPU-row parity**（CUDA test exe；用现成 sidecar + GGUF 双布局喂两内核；1 跑内可判死）——**这一步不过，整个 Stage D 停**（和 R57 的 CPU parity 一样，把原则问题放在最便宜的地方判）。
2. **D3+D4+D5（槽/staging/fill 的数据源切换）+ D9 开关**：默认关，先只让 GPU 驻留专家的 down 来自 T，跑 ids 门。
3. **D2（prefill MMQ down）**：prefill staging 改 T，测 prefill 是否 ≥1700。
4. 最后做 D6b（可选，省盘），并跑 v3 全套质量门 + 3 跑中位。

## 7. 落盘与依据（绝对路径）

- 本提案：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\计划-StageD-GPU侧T布局-20261018.md`
- 立项回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R57b-StageD立项-20261018.md`
- 前置证据：`…\回执\R53-ReLU2-sidecar-StageA-20261018.md`（sidecar 31.71 GB/20,480 专家/可逆 0 diff）· `R56-ReLU2-StageB-20261018.md`（loader/索引/自证/A2/A3/A4）· `R57-ReLU2-StageC-20261018.md`（位精确 T 内核、CPU parity 0 差、GPU ids 逐位同、方案 A/B RAM 死路）
- 代码现状：`PolyStrata\src\poly\relu2_down_t.hpp`（CPU T 内核）· `cpu_experts.cpp/.hpp`（`downT`/`Relu2Active`）· `engine.cpp/.hpp`（loader/pin/staging 的 B 版改造，默认关）
- 工具：`C:\sp3\stageC-parity.cpp`（CPU parity + bench）· `C:\sp3\run_R57_all.ps1` / `run_R57D_all.ps1`（跑臂脚本）
