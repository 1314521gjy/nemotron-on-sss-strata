# 项目书：Nemotron 3.5 Super VL 120B A12B × Strata 专武适配（第三号）

- 日期：**2026-10-15**
- 状态：**待决策基座分叉（§2.4）**；在此之前不动一行代码、不编译、不跑 GPU
- 证据等级：`A`=本机实测 ｜ `B`=同机另一条线 ｜ `C`=外部实测（我读到的线上读数）｜ `D`=推论/估计
- 本件的定位：**计划与判据的唯一出处**。事实各有归属处，本件只**指路**、不复刻正文（例外：本会话新测出的读数在此登记，因为它们是新的）
- **追加状态（2026-10-16）**：**基座已裁决 = B**（`VecSzn/PolyStrata`，HEAD `eb5ef11`；接受升一代上游 1.43.1→1.44.1）。P0 / P1 / U2 / U3 / U4 / U14 / P2 均已收口；**修正登记与指针见 §10**。**上文 §0–§9 原结论一律保留不改**（§10 只并列新旧）。

---

## 0. 一句话

把 **Nemotron 3.5 Super VL 120B A12B**（Mamba-2 + MoE + 注意力混合）接上 **Strata 的 MoE 专家流式加载体系**，做成第三号专武；**新写的核心只有一个层型——Mamba-2 cell**，其余是几何泛化与打包。

**为什么值得做**：Strata 那套专家驻留/分层/流式加载（`--expert-cache*` / `--expert-profile` / `--pcie-frac` / `--peer-device` / CPU 瀑布）是 **NInfer 线上没有的能力**；从 NInfer 走要从零造。**它是资源，不是对手。**

---

## 1. 靶子

| 项 | 读数 | 等级/出处 |
|---|---|---|
| 模型 | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` | `C` HF/ModelScope |
| 架构 | **Mamba-2 + MoE + 注意力交错** · C-RADIO 视觉编码器 · VL connector · MTP | `C` 官方 README |
| 规模 | **120B 总参 / 12B 激活** · 上下文至 **1M** | `C` 官方 README |
| 许可 | **OpenMDW-1.1**（README 明写可商用） | `C` |
| 精度族 | 官方 4 个仓：**BF16**(115.91 GiB, 65 文件) · **Base-BF16** · **NVFP4**(88 分片, **69.86 GiB**) · **NVFP4-LiLiCorr**(草稿) | `A` ModelScope API + hf-mirror API 实测 |
| 第三方 | **`RedHatAI/…-FP8`**(dl=73) · **GGUF 4 仓**（ggml-org / unsloth / lmstudio-community / natalie5） | `C` |
| NVFP4 配方 | 专家 NVFP4 / 注意力·KV·Mamba linear·lm_head **FP8** / 视觉·嵌入 BF16 / router FP32 / **SSM cache FP16 + 随机舍入**；**4.82 BPE** | `C` NVIDIA 量化文档 |
| 量化代价 | PTQ + 200 步 QAD ⇒ 35 基准**中位恢复 99.52%**；掉点项：WebVoyager −14.7%、DeepSWE −26%、MMLU-ProX −4.1% | `C` 同上 |
| 官方部署矩阵 | 1×DGX Spark / 1×H200 / 2×H100 / 1×GB200（**无 Ada**） | `C` |
| 本机 | **RTX 4080 SUPER 32,760 MiB · sm_89 · driver 616.56** ｜ RAM **95.8 GB** ｜ 24 逻辑核 ｜ E: 579.6 GB | `A` `nvidia-smi` / WMI |

> ⚠️ 权重来源建议：**绕开 NVFP4，从 BF16 版走自家量化线**（`KP-STRATA / M9` 已记过 NVFP4 "除以 `d_w`" 的舍入坑）。此为本件建议，未验。

---

## 2. 基座

### 2.1 候选（`A` 本机 + `C` 线上）

| 代号 | 位置 | 身份 | 状态 |
|---|---|---|---|
| **A** | `E:\strata-v01431` | 上游 `Niko1221/Strata` **v0.1.43.1**，HEAD `a0116b2782a5b82622358c04a06496485d12b2d6`，`describe=v0.1.43.1`，51 tag，46.9 MB | **已定**；与现役二进制同版（`E:\strata\engine\strata.exe` sha16 `AA690AE65BFD306D`，二进制内含 `0.1.43.1`）|
| **B** | `VecSzn/PolyStrata` | **Strata 的多模型化成品**：126★/26 fork/**MIT**/3293 文件/`push=2026-10-15` | **新发现，待裁决** |
| **C** | `E:\cyber-on-strata` | Cyber 专武成果（`qwen35moe` = GDN+MoE），0 未提交 | 参照清单 |
| — | `E:\strata` | 上游 main + **84 条未提交** + **现役引擎所在** | **⛔ 不碰**（`E:\视频\_agents\计划-C工程-执行层-20261015.md:123` 明列为"别人的"）|

### 2.2 基座 B 的关键读数（`C`，本轮新测；这是本件最要紧的一条）

- PolyStrata = **两个引擎**：Qwen3.8-Flash-Next 用 **Strata 引擎原件（未改）**；**另外 7 个模型用它自己的引擎**（`src/poly/`）
- 目录结构：`src/{artifact,core,kernels,ngram,plan,platform,**poly**,prefill,program,qwen35,spec}` · `include/strata/…` · `metal/` · `sycl/` · `serve/` · `tools/` · `ref/` · `bench/` · `tests/`
- 已见的分后端件：**`src/poly/gemma4.cpp`** · **`src/poly/deepseek4.cpp` + `.hpp`**
- ⭐ **`include/strata/core/arch_defaults.hpp` + `src/core/arch_defaults.cpp` + `src/core/arch_defaults_test.cpp`** —— **这就是"把编译期几何泛化出来"的那一层，还带测试**
- 关键词命中（全仓 3293 文件）：**`mamba` 0 · `ssm` 0 · `recurrent` 0 · `conv1d` 0 · `state_space` 0 · `nemotron` 0**；`gdn` **21**
- 它已跑的 7 个模型：GLM 5.3 Flash / DeepSeek V4 Flash / **LFM2.5-24B-A2B** / Qwen3.6-35B-A3B / Ornith 1.5 / Gemma 4 26B-A4B / Tiel Coder 35B-A3B
- **LFM2.5 值得注意**：官方 README 说它 *"40 层里大部分没有注意力…其中 30 层用一个只看前三个词的短卷积来混合词，每层只记两个向量"* ⇒ **PolyStrata 已经处理过"以非注意力层为主"的架构**（最接近 Mamba 的先例）
- Windows：Qwen3.8-Flash-Next + **5 个小模型**可 Windows；GLM/DeepSeek 仅 Linux
- `docs/` 里有 **`BACKENDS.md`（怎么加一个后端）** · **`NVFP4.md`** · `MODELS.md` · `FAQ.md`
- ⭐ **上游 Strata 自己开始多模型化**：PolyStrata README 记 *"Strata 从 0.1.44 起也能用它自己的引擎跑 Qwen3.6（`--family qwen36-strata`）"*

### 2.3 基座 A 的关键读数（`A`，本轮新测）

- `mamba` 字面 **仅 2 处，都是注释**：`src/kernels/gdn_parity.cpp:157`、`src/kernels/elementwise_parity.cpp:76`（*"`ssm_a = -exp(A_log) < 0`"*）
- 但 Mamba 的原料在场：`fused_gdn_ab(..., const float* dt, const float* ssm_a, ...)`（`src/kernels/cuda/fused_gdn.cu:138`）· conv 历史 · S×S 逐头状态 · `gdn_step`
- ⇒ **现役 recurrent cell = GDN（门控 delta 规则，带 q/k L2 归一与 beta/gate），不是 Mamba-2 的选择性扫描**
- `include/strata/core/layout.hpp`（97 行）：层交错是**单一周期**——`qsa_interval = 4` + 派生 `n_qsa_layers()`/`n_gdn_layers()`/`is_qsa_layer() = (layer % 4 == 3)`
- `setup.py:195`：`MIN_ENGINE = (0, 1, 43, 1)`，其注释自带 **0.1.19→0.1.43.1 的逐版卷宗**
- 编译工具链（从 `build\CMakeCache.txt` 读到，`A`）：**Ninja** + `cl.exe`(**VS18 BuildTools / MSVC 14.51.36231**) + `nvcc` = **`E:\cuda-13.3`**（注意：PATH 里的 `nvcc` 是 `v12.8`，**不是**它用的那套）

### 2.4 ⚠️ 需要裁决的分叉

| 路线 | 做法 | 代价 / 收益 |
|---|---|---|
| **A→ 在 1.43.1 上做** | 照 lcz 那篇的结论泛化几何 + 新写 Mamba-2 cell | 已定；上游单模型设计意图与我们相反，每次升版冲突 |
| **B→ 以 PolyStrata 为基座/上游** | 直接继承 `arch_defaults` 泛化层 + `src/poly/` 分后端结构，只加 `nemotron` 后端 | 省掉"泛化几何"这一大块；**但**它是**第三方**仓（126★，10-06 才建），且**未验**它是否已有非注意力 cell |

**我的建议（`D`）**：**先做 §5 的 P1（只读判定 PolyStrata），再拍基座**。若 P1 证明 `src/poly/` 能容纳新层型，则走 **B**；否则回 **A**，但**照抄 PolyStrata 的 `arch_defaults` 结构与 PR #1518 的 `add_subdirectory(<model>_ext)` 外挂结构**。**我不擅自改你已经定的 A。**

---

## 3. 缺口（分层）

| # | 项 | 判定 | 依据 |
|---|---|---|---|
| 1 | **recurrent cell = Mamba-2** | **需新增**（原料已在：A_log/dt/conv/state） | §2.3 |
| 2 | **层交错**：现为单一周期 `qsa_interval` | **待定**：若 Nemotron 交错**非等周期** ⇒ 必须改成**显式层型表** | `layout.hpp:30,56-64` |
| 3 | 几何常量散落多文件 | **需按 1.43.1 现行路径重核**（lcz 那篇引的是 **0.1.2x 路径**，不能直接用） | `C` lcz.me/topic/2153 |
| 4 | `general.architecture` 守卫（只认 `qwen4exp`） | **需换一族**（Nemotron 的元数据键名不同） | `C` 同上 |
| 5 | 视觉（C-RADIO + VL connector） | **需新增**（Strata 是 `strata-vision.exe` + mmproj 路线，不同族）⇒ **建议后置，先只做文本** | `C` |
| 6 | 混合精度（FP8 attn/KV/Mamba linear/lm_head · **FP16 SSM cache + 随机舍入**） | **需跟随** | `C` NVIDIA 量化文档 |
| 7 | 1M 上下文（YaRN） | **零实现**（Strata 侧）⇒ 与 Qwen4 预研同款结论：**暂不做** | 本机既有台账 |
| 8 | 打包（专家帧格式） | **从 BF16 走自家线**，可评估 `StrataShift` | `C` |

---

## 4. 外部既有工作（立项检索结论，`C`）

| 线索 | 读数 | 对我们的意义 |
|---|---|---|
| Strata **issue #2657**（2026-10-15 开，**open，0 评论，0 认领**）| "Support: Nemotron 3.5 Super VL 120B A12B"（作者 AMD 6800XT，想 16 GB 显存 + 16 GB 内存） | **需求侧有人喊，供给侧没人做** |
| `repo:Niko1221/Strata` + `mamba` | **0 命中** | Strata 世界**从未碰过 Mamba** |
| `strata+nemotron` / `nemotron+super+vl+engine` 仓库检索 | **各 0 命中** | **没有别人的"Nemotron 专用引擎"立项** |
| **PolyStrata** | §2.2 | **最相似的工作**，但**不含 Mamba** |
| **Strata PR #1518**（open, 53 文件, `perronemirko/Strata:deepseekv4-flash`） | 做法：**`add_subdirectory(dsv4_ext)`，核心不动** | ⭐ **工程结构先例：新模型做外挂子目录** |
| **Strata PR #2321**（open，带 parity 测试，已有人验 `0 mismatches`） | 把 `route-resident` 从 **512/10 泛化到 256/8** | 现成可借的几何泛化件 |
| Strata issue #1907 / #1501 / #310 / #115 | 全是"能不能支持别的模型"，**全 closed、无一落地**；#1501 里作者自述成果被 **repo clean-up 抹掉** | 需求长期存在、实现极易蒸发 ⇒ **落盘纪律要更硬** |
| **lcz.me/topic/2153**《改源码跑 DeepSeek V4 Flash》 | 同路先例的**失败报告**：*层型是位置算术*（与我方独立读数一致）· 常量散落 14 文件 · 粗估 **6,000–10,000 行**新后端 · **"调度框架能复用，调度参数不能"** | 唯一的外部工期估计；也是我们最重要的风险清单 |
| **llama.cpp PR #30520** | **state=closed, merged=true**，15 文件，2026-10-14 | ⭐ 而且 `src/models/` 现成有 **`mamba2.cpp` · `mamba-base.cpp` · `nemotron-h.cpp` · `nemotron-h-moe.cpp` · `nemotron.cpp`** ⇒ **cell 算法有现成来源**（照 `KP-NINFER-0086`：算法层可搬、接口层重写） |
| `devco-hyunha/StrataShift` | "PolyStrata-compatible MoE streaming **convert/quantize** toolkit" | 打包线可评估借用 |
| 上游版本 | **已有 `v0.1.44.1`**（`296b59d2`，2026-10-14）；`v0.1.43.1..v0.1.44.1` = **282 提交 / 358 文件 / +47,208 −1,243** | ⚠️ `罗盘.md` §0.12 "上游最新 = v0.1.43.1，没有更新的了"**已过期**（旧结论保留，此处只加新读数） |

---

## 5. 阶段计划（每步：动作 / 判据 / 负控 / 回滚）

### P0 — 缺口表（只读；需 28 MB 下载）
- **动作**：取 Nemotron `config.json`(8.67 MB) + `model.safetensors.index.json`(19.5 MB)，抽真实几何（层数 / 专家数 / top-k / **Mamba↔Attention 交错序列** / Mamba 参数族 / 每专家帧字节 / MTP 与视觉几何）；同时按 **1.43.1 现行路径**重核"几何散落在哪几个文件"
- **判据（能变红）**：`层数 × 专家数 × 帧字节` 与落盘量**闭合**
- **负控**：**拿 Flash-Next 跑同一抽取脚本**，必须复现已知几何（`48 × 512 × 2,764,800 B = 63.28 GiB`）。复现不出 ⇒ **是脚本错，不是模型新**
- **回滚**：无（只读产出）

### P0-b — 基座裁决（决策人）
### P1 — PolyStrata 判定（只读；需 35 MB clone 或纯 API 读）
- **动作**：读 `src/poly/` 全表 + `arch_defaults.{hpp,cpp}` + `docs/BACKENDS.md`
- **判据**：能否指出"**新层型挂在哪一个函数/分派点**"，并给出 `poly` 引擎与 `strata` 引擎的**同源程度**（是 fork 改名还是独立实现）
- **负控**：找一个**已知存在**的非注意力层先例（LFM2.5 的 30 层短卷积）——**找不到就得报"未验"，不许说"没有"**

### P2 — Mamba-2 cell 探针（最小代价，不碰主线）
- **动作**：只做**一层 Mamba-2**（合成权重）对参考实现数值
- **算法来源**：llama.cpp `src/models/mamba2.cpp`（已 merged，MIT）
- **判据**：容差内对齐
- **负控**：故意错一个 `dt` ⇒ **必须**偏差超限或硬抛（Strata 规矩：几何不匹配硬抛，**不许静默降级**）

### P3 — 打包线
- **动作**：从 **BF16 版**量化到 Strata 的专家帧格式；评估 `StrataShift`
- **判据**：`verify_experts` 零残渣（照 `strata_pack.py` 口径）

### P4 — 几何泛化
- **动作**：四个中枢（唯一出处：`E:\视频\_flashnext\experiments\20261006-whittle-on-strata\RUN-20261006-阶段19-移植面全图.md` §1）+ **1.43.1 现行**的写死形状清单 + `arch` 守卫 + （若需要）**显式层型表**
- **判据**：`--help`/启动日志里该模型的几何自证字段与实际一致

### P5 — 端到端
- **动作**：先文本 32K；视觉、1M 后置
- **判据**：与 llama.cpp 同模型同提示**输出一致性**在容差内 + decode/prefill 读数

---

## 6. 风险与红线

1. **不许碰 `E:\strata`**（别人的树 + 现役引擎 + 84 条未提交）
2. **上游设计意图与我们相反**（官方第一句："Strata runs one model"），但**上游 0.1.44 已开始 `--family` 多模型化** ⇒ 冲突面会变；对策：**外挂子目录**（照 PR #1518）
3. **一次编译原则**（共识 §6）；**每次编译前清 `.obj`**——该树头文件依赖表为空（中文 `cl.exe` + CMake `/showIncludes`）⇒ **增量编译不可信**（唯一出处：`…\RUN-20261006-阶段20-适配落地与假绿根因.md`）
4. **内存边界**：GLM 5.3 Flash 3.0BIT 官方要 **96 GB 内存**，本机 **95.8 GB** ⇒ 同规模的 Nemotron 必须**精算驻留集**，不许"先跑起来再说"
5. **混合精度**：FP8 注意力/KV + **FP16 SSM cache + 随机舍入**，新 cell 上要跟着落地，不能只做权重
6. **产出物易蒸发**（#1501 的先例）⇒ 每一步落盘，判据与负控一起落

---

## 7. 未验清单（明写，不许当结论用）

- Nemotron 的 Mamba/Attention **交错是否等周期**（决定要不要层型表）
- Nemotron 专家几何（n_expert / top-k / 帧字节）——未抽
- PolyStrata `src/poly/` 是否已有非注意力 cell；`poly` 与 `strata` 的同源程度
- "几何散落 14 文件"在 **1.43.1** 里的现行清单（lcz 那篇引的是 0.1.2x 路径）
- 本机 32 GB 显存 + 95.8 GB 内存**能否装下** Nemotron 任何量化档（未算，需 P0）
- 我们的 `E:\strata-v01431` **能否原样编出一个可用引擎**（工具链在 CMakeCache 里有记录，但**我没编过**）

---

## 8. 出处

**本机（绝对路径）**
- 基座源码：`E:\strata-v01431`（`git describe` = `v0.1.43.1`）
- 中枢与移植面（唯一出处）：`E:\视频\_flashnext\experiments\20261006-whittle-on-strata\RUN-20261006-阶段19-移植面全图.md`
- 假绿与增量编译教训：`E:\视频\_flashnext\experiments\20261006-whittle-on-strata\RUN-20261006-阶段20-适配落地与假绿根因.md`
- Cyber 专武（前例）：`E:\视频\_flashnext\experiments\20261008-cyber-on-strata\状态收拢-20261009-Cyber专武适配-v3.md`
- 不碰清单：`E:\视频\_agents\计划-C工程-执行层-20261015.md`（第 123 行）
- 当行主线：`E:\视频\_flashnext\罗盘.md`（§0.24 赶超 Strata；§0.12 版本读数已过期）

**线上**
- 靶子：<https://modelscope.cn/models/nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4> · [NVFP4 仓](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
- 量化配方：<https://docs.nvidia.com/nemotron/latest/nemotron/super35/quantization.html>
- 基座上游：<https://github.com/Niko1221/Strata>（25,232★）· [#2657](https://github.com/Niko1221/Strata/issues/2657) · [#1518](https://github.com/Niko1221/Strata/pull/1518) · [#2321](https://github.com/Niko1221/Strata/pull/2321) · [#1907](https://github.com/Niko1221/Strata/issues/1907) · [#1501](https://github.com/Niko1221/Strata/issues/1501)
- 多模型化成品：<https://github.com/VecSzn/PolyStrata>
- cell 算法来源：<https://github.com/ggml-org/llama.cpp/pull/30520>
- 同路失败报告：<https://lcz.me/topic/2153>
- 打包工具：<https://github.com/devco-hyunha/StrataShift>

---

## 9. 需要拍板（三件）

1. **基座分叉裁决**（§2.4）：先做 P1 再拍？还是坚持直接在 A（1.43.1）上做？
2. **允许下 28 MB**（Nemotron `config.json` + 索引；**不碰权重**）以开工 P0
3. **允许 clone PolyStrata（约 35 MB）** 做 P1（或我只用 API 读，不落盘）

---

## 10. 追加（2026-10-16）：裁决结果与修正登记

> **本节为回写追加，不删不改上文 §0–§9 任何原结论句。** 依据：P0 / P1 / U2 / U3 / U4 / U4b / U14 / P2 主件（见 `台账-第三号专武-20261016.md` §1 与 `基座裁决材料-20261016.md`）。**并列新旧，决策理由级不替换**；读数标 `A/B/C/D`。

### 10.1 ★ 裁决结果（2026-10-15）

- **基座 = B**：以 `VecSzn/PolyStrata` 为上游；工作树 = 实验区 `PolyStrata\`，HEAD `eb5ef11`。
- **接受"升一代"**：B 内嵌上游 = **v0.1.44.1**（= 基座 A `v0.1.43.1` 的下一代）⇒ **与现役二进制（`strata.exe` 内含 `0.1.43.1`）脱离同版**（新增最高级代价，原文未记）。
- 依据：base-judge（task #4）三硬理由 + P0 非等周期 + P1 挂点/先例；U1 正向关闭。
- **翻盘闸门已清空**：U1 ✅正 · U2 ✅正 · U3 ✅解（放行但未测）· U4/U14/U15/U16 ✅结。
- 原 §2.4 的 A/B 并列表**保留**（W1 只新增裁决结果句）。
- **裁决 2（同日，2026-10-15）——量化靶 = GGUF `Q4_K_M`（65.33 GiB）**：格式 GGUF（非 NVFP4、非 BF16 自家量化线）。repo `lmstudio-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`（`Q4_K_M-00001-of-00002` 37.00 + `-00002` 28.29）；视觉 `mmproj-…-BF16`(1.61) 后置；备胎 `unsloth UD-IQ3_XXS`(55.66)；排除 `UD-Q4_K_XL`(80.94)。理由（`D`）：已排除 NVFP4 + BF16 被 U3 证伪；Q4_K_M 比 NVFP4 更小且 poly 立身即 **GGUF mmap + CPU 专家**（天然匹配）；MoE 512/top-22 对专家质量敏感（社区甜点档）；`GSQ-RCO` 暂无 Nemotron 档但产物亦 GGUF（可无痛换档，挂账）。**对下游**：**P3 改向**（评估现成 GGUF 直接被 poly 读 / 或转专家帧）；**P2 的 fp16+SR 仍有效**；本机靶体积 69.86→65.33（ram 账更松）。
- **下载已完成（task #13，2026-10-16）**：二片落 `E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\`（00001 39,730,340,416 B / 00002 30,372,951,904 B，合计 **65.29 GiB**）；**sha256 = HF LFS oid 逐字同**（`7e8ccf2e…` / `6d256028…`），**KP-UNI-0205 过**（非只比大小）；`sha256_local.txt` 同目录。

### 10.2 修正登记（W1–W12；旧读数见对应 §，此处只加新读数）

| # | 位置 | 旧读数 / 定性（原文保留）| 新读数（2026-10-16）| 等级 |
|---|---|---|---|---|
| W3 | §1 靶子表 | BF16 = **115.91 GiB, 65 文件** | **231.84 GiB**（≈2×，疑似口径错；BF16 参数 124.45B）| `C` |
| W12 | §1 旁注（"绕开 NVFP4，从 BF16 走自家量化线"）| 建议走 BF16 | ⛔ **本机不成立**：BF16 `poly_ram_need` ≈ **206 GB RAM** vs 本机 **95.8 GB** | `D`(算式)/`C`(体量) |
| W13 | §1 靶子表旁注（**裁决 2，新**）| 曾推 NVFP4（W12 后） | ⭐ **量化靶定为 GGUF `Q4_K_M`（65.33 GiB）**（`lmstudio-community/…-GGUF`，37.00+28.29 两分片）；视觉 `mmproj-…-BF16`(1.61) 后置；备胎 `unsloth UD-IQ3_XXS`(55.66)；排除 `UD-Q4_K_XL`(80.94，撞空闲 79)。**本机靶体积 65.33 比 NVFP4 69.86 更小** ⇒ U3 ram 账更松 | `D`(选型) |
| W14 | §3 第 8 条（打包/专家帧格式）| 「打包：从 BF16 走自家线」 | ⚠️ **走 poly 路径「专家帧格式」不存在**：poly **直读 GGUF 的 3D `_exps`、不打帧**（全 7 模型无 pack/convert）⇒ 该条**只对上游 A 成立**；**走 B 应记「无打包环节」**（改的是接线 `load_moe`，非打包）| `A(码)` |
| W15 | §3 缺口 #2 / §7（层型表机制）| 曾记「`head_count_kv` 数组 = 层型表，poly 机制天然适配」 | ⚠️ **更正**：数组 0 **同时是 mamba 层与纯 MoE 层**（`kv==0 && n_ff==0`）⇒ 只能分「注意力/非注意力」；**层型表改按「张量在场」建**（`ssm_in`/`ffn_gate_inp`/`attn_q`；`qwen35.cpp:86` 同法，`head_count_kv` 作交叉校验）。**层型表由 GGUF 数据驱动这一结论不变**，换的是判据 | `A(码)` |
| W16 | §4（PR #30520 日期）| 「llama.cpp PR #30520 … 15 文件，**2026-10-14**」 | ⚠️ **15 文件对，但 10-14 是 `created_at`；`merged_at=2026-10-15T13:34:57Z`**（merge `cd0a5b6a180d`）⇒ 记「**created 10-14 / merged 10-15**」（**参考实现落地于昨晚**）。且其 `tensor_mapping.py +14 行全是视觉张量**（`# Nemotron Omni`）、**未改 `nemotron-h-moe.cpp`** ⇒ **我们要的 MoE arch 非该 PR 带来** | `A` |
| W4 | §3 缺口表 / §7 | 隐含主干 **48 层** | 主干 **88 层** = 40 mamba + 40 moe + 8 attn（mamba/attn 与 moe **各自独立成层**）。**MTP = 1 个「折叠块」**（块号 88，内含 attention+MoE 两子层）——**口径**：HF `layers_block_type`=88 主干 / HF safetensors=90 命名空间 / **GGUF `block_count`=89（权威）**（`conversion/nemotron.py:256-261`、`nemotron-h-moe.cpp:13,18`）| `C` |
| W5 | §3 缺口#2 / §7 | 交错等周期性「**待定**」 | **非等周期**：attention 位置 `[7,16,25,36,47,58,69,78]`、间隔 `[9,9,11,11,11,11,9]`；**须显式层型表** | `C` |
| W2 | §2.2 第 53 行 + §2.4 路线 B | `arch_defaults` = "把编译期几何泛化出来的那一层"；走 B"省掉泛化几何一大块" | ⛔ **定性错**：`arch_defaults` 是 **GPU 架构（gfx1151/gfx1200）环境开关默认表**（入参 `gcnArchName`）⇒ §2.4 该收益**不成立**；B 真实收益 = **逐层数据驱动的层型机制** | `A(码)` |
| W6 | §2.2 关键词 + §4 | "`mamba`/`ssm`/`state_space` = 0 ⇒ Strata **从未碰过 Mamba**" | 字符串**确实 = 0**，但 **`SSM` = 386 且是真实张量加载**（`src/core/layer.cpp:229` 载 `ssm_conv1d`/`ssm_dt`/`ssm_a`）⇒ 口径改为「**Mamba 参数族已落地（以 GDN 名义），只缺 selective scan 本体**」| `A(码)` |
| W7 | §2.4 路线 A / §3 缺口#2 | 非等周期 ⇒ A"**需改成**显式层型表" | **更硬**：A 须**改层模型**（`layout.hpp:26-52,:63-64`）；`check_all` 12-QSA 断言**有定义无调用 = 死代码 ⇒ 静默误跑**（**非**硬失败）| `A` |
| W8 | §2.1（A 与现役二进制同版）| 走 B 无版本代价 | ⚠️ 选 B = 上游 **1.43.1→1.44.1**，**与现役二进制脱同版**（最高级代价）| `A` |
| W10 | §4（PR #2321）| "现成可借的几何泛化件" | **已在 B、A 没有**：A `native_router.cu` **写死 512/10**；B(1.44) `route_topk<NE,K>` ⇒ Nemotron **512/top-22**：**A 重写核 / B 加一行 `route_topk<512,22>`** | `A` |
| W11 | §0（专家驻留=资源）| 视作 A 的优势 | B(1.44) 驻留件**多于** A（按需 pin/页对齐专家布局/O_DIRECT/Foresight swap）；⚠️ 全 opt-in 且部分 **Linux only** ⇒ Windows 打折 | `A` |
| W9 | §8 出处 | — | 追加指针：P0/P1/U2/U3/U4/U4b/U14/P2 主件 + `台账-第三号专武-20261016.md` + `基座裁决材料-20261016.md` | — |

### 10.3 关键定案读数（指针）

- **P1**：新层型挂点 = `src/poly/engine.cpp:821`（架构级）+ `layer_head()` 逐层 if/else；`src/poly/` **独立实现**（非 fork 改名）；负控 LFM2.5 短卷积**命中**。
- **U2（判定正）**：poly 专家体系对 Nemotron 单机形态**够用**：**无 41 层上限**（产线 GLM=42 MoE 层）；**miss 不走 PCIe**（ggml-cpu 从 mmap 算）；**top-22 ≤ `kMaxUsed=32`**；32GB 驻留 ~20–30%；**单机每档都有对应物，唯一不等价 = 跨机 peer（本机无关）**。
- **U3（放行但未测 —— 措辞 = "没测过"，非"已验证"）**：能建（`src/poly/` 仅 **2–3 处**良性 `_WIN32`；setup 直接发 Windows 预编译包 `strata-poly-windows-x64-*.zip`）；无硬门（唯一 "does not run" = **AMD+HIP**，本机 NVIDIA 无关；Windows 闸门 = **警告 + `--yes`**）；**NVFP4 按 setup 公式 fits**。⚠️ **5 条 Windows 特有折扣**：① `open_prompt_direct` Linux only ② arena `mmap/madvise` Linux only ③ direct-IO 线程 Windows 4 / Linux 16 ④ aux CPU 绑定 Linux only ⑤ ★ **显存外溢吃 RAM**（卡不够的挪进 RAM，文件可用的 RAM 更少）—— 对本形态最要紧。⚠️ **2 条本机风险**：**WDDM 显存超额掉速 100→14 tok/s**（性能悬崖非崩溃，有缓解）+ **RTX 4080S 驱动 wedge #961**（本机同型号，driver 616.56 未验，唯一真·硬件风险）。**注**：poly 的 Windows 模型 = **5 个小模型**，**GLM/DeepSeek（最像 Nemotron 的两 MoE 先例）无 Windows 一方验证**（R19）。
  - **加固"不是跑不了"**：poly 有 Windows 现成二进制 · **大 GGUF mmap 有 Windows 实现**（`gguf_reader.hpp:403-408` `CreateFileMapping`/`MapViewOfFile`）· CPU 专家走可移植 ggml-cpu · Windows 关键链全有（二进制/mmap/CPU 专家/专家池/file tier + 32GB 低内存常驻 #467/#1984）。
  - **Linux-only 项（`open_prompt_direct`/THP/peer）全在 `src/core`（上游 A），不在 `src/poly`** ⇒ poly 引擎不依赖，对 Nemotron 单卡不关键。
  - **WDDM 掉速缓解**：`--vram-reserve-mib` / `--cpu-share`。
  - **建议措辞**："Windows 可用；大体量 MoE 流式属一方未验（社区有 GLM 单卡报告）；本机 4080S 有已知驱动 wedge 风险（#961），非引擎问题" —— **不要写成"Windows 不支持"**。
  - 两份同闸门件：`U3-windows-可用性-20261016.md`（p1-scout）+ `U3-Windows-32GB专项-20261016.md`（p0-scout）。
- **P2（判据绿）**：一层 Mamba-2 cell 对 float64 参考 **worst 2.42e-6（容差 3e-4）全绿**；负控全红（含 **3 例几何硬抛**）；**新写 49 < 搬迁 116**；`d_state=128` 在 `MAX_D_STATE=512` 检查界内。
- **U4/U14**：B 上游 ≈ **真·上游 v0.1.44.1 原件**（主体 **14 核心目录差异全 0**、**10/10 点名文件 SAME**）；例外 = **7 个共享内核/头纯加法补丁 +408/−2**（未碰专家驻留）。
- **P3 前置侦察（`A(码)`，含头部 ranged 实测）**：**容器层直读「能」**（poly 用共享 `gguf_reader.hpp`；张量名 **`blk.<il>.<suffix>`**；quant 面 = ggml-cpu）。**实测**：`general.architecture = nemotron_h_moe`、张量 **781 = `split.tensors.count` ✓**；`ffn_up_exps` [1024,2688,512] = **Q4_K(12)**、**`ffn_down_exps` [2688,1024,512] = `IQ4_NL`(20)（非 K-quant）** ⇒ **「Q4_K_M ≠ 每张量都是 K-quant」成立**（行宽整除成立、IQ4_NL 两路都有）；`ffn_gate_inp.weight` [4096,512] F32 = poly 期望。
- **接线层「要改 4 处」**（原记 5 处，**撤回 bias 那条**）：① 非 gated（无 `ffn_gate_exps`/`ffn_gate_up_exps`）② 宽度用 `n_embd_`(4096) 而专家是 **latent 1024** ③ shared 无 gate（`ffn_gate_shexp` 缺失）④ **`ffn_latent_up/down` 无概念**。⇒ **不需自定义专家帧/转换工具**，需**泛化 `load_moe`** 或**新架构类自装载**。
- **★ 层型表（正面发现 + 一处更正）**：GGUF 自带 `attention.head_count_kv` **89 项数组**（非零在层 **7,16,25,36,47,58,69,78,88**、非等周期）。⚠️ **但该数组的 0 同时是 mamba 层与纯 MoE 层**（llama.cpp 判型 = `kv==0 && n_ff==0`）⇒ **只能分「注意力/非注意力」，分不出 mamba vs MoE**。**实改**：按**张量在场**建表（poly 既有机制，`qwen35.cpp:86` 同法）——`ssm_in`⇒mamba · `ffn_gate_inp`⇒MoE · `attn_q`⇒Attention · 皆无⇒**硬抛**；`head_count_kv` 作**交叉校验**。（W15）
- **⚠️ 命名口径（两条同族）**：① P0 的 `…layers.N.mixer.*` 是 **HF/safetensors 名**，GGUF 是 **`blk.N.*`**；② **llama.cpp 源码的 C++ 成员名 ≠ GGUF 张量名**（`ffn_exp_probs_b` 落盘实为 **`exp_probs_b.bias`**）⇒ **判命名必须看 GGUF 目录**。
- **层数口径（已定案）**：GGUF **`block_count = 89` 权威**（`blk.0–87`=主干 88；**`blk.88`=1 个折叠 MTP 块**，内含 attention+MoE 两子层）——与 P0 的 88 是口径差，两边都对（见 W4/R26）。

### 10.4 未验（不许当结论；完整见台账 §5）

U17 / U18 / U19 / U20 / U21 / U22 / U23 / U24 / U25 / U26（latent 执行路径参数化 · top-22/512 实测吞吐 · U14 的 98 个 tools/docs 差异未逐行 · "上游 main=1.44.1 tag"是前提假设 · fp16+SR 无偏性 · P2 参考为 fp64 转写+合成权重 · `vram_beside_gb` 估计值 · 校准机 120GB vs 本机 95.8GB 余量或偏乐观 · NVFP4 本机实测吞吐 · Windows 外溢代价未量化）。

### 10.5 独立验收（task #6, verifier）与补漏

- **总判**：**P0 = PARTIAL** ｜ **P1 = PASS** ｜ **P2 = PASS** ｜ **U4 = PASS** ｜ router 双核 = 确认。
- **P2 PASS 亮点**：verifier **自己清编重跑**（`build_run.cmd`→BUILD_OK→EXIT=0）读数逐字一致；**转写对回 `E:\llama-cpp\prism-b10687` 六点全对**（conv channel-major / split 在 conv+SiLU 后 / D 在 swiglu 前 / norm 宽 = d_inner/n_group / softplus 与 `ssm-scan.cu:383` 逐字同 / scan 公式）；负控真红 100×tol；`grep clamp/fmin/fmax` 零命中。
- **P1 PASS**：file:line 亲验（**小偏**：`Engine::open` 签名在 **811** 不是 814；**挂点 :821 无误**）。
- **P0 PARTIAL 两缺陷（§5-P0 判据）**：
  1. **判据「只绿不红」**：改 `num_experts 512→511`，脚本 **exit 0、静默打出 63.158 GiB** ⇒ 项目书 §5-P0「闭合判据**能变红**」**在脚本层面不成立**；真·可红的是 **Qwen 交叉复现**（已过）。**建议补 assert**。（证据：`P0-20261016\_mut.json`）
  2. **分级错**：`ngroups=8` 是 **config 直读**（`out_nemotron.txt:28`）⇒ 应为 **`C` 不是 `D`/反解**；P2「ngroups 未直读」随之过时。
- **router 双核（走 B 的实际收益点）**：Nemotron **512/22 只能走 poly 的 `route_rows`**；若走上游 `native_router`（A 写死 512/10）**会落空** ⇒ 见 §10.2 W10。
- **U3 交叉补充（两件独立复现 + 本机实测）**：
  - ① **Windows commit limit 静默死（#1607）而 poly 未接守卫**（`host_limits.hpp:4-10`；唯一调用点 `src/program/generate.cpp:7273` 在上游主程序；`strata-poly` 链接集不含 `src/program` ⇒ `grep = 0`）—— **不翻盘**；**本机实测后下调为「低概率、条件性、可缓解」**：GGUF 走 `CreateFileMappingA(PAGE_READONLY)` **只读映射不计 commit**，本机 Commit limit **115.8 GB**、可用 **82.2–82.6 GB**、页文件（手动 20 GB）**峰值仅 3.5 GB**。"页文件调到 RAM 大小"从**前置动作**降为**可选保险**。
  - ② **`ram_tight_` 阈值（≈79.3 GB）正好卡在 Nemotron NVFP4 上**，随环境抖动翻转 ⇒ 同机同日可能不同驻留布局。**本机实测坐实且偏紧**：物理空闲 **78.9–79.2 GB**，恰卡阈值 79.3 ⇒ 翻"紧"侧取保守 reserve。**物理 RAM 余量两界**：poly 公式 `ram_need = arena − held + 10`（`setup.py:5805-5808`，`poly_beside_gb` 每档可自定 `:5774-5776`），**NVFP4 代入 P0 实测非专家 9.3 GiB（下界；字段含 KV/缓冲，真实近 11–12）⇒ `ram_need` 区间 51–53 GB**（⚠️ 勿与 BF16 的 16.6 分项混用）⇒ 余量 ≈**27–29 GB（poly 判据）/ ~75 GB（整文件保守）**。
  - ⇒ **风险排序**：**① 物理 RAM 余量（紧）＞ ② WDDM 外溢（中，与①耦合）＞ ③ commit limit（低）**。
  - **★ GLM 对比（`D`，勿过度概括）**：**GLM 有三档**（`setup.py:288-305`，与 `GLM.md:66-71` 逐档一致）：`3.0BIT`(113.6GB/**96GB RAM**)、`UD-IQ2_XXS`(101.8/82)、`REAP50`(78.8/60)。**fit 口径 = `ram_need − 4`，`ram = ram_gb() = ullTotalPhys`（总量）**（`setup.py:854-856`+`:5811-5813`）⇒ **按总量 95.8：3.0BIT 阈值 92 ≤ 95.8 ⇒ 三档全判"装得下"**；且 **`poly_default()`（`:5796-5800`）默认选 `3.0BIT`**（余量仅 3.8 GB、官方只注"在 120 GB 跑过"⇒ **实操存疑**）；**按空闲 79.2：只剩 UD-IQ2_XXS/REAP50**。**Nemotron NVFP4（ram_need 51–53 GB）比 GLM 最小档 REAP50(60) 还省 ~7–9 GB**。**⇒ 不可写"GLM 本机装不下"，也不可写"3.0BIT 装不下"。**
  - **⚠️ 记账陷阱**：`ram_gb` 的**卡口径因家族块而异、不可跨块混用**——GLM/DeepSeek = **32 GB 卡**（`setup.py:284-285`,`:486`，本机正是 32 GB 卡 ⇒ 96 直接适用），小 poly 家族 = **8 GB 卡**（`setup.py:508`）；跨家族比 RAM 前先看该块注释，**Nemotron 一律按 32 GB 卡口径算**。`vram_beside_gb` 逐档不同（GLM 12.5/11），对 Nemotron 用 P0 实测 **9.3**。
  - **P4/P5 可执行判据**：起跑同时记录 `usable physical` 与 `CommitAvail`；物理空闲 < 80 GB 就按实测重算驻留预算；`CommitAvail` 跌破 ~30 GB 才是 commit 侧警戒线（现 82.2–82.6）。
- **★ 命名对照（防错账）**：§0/§2.4 点名的 `--expert-cache*` / `--pcie-frac` / `--peer-device` **是上游引擎的开关，不是 poly 的**；poly 侧 = `--expert-profile`/`--expert-swaps`/`--vram-reserve-mib`/`--no-gpu-experts`/`--cpu-share`/`--tail-skip`/`stage_*`（平台中立）。**走 poly 须按 poly 的名字记账。**
- **P3 MoE 第 1 轮（relu2）— 独立验收 PASS（组件级）**：`relu2 = max(x,0)²` 对回 llama.cpp `LLM_FFN_RELU_SQR`；正控 1.38e-05 < 1e-4、负控 gated 1.38 真红、latent 错宽硬抛；形状与 GGUF 一致。**2 caveat**：① latent 错宽 `validate()` 是**测试本地函数**、未进生产 `load_layer`；② 负控①是两激活公式之差（非注入坏 kernel）。**边界**：**组件级通过；MoE 生产通路未接，Nemotron 整体仍跑不起来**。下一轮：少 token 路径（`moe()`+`cpu_experts`+slot）。
- **P3 MoE 第 2 轮 — 独立验收 PARTIAL（真缺陷，已退回修）**：**`moe_out_` 从未清零**（latent 路累加器是新缓冲、`add_scaled`/`axpy` 为 RMW；gated 路因 `moe_shared` 先覆盖 `fo_` 而侥幸正确）⇒ **40 个 MoE 层输出污染风险**；**已退回 cell-probe 修（`cudaMemsetAsync` 清 `moe_out_`），修完才准开 e2e**。其余 PASS（四目标清编 / `kMoe` 不熔断+几何硬抛 / gated 不回归 / `moe_latent` 语义 / 非 gated 禁批）✓。**缺口**：通路级运行负控待真权重；TX/NEW 施工后补（`TX=92/NEW=58`，ratio 1.59）。
- **★ P3 e2e 首次跑通（2026-10-16）**：`run_e2e.log` **EXIT=0**（裸参数冒烟）——启动自证 **`nemotron_h_moe 88 layers: 40 mamba, 8 attention, 40 moe (latent 1024)` 与 P0 几何逐字吻合** ✅；加载 10.6 s、**8361/20992 专家进 VRAM(24.1 GiB)**、40.6% from VRAM、CPU **95.9 ms/tok**（未优化）；运行时 U3 三数 gpu free 31267 / usable 95.8 / **CommitAvail 33.2**（vs 空闲 82.2–82.6 ⇒ 印证"起跑记录 CommitAvail"，且逼近 ~30 GB 警戒线）。**跑通路上连修 4 个加载缺陷**：① `moe_out_` 未清零 ② `layer_norm_rms_epsilon`→GGUF 实为 `layer_norm_epsilon`（`meta_alias`）③ `rope.freq_base` 缺省（宽容读）④ ★ `begin_pass()` 清层型计数器而 `load_layer` 不重加 ⇒ `n_mamba_=0` ⇒ state 悬空 ⇒ **illegal memory access**。**下一步**：正常 prompt 对拍 llama.cpp + 优化第二跑 + 负控。
- **夜间巡检（2026-10-16，见台账 §10）**：llama.cpp **`#30520` 昨夜合并**（W16/R30）；GGUF 供给侧扩容（unsloth/ggml-org/natalie5）但**我方靶 `lmstudio-community` 未漂**（sha `716398014e94` 与实测一致 ✅）；PolyStrata 昨夜 30 commit（**施工副本落后 20** ⇒ **已决不 advance**，U48）；上游无新 tag；**定位仍空**（社区在 128GB+ 统一内存机，无 32GB 消费卡）。**网络口径**：**GitHub 直连 / HF 走 `hf-mirror.com`**（代理 7897 已停，R32）。
- **★ 输出乱码根因 + 修复（2026-10-16，两处）**：**① `moe_router()` 从未被调用**（各架构在 `layer_head` 调它，唯 `NemotronHMoe` 的 kMoe 分支漏了 ⇒ 专家读陈旧缓冲；补 `moe_router(L,n)`）；**② `head_norm_rope` 强加了 RoPE + Q/K RMS 归一，而 Nemotron 两者都不做**（`ROPE_TYPE_NONE`、GGUF 无 `attn_q_norm/k_norm`；修为 `attn()` 不调它）。**读数**（同 prompt）：top-1 修前 `11`(EOS) **0.1148** → 只修① `1032` **0.7076** → ①+② **0.7996**；速度 **13.7 tok/s**。⚠️ **`1,2,3` 是垃圾输入，续写无意义**；**真正判据 = `The capital of France is` 首 token 对拍 llama.cpp**（进行中）。**方法论 #4**：**层型/张量名/几何全对 ≠ 前向对**，还要逐项对照 llama.cpp 的 **rope_type / qk-norm / 激活 / 残差**。**fresh build `7b133f9b…` 续测**：真 prompt 不再 EOS，生成 `,    and, `（top-1 0.865/0.824/0.887），**期望 ` Paris`(6993) 未出**；**⛔ 新 blocker**：非 gated 路径 **>8 token 批提示直接失败**（`native MMVQ requires 1 <= ncols <= 8` EXIT=1；ChatML 模板 ~11 token ⇒ **模板化输入无法测**）⇒ **正确性收口被批路径挡住**（U50）。**方法论 #5**：**跑读数前先核 exe sha**（旧件读数已作废）。**G1-a 消融**：② attention **已排除**；**① MoE 原判"已排除"作废**（消融太窄：只关 routed，未关 latent_up/shared ⇒ **方法论 #7：消融必须覆盖全部子通路，只关一条=假排除**）⇒ 重做中。**G2-b 合看**：MTP 默认开但 blk.88 **半载**（缺 3 处接线）⇒ 可能真凶，**隔离臂 = `--no-draft --lookup 0`**（两者都要关）。**★ 另一条红线缺口（§8.17，task #25）**：**Mamba 状态投机回滚**——`Engine::keep()` 废稿时**无守卫**从 `snap_` 拷回，而 `NemotronHMoe::mamba()` **不写 `snap_`**（签名无 snap 形参）⇒ `snap_` 是**垃圾** ⇒ **有废稿即用垃圾覆盖 40 层 Mamba 状态**（`qwen35`/`glm5next`/`lfm2moe` 都传 snap，唯本架构缺）⇒ **#14 候选根因**；其余 6 项状态生命周期两侧一致。
- **KV 布局对照（task #26）— 7/7 一致（排除性负结果）**：stride/行宽（行宽 256、512 B/位置）· GQA（`per=16`）· 因果/滑窗 · `kq_scale`(`1/sqrt(128)`) · **RoPE（都不做）** · Q/K 无额外 norm/scale/转置 —— **KV 路径不是乱码根因**。追加：张量 781 全打、`attn_output.bias`/`attn_q_norm/k_norm`/`_s` 族 **全 0** ⇒ poly 条件加载**全是 no-op、无漏读**。**两条非判定项**：① **prompt 路 Q 转 fp16 / decode 路 fp32** ⇒ `G1-b` 对拍**须分两条路**（U52）② `row=pos` vs 参考 slot 分配器仅"新上下文+连续+不复用"下等价（U53）。**已写卡 `KP-UNI-0243`**（定位布局映射必须钉死"实际执行的 kernel"）。**★ 硬结论（纯 Python 手算对照 `--dump-layers` 逐值）**：**前端 + layer0 + 残差 全部正确**（attn_in 6 位小数完全相同；mamba 全链差 ~0.5%；blk1 attn_in 差 0.007%）⇒ 排除 GGUF 读取 / **Q4_K 反量化** / **layer-0 Mamba-2 全链**（含 `z|xBC|dt` 切分、conv 布局、scan 的 A/D/dt_bias、gated_norm）/ 残差 / 首层 attn_norm。脚本 `E:\tmp-p3\mamba0_check.py`（verifier 硬验收所需已具备）。
  - ⚠️ **同日更正**：**手搓反量化有 bug**（"置换检验"抓到）⇒ 据它的「shared≈0」**作废**；**方法切换**为 **llama.cpp 官方 `llama-eval-callback`**（逐节点转储，CPU-only）；手算仍证 **layer 0/1 及传播正确**（更正规读数：attn_in 1706.999/1706.998883 · mamba 54.562/54.630 · blk1 826.225/826.217 · blk2 1688.374/1685.701）。**`blk=1 ffn` 4.058 vs 2.956 未定性**（U54，等真值）。**方法论 #8：需要参考数值时先找官方现成工具**，手搓只在查不到时且**必须配能变红的控制实验**。**R33：钩子未编进 exe**（`strata-poly.exe` mtime 9:40:53 < 源 9:44:15）⇒ **"开关不生效"先比 exe 与源 mtime**。
- **引擎接线审计（task #28）— 四项均无接线缺口**：残差 / 归一化 / 主循环顺序 / **blk.88 不进 trunk**（`hi_=n_layer_=88`）✓。**★ 口径纠正**：`Engine::dbg()` 的 **`abs` = Σ|v|（L1 和），不是 max** ⇒ `attn_in=826.2` = **均值 0.20 正常** ⇒ **输入侧无发散；发散在 MoE 模块输出**（均值 **0.00099 → 18.02**）。**★ 现象（已回改，⛔ 原"异常只在 MoE 层"不成立）**：**模块输出/`attn_in` 的增益随深度增长**（0.005 → 22.7–33.1，**≈5000×**），**MoE 与 Mamba 层皆有**（层 60/66 为 mamba，增益 13.6/33.1 > 层 87 的 22.7），**非某一层型特有**；attention 层（层 7 增益 0.42）同深度下明显更低。**另一条独立异常（未结论）**：**`attn_in` 是 RMSNorm 输出，量级本应与残差尺度无关**，却**层间摆动 19.6↔2136**（层 6=2136、层 7=19.6、层 87=3247）⇒ **残差流在若干层被压到近零**（与增益增长是两个现象）。**"偏大/偏小"类说法无逐层真值 ⇒ 不入账。** `A(码)`/`A(测)`/`D`。
- **★★ G1-e（task #29）— 恒等式失效**已证** / 是否 bug**未证**（`A(测)`）**：88 层 `attn_norm.weight` 的 L1 **只差 30.5×**（347.4@blk3 / 10598.8@blk46 / median 3013.6），而 `L1(attn_in)` 摆 **230×** ⇒ **`L1(xn)≈0.8·L1(w)` 在中间层偏离**（median **0.221**）。**首个 `<0.2` = blk 7（0.055，第一个 attention 层）**；blk 0–6（0.80–0.98）· blk 7 起 0.055–0.22 · **blk 86/87 回升 0.54/0.57**；**阳性对照 `output_norm` = 1.016** ✅。**⛔ 不写"残差被压塌"/"方向锁死"**：展开式显示 ratio 小 = **离群主导/形态变了**，且 `RMS(xn) ≡ RMS(w)` 意味着**量级没塌**；**L1 近似可假阳性**、**massive activations 本身是已知真实现象** ⇒ **只证到「形态在 blk 7 突变」**，**是不是根因要等 #27 逐层对拍**（参考 `P3-20261016\evalcb-pos0.log` 已取到，`blk=0` 前 3 数 4 位小数吻合）。判据由 team-lead 出，本席只取证。
- **★ 正确生成模板必须带 `<think>`（§8.16，影响所有后续测试）**：GGUF `chat_template` 生成回合 = `<|im_start|>assistant\n<think>` ⇒ **正确 token 序列 16 个**（`…,12,1010`，`12`=`<think>`）；**此前测试均用 14-token 版（少 `12,1010`）⇒ 那些读数测的不是真实推理路径**（已加注，结论不废）。**口径：回归一律用 16-token 版**。**已知**：补 `<think>` 后**仍吐空格/逗号** ⇒ 模板修正不解决本 bug。
- **新嫌疑 `lm_head` 方向（待查）**：乱码下草稿接受率 **40%（6/15）** ⇒ **非线性噪声，是系统性的坏**；形态符合**"最后一步映射错、分布仍尖锐"**。已派查 4 项（`output.weight` 方向 / `output_norm` / `finish_logits` vocab 取法 / **转置负控**）。**逐层类型分类与 config 前 20 层逐项一致 ⇒ 分类无罪**。
- **U50 根因定位（G2-c）**：**`ncols=13` 来自 `engine.cpp:1873`**（latent 分支把 token 数 `n` 当 `ncols`）→ `native_quantize_q8_1:2334` → `:1772 throw ncols<=8`。**归属更正**：非 `q8_packed_mmvq:2006`（两条路**产出同一异常串**，须靠**调用点+可达性**判，**方法论 #6**）。防误改：`engine.cpp:1974` 类似但安全（`m ≤ kSlots=8`）。**KB 已回写卡 `KP-UNI-0242`**（520 张、lint 全绿）；**新缺陷 U51**：`native_mmvq.cu:1772` 异常串**硬写 `8`**（真上限 = 常量 `MAX_NCOLS`），违 `KP-UNI-0204`「报错文案不写死数字」。**旁证**：上游 Qwen 与 poly `lin()` **两条惯例都靠分块压 ncols≤8** ⇒ latent 分支**同时背离两者**。
- **KB 交叉核对（G2-b，先查库）**：`before` = 本库无「poly MTP 通路」直接知识 ⇒ **新知识、须回写卡**；**6 卡命中**：KP-MOEDEP-0005（乱码先查闸门 ⇒ `#14` 第一步改 **Mamba-2 闸门激活对表**）/ 0007（`draft_` 使专家块 40→41，指纹负控）/ KP-STRATA-0016（**判据禁用接受率**，改"每验证 token 数+墙钟"）/ 0024（以引擎打印为准）/ 0028（质量走 CLI，MTP=速度实验）/ **0006（乱码未修前 decode 读数不可引用）**。
- **MTP 通路判定（G2-b）**：机制齐、**接线缺 3 处**（① 无 `block_attend` ② `kind_of(88)=kMoe` ⇒ blk.88 attention 权重静默丢 ③ `L.ffn_norm` 从不设）；**内置默认开、复用主 `lm_head`**（与论文 §5.2 一致）；**`--no-draft` 可作 #14 隔离臂**（一跑即判乱码/EOS 是否源于 MTP 半载）。更正：`nextn.*` 已由基类 `load_drafter` 载。**缺口已由 Qwen3.8-Flash 全树扫独立复核**（单文件 + 全树**双路一致**）。`C`/`A(码)`。
- P4/P5 **建议起跑动作**：① 先查本机 **Windows 页文件配置**（决定 commit-limit 风险是现实还是理论）；② 评估给 poly 补 `commit_verdict` 调用（上游有现成实现，属搬件）；③ **e2e 前逐条过「淘金清单」**（台账 §9：激活逐处对表 · 首 token 逐字节对拍 · Norm 用 GGUF 实值 · 先正确后快 · profile 记 sha · 勿自改量化；KP-MOEDEP-0005 / KP-KVMEM-0005 / KP-NINFER-0089 / KP-UNI-0230 等）。
- **P3 施工（切法 A）— 独立验收 PASS**：`P3-20261016\READING-P3-status.md` —— `kernels.cu +146`（mamba2 三内核自 P2 搬迁）· `nemotronhmoe.{hpp,cpp} 85+314` · `engine.cpp` 挂点 `nemotron_h_moe` · `CMakeLists +2/−2`；**三目标清编 BUILD_OK（C++20 实证，非 C++17）**。**verifier 总判 PASS**：层型表更正成立且必要（`nemotron-h.cpp:19-20`）、张量名对回 `llama-arch.cpp` 逐条一致、三内核公式与 P2 正本同、**MoE 熔断=硬抛非静默**（`load_layer:147`/`layer_head:294`）。**caveat（不阻断）**：① **挂点 `821`→实为 `822`**（R29）；② P3 本地无 TX/NEW 计数、无可跑负控（红能力靠 P2 继承，未就地复证）。**下一步**：**MoE 通路泛化**（跨文件 150–250 行）。**编译标准口径**：跟**工作树**走 —— B = C++20；§2.3 的 A 工具链记录保留作历史（R28）。

### 10.6 未验（追加；不许当结论）

U6 已解（ngroups=8 为 `C`）· **U31 已结（本机页文件/commit 已实测）** · U17/U18/U19/U20/U21/U22/U23/U24/U25/U26 · **U27**（WDDM 掉速）· **U28**（4080S 驱动 wedge #961）· **U29**（poly 无 commit 守卫，**已下调**）· **U30**（`ram_tight_` 抖动，**实测坐实偏紧**）· **U32**（补 `commit_verdict` 工作量未估）· **U33**（P0 脚本无 assert）· **U34 已结** · **U35 已结**（容器层直读能 / 接线 4 处）· **U36**（GSQ-RCO，挂账）· **U37 已结**（`ffn_down_exps` = **IQ4_NL**，非 K-quant）· **U38 已对账** · **U39 ◐部分结**（`ffn_gate_inp` [4096,512] ✓；`ffn_latent_up/down` 语义仍未验）· **U40**（`nextn.*` 或已支持 ⇒ MTP 可能免改）· **U41 已结**（口径差：GGUF `blk.0–87`=主干 88、**`blk.88`=1 折叠 MTP 块** ⇒ GGUF `block_count=89` 权威；MTP 口径见 W4）· **U42 已解**（Q4_K_M 权重已下载并 **sha256=HF LFS oid 逐字同**（KP-UNI-0205）⇒ 可开端到端；路径 `E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\`）· **U43**（层张量名未与 GGUF 目录逐条对账）· **U44**（注意力 qk-norm 未核）· **U45**（`unsloth`/`natalie5` 37 件未展开 ⇒ 备胎 `UD-IQ3_XXS` 存在性未证）· **U46**（Super VL arch 字符串未核）· **U47**（llama.cpp `#30610` 根因）· **U48 已决**（**不 advance，条件触发**：正确性基线通过前不换基座；除非 20 commit 里出现 Nemotron/MoE/Mamba/ggml ABI 相关改动或需 `--kv q8_0`）· **U49**（`#2321` 为何未合并）· **U50**（⛔ 非 gated 路径 >8 token 批提示失败，`MMVQ ncols<=8` ⇒ **挡住正确性收口**；**已派 cell-probe**，**先 (a) 后 (b)**：(a) `moe()` 内按 ≤8 token 分块累加（20–40 行，立刻解开 ⇒ 正确性收口）；(b) `moe_rows()` MMQ 批路径（~300 行，归性能轮）。判据 = 模板化 prompt 首位出 ` Paris`(6993)）· **U51**（报错文案硬写 `8`，建议由 `MAX_NCOLS` 拼出）。**保持未验**：HF 231.84 外部读数 · fp16+SR 无偏 · 真实权重 · P1 §7 八条 · poly 在 Windows 跑 60 GiB 实测（属 P4/P5）。完整见台账 §5。

### 10.7 官方技术报告的定位（新证据源，`C` 厂商自报）

**件**：《NVIDIA Nemotron 3.5 Super and Lightning Technical Report》（79 页，本地 `E:\tmp-p3\nemotron_report.pdf`）。**等级 `C`（厂商自报）—— 定"设计意图"，定不了"我们实现对不对"**（后者以 llama.cpp 源码 + GGUF 实测为准）。要点见**台账 §11**，摘：
- **架构**：Latent MoE（`latent_down → 专家 → latent_up`），**router/gate FP32**；每 MoE 层 512 routed + shared；层 = *"interleaved Mamba-2 / MoE / attention"*（**与本件 §10 的 88 层三分吻合**）。
- **注意力（§5.2 原文）**：*"fewer attention layers, with narrower key-value heads … Mamba-2 layers whose state does not grow with context"* ⇒ 论文**未写 NoPE**，那是源码实证（§8.11 根因②）。
- **推理**：**MTP 共享单个 `lm_head`**（*"a single lm_head, shared between target verification and MTP drafting"*）⇒ 佐证 `blk.88` 不该有独立输出投影；*"a large per-request cost … is updating the Mamba state"*；SSM cache 量化 + **stochastic rounding**（**与 P2 的 fp16+SR 一致**）。
- **硬件贴合**：验证含 **RTX-class**，但公开读数仅 H100/GB200；NVFP4 = **Blackwell W4A4 / 非-Blackwell W4A16** ⇒ **本机 Ada 是降级档**。⇒ **架构层红利拿得到；量化/kernel 层拿不到**（我们走社区 **Q4_K_M**，已绕开官方 NVFP4）。

---

*§10 为 2026-10-16 追加回写；上文 §0–§9 原结论保留不改。*
