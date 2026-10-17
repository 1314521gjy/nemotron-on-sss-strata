# 同门生态调研：GLM-5.3 / DeepSeek V4 Flash 在 Strata 上的适配 — 2026-10-16

- 席位：`scheduler` · **只读、零 GPU**；**未用 GitHub API**（读的是**我们自己 clone 的 `origin/main`**：`PolyStrata/`，`git show origin/main:<path>`，**无写操作、无 fetch/merge/rebase**）。
- 触发：外部给的同门实测数据（GLM-5.3-Flash 321B-A18B on **Strata Engine / Project Maya**，44GB 双卡）⇒ 要"同门还有什么内容可参照"。
- **证据级**：**C** = 上游仓库原文（`origin/main`，sha `51dc242`，2026-10-15）；**A** = 我们本机实读。
- **查库**：本件为新题（同门前作 `P3-20261016\上游适配侦查-ninfer与strata-20261016.md` 只覆盖 **提交标题级**，本件是**内容级**）。

---

## §0 结论速览（含**两条旗标归属更正**）

1. **★★ 更正①：同门口中的"VL 模型纯文本跑要加 `--no-vision`"——上游的形态是 `setup` 的 `--vision no`，不是引擎旗标。** 原文：`./setup.sh --setup --family qwen --model IQ3_XXS --gguf-dir <folder> --gpus 0,1,2,3 --cuda 12 --vision no --yes`（`bench/results/2026-10-06-community-4x-tesla-p100/README.md:40`）。⇒ **我们引擎本来就没有 vision 通路**（另件已回执），**这条对我们不适用**。
2. **★ 更正②：`--gguf-dir` 也是 `setup`/`START-HERE.bat` 的**（`docs/BACKENDS.md:86`、`bench/.../prompt-32k.txt:408`），**不是 `strata-poly` 的旗标**。⇒ **又一条"别人的旗标不能照搬"**（`KP-UNI-0242`/`KP-NINFER-0005` 同族）。
3. **★★ 最值钱的一条：上游把"非专家固定开销"和"OS 文件缓存"写成了**硬性要求**：GLM **~12 GB**、DeepSeek **~11.5 GB**「go to the weights, the caches and the buffers **before any expert**」，**setup 低于 14 / 13.5 GB 直接拒**；并且**离线专家必须留在 OS file cache**（否则从盘读、慢一半）。⇒ **我们从没查过这条腿**（§3-A）。
4. **★ 上游公开了它的 benchmark 口径（六列）**，与我们的读数**可逐列对齐**（§4）。
5. **★ `--tail-skip 6` 是上游 `setup` 给 Qwen3.6 的**安装档**（不是我们发明的）—— 我们 P1-b 要"对齐 NVIDIA 减激活"，**其实上游自己就在用**（§3-D）。

---

## §1 同门是谁、在哪个仓（C）

- **一个仓两个引擎**：`github.com/VecSzn/PolyStrata`。**`strata`** = 老引擎，跑 **Qwen3.8-Flash-Next**；**`strata-poly`** = 第二个引擎，跑 **GLM 5.3 Flash**（`docs/GLM.md:1-5`），**同一个 `strata-poly` 也跑 DeepSeek V4 Flash**（`docs/DEEPSEEK.md:1-3`，llama.cpp arch `deepseek4`，读 unsloth 的 GGUF）。
- **适配件在仓里的位置（可执行引用）**：
  | 件 | 路径 | 是什么 |
  |---|---|---|
  | **专家 profile** | `data/expert-profile-glm.txt` · `data/expert-profile-glm-reap50.txt` | ★ **仓里直接附带 GLM 的 profile**（"the expert profile the repository ships"） |
  | **草稿块取件** | `tools/glm_draft_fetch.py` + `tools/test_glm_draft_fetch.py` | GLM 的 draft/MTP 文件获取器 |
  | **文档** | `docs/GLM.md` · `docs/DEEPSEEK.md` | 各自的实测表 + 需求 + 安装 |
  | **服务测试** | `serve/test_glm.py` | —— |
  | 引擎源 | `src/poly/glm5next.cpp` · `src/poly/deepseek4.cpp` | 两个架构实现 |
- **★ "Project Maya"：本仓 grep 无命中**（`git ls-tree -r origin/main | grep -i maya` ⇒ **空**）⇒ **当前 `origin/main` 里没有以 maya 命名的目录/文件** ⇒ 它要么是**上游对外的产品名**、要么在**另一仓**。**负面结论（未找到）**。
- **其他已适配的模型（从提交标题看）**：MiniCPM-V 4.7（`107f1c5`）、LFM2.5（`eb5ef11`）、Tiel Coder（`3d038c4` 附近）⇒ `strata-poly` 已是**多架构引擎**。

---

## §2 ★★ 两条旗标归属更正（避免误搬）

| 同门口中的旗标 | 实际归属 | 原文 |
|---|---|---|
| **`--no-vision`** | **`setup` 的 `--vision no`**（安装期决定要不要装/启用视觉那条链路） | `./setup.sh … --vision no --yes`（`bench/results/2026-10-06-community-4x-tesla-p100/README.md:40`）；另 `--vision gpu`（`bench/results/2026-10-03-community-v100/README.md:154`） |
| **`--gguf-dir`** | **`setup` / `START-HERE.bat` 的**（"use GGUF files you already have"） | `docs/BACKENDS.md:86`；`bench/.../prompt-32k.txt:408`；`docker-entrypoint.sh:83`（读 `$GGUF_DIR`） |

⇒ **两条都不是 `strata-poly` 的运行时旗标**。**我们引擎的旗标面（42 个）已另件列全**，不得混引。

---

## §3 可参照清单（逐条：**能参照什么 · 与我们的差异 · 证据**）

### A. ★★ "离线专家必须留在 OS file cache" —— **本条最可搬，且我们从没查过**
- **上游原文**：「RAM for the experts the card does not hold. **The file is mapped and they are read from it while the model answers, so they should stay in the OS file cache**: about 86 GB of the 107 GB on a 32 GB card. Setup's estimate for that card is **96 GB** of RAM.」；容器里「the memory limit must leave about **8 GB above the model's files**: **with less, answers read experts from the disk**, and the container may end the engine」（`docs/GLM.md`；DeepSeek 同构：62/84 GB、估 72 GB、11.5 GB 固定）。
- **★ 机制证据（同一机制的实测）**：tip commit `51dc242`（2026-10-15）—— 「The **first benchmark after the PC started** wrote at **14 to 24** tokens/s and the next at **28 to 29**, because the **model file had to come from the disk again**」。
- **我们**：CPU 专家同样是 `exp_up/exp_down` 的 **host 指针（mmap 文件）** ⇒ **同一条腿**。**但我们从来没量过**：非驻留专家是**从页缓存来**还是**从盘来**。
- **能参照什么**：**把"预热 1 跑丢弃"升级为"预热 + 确认文件已在页缓存"**；并在 log 头记 **RAM 总量 vs 模型文件大小**（上游把它写成了**硬性安装门**）。**零 GPU 就能先算这笔账**（本机 RAM ↔ 70 GB 模型）。
- **差异**：我们的模型 **70 GB（两片）**、非驻留专家约 **60%**（20480−8010）⇒ 需要留在页缓存的是**几十 GB** 量级 ⇒ **本机 RAM 是否够，是一次可查的账**。
- **级**：**C（上游原文 + 实测）+ A（我们的 mmap 事实）**。

### B. ★ "非专家固定开销"是硬门：**12 GB（GLM）/ 11.5 GB（DeepSeek）**
- 上游：「**About 12 GB go to the weights, the caches and the buffers before any expert**, so setup **stops and asks below 14 GB**」（`docs/GLM.md`；DeepSeek 11.5 / 13.5）。
- **我们**（t16/显存三方账）：非专家可动项 = KV 64 MiB + Mamba state 165 MiB + `snap_` 494 MiB + 草稿 16 MiB + logits 4 MiB ≈ **0.72 GiB**，另加**未读到的 dense 权重 ~6–7 GB**。
- **能参照什么**：**"先扣固定 baseline 再谈专家容量"这个口径**（与我们的 `reserve`/arena 关系同构）；**但具体数字不可比**（他们的 12 GB 里含的是 321B 模型的非专家权重与更大 cache）。
- **差异**：**数量级不同**（12 GB vs ~7 GB），且他们的目标是 DGX/32GB 档。
- **级**：C。

### C. ★ **profile 的收益口径（外部旁证）**
- 上游：「the engine's numbers are **with the expert profile it writes as it runs**; **the first answers after a start without one came at 42 to 43 tokens/s**」（`docs/DEEPSEEK.md`）⇒ **42–43 → 48–53（+12–23%）**；GLM.md 也给"an earlier build … without the draft block"对照。
- **能参照什么**：**与 `KP-MOEDEP-0008`（+31.7%）同族**，是"profile 值多少"的**第二个独立实例**（不同模型、同一机制）⇒ **强化我们"profile 是量级手段"的结论**。
- **差异**：他们的 profile **随跑自动写**（我们的 `--expert-profile-write` 默认关 —— 我们因正确性把它关了，**这是我们的差异而非他们的**）。
- **级**：C。

### D. ★ **`--tail-skip 6` 是上游 setup 的安装档**
- 上游原文（Qwen3.6 那行）：「wrote after the three prompts at 395 to 396 … **with `--tail-skip 6`, as setup installs it**」（`docs/BACKENDS.md`）。
- **能参照什么**：★ 我们 P1-b 的 `--tail-skip 6`（对齐 NVIDIA `Bolt-74B-A9B` 的减激活）**与上游的安装默认完全同值** ⇒ **两个独立来源指向同一档** ⇒ **这条假设的外部支持度提高**。
- **代价**：**改结果**（近似档）⇒ 仍须单独成档 + `--top-probs` 逐位判据。
- **级**：C。

### E. ★ **量化族决定 CPU/GPU 分工的优劣方向**
- 上游原文：「Its experts are **i-quants, which pass through the card faster and through the CPU more slowly than K-quants**: it reads a prompt faster than `3.0BIT` and writes more slowly」（`docs/GLM.md`，讲 `UD-IQ2_XXS`）。
- **我们**：`ffn_up_exps` **Q4_K**、`ffn_down_exps` **IQ4_NL**（**i-quant 系**）⇒ **专家是混精度** ⇒ **"过 CPU 更慢"的那一半正好在我们的 down 段**（而 down 是 CPU 专家的主计算）。
- **能参照什么**：**可测假设**：**把 down 换成 K-quant（若可行）是否让 CPU 路径变快**；或**验证现状**：我们的 CPU `ms/token` 里是否 down 段占比异常。
- **差异**：i-quant 的定义族一致，但**具体类型不同**（IQ2_XXS vs IQ4_NL）。
- **级**：C + D（假设）。

### F. ★ 长上下文：`b9aef4f`（把 cache rows 还给专家）**现在有实测上下文**
- 上游 tip commit：**`51dc242`** —— 一位用户（issue 10）用 **RTX 5090 D v2（24 GB）+ Windows**、GLM 3.0BIT，**读 1,002,452 token 用 45.7 分钟**（poly.5；poly.2 是 66.3 分钟 ⇒ **+45%**），三个植入事实都找到；并注「**where the engine gives the rows back**」（= `b9aef4f` 的那条机制）。
- **能参照什么**：那条我们侦查过的上游 commit（"请求开始时把超出的 attention cache rows 还给专家"）**在 1M 上下文里确实被吃到**，且量级是 **24 GB 卡**。
- **差异**：★ **我们 `--max-context 8192`（0.008M）** ⇒ 相差 **122×** ⇒ **该项目前与我们无关**（但**如果将来上长上下文，这就是那条要照搬的机制**）。
- **级**：C。

### G. 官方 benchmark **六列口径**（可直接对齐，见 §4）
- **级**：C。

---

## §4 上游公开的 benchmark 口径（**可逐列对齐**）

上游 `docs/BACKENDS.md`（"Benchmark" 节）的表头六列：
**`Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes, tokens/s | Experts from VRAM | Guesses right`**
并要求**同一模型跑三个 prompt 长度**（短 / ~2.5K / ~10K），且注明「**A short prompt is read a few tokens at a time, which is why its reading speed is low**」。

**⇒ 与我们对照**：
| 上游列 | 我们的对应读数 | 出处 |
|---|---|---|
| Prompt, tokens | `prompt N tokens` | `main.cpp:716-721` |
| First token, s | `r.prompt_ms`（+ 首 token 生成） | `main.cpp:660-668` |
| Reads, tokens/s | `prompt … tok/s` | `main.cpp:720` |
| Writes, tokens/s | `generated … tok/s` | `main.cpp:721` |
| **Experts from VRAM** | **同名**（`experts from VRAM %`） | `main.cpp:722` |
| Guesses right | `N of M guessed tokens right` | `main.cpp:724` |

★ **两条值得抄进我们口径的**：
1. **短 prompt 的读速天然低**（"read a few tokens at a time"）—— 与我们今晚读出的 **`--stage-ahead` 的门是"每次调用的 token 数"** 是**同一件事**（短调用走逐调用路径）⇒ **两个独立来源印证**；⇒ **报读速必须带 prompt 长度档**（我们已有"读数带生成长度"的 D5，**可扩展为"带 prompt 长度"**）。
2. **上游公开"Guesses right"** ⇒ 与我们 `KP-MOEDEP-0028/0029`（投机只做速度实验）**取向一致但更宽松**（他们把它进基线，我们把它排除）⇒ **差异写明**。

---

## §5 负面结论 / 与我们不可比之处

| # | 负面 | 依据 |
|---|---|---|
| N1 | **没有独立的 "Project Maya" 仓痕迹**（本仓 grep 无 `maya`） | `git ls-tree origin/main | grep -i maya` ⇒ 空（C） |
| N2 | **没有"面向 24–32 GB 消费卡的 offload/驻留调优文档"** —— 他们的**目标机是 RTX 5090 32 GB + 120 GB RAM**，且把"专家放 RAM 靠页缓存"当成**前提**而不是**优化对象** | `docs/GLM.md` / `docs/DEEPSEEK.md`「What it was measured on」（C） |
| N3 | **量化档、模型几何、上下文长度全都不同**（GLM 321B-A18B/288 专家/113.6 GB/32K；DS V4 Flash 90.9 GB/32K；我们 120B-A12B/512 专家/70 GB/8K）⇒ **任何**绝对 tok/s **不可比** | 同上 |
| N4 | **他们的窗口是 Linux（作者线）+ 一位用户在 Windows 24 GB（issue 10/21）** —— Windows 侧只有"外部报告"，**没有官方 Windows 配方** | `docs/GLM.md:6-7`、`docs/DEEPSEEK.md:7-8`（C） |
| N5 | 上游把 profile **自动回写**当默认；**我们为正确性把它默认关了** ⇒ 他们的"跑得越久越好"**我们拿不到**（除非受控学习窗口） | `docs/DEEPSEEK.md` + `KP-UNI-0244`（C/A） |

---

## §6 未验 / 边界
1. **【C】** 本件全部上游内容为 **C 级**（读的是**我们 clone 的 `origin/main`，sha `51dc242`**）⇒ **未对上游仓库做任何写操作**（`git show`/`ls-tree`/`grep` 只读）。
2. **【D】** §3-A 的"我们 RAM 是否够留页缓存"**未算**（是一次零 GPU 的账：本机 RAM vs 70 GB 模型 + 工作集）⇒ **建议排给 builder/verifier**。
3. **【D】** §3-E 的"down 段 IQ4_NL 在 CPU 上吃亏"是**假设**，未测。
4. **【C】** "Project Maya" 只在本仓**未找到**，**不等于它不存在**（可能在另一仓或指产品名）。
5. 本件**未改码、未编译、未跑 GPU**；**未碰 `E:\strata`**；未跑重 CPU/磁盘扫描。

---

# §7 ★ 追加（队长令）：专门找**投机 / MTP** 这一类（怎么做、怎么配、与专家驻留如何共存）

> 触发：外部点评指出「Strata 的重要加速来源之一是**投机解码/MTP**（"先猜，再批改"）」⇒ 队长要求在同门件里专门找这一类。

## 7.1 ★★ 最关键的一条：**GLM 上 MTP 的实测收益 ≈ 0**（C，但来自作者自己的文档）
- **原文**（`docs/GLM.md:209`）：「33.7 and 31.4 tokens/s **with its block** and **31.3, 30.9 and 31.4 without**. **`--no-draft` and `--lookup 0` turn the two off**」
  ⇒ **开/关投机块：33.7 / 31.4 vs 31.3 / 30.9 / 31.4** ⇒ **基本无差（甚至有的读数更低）**。
- ⇒ ★ **与本项目 `KP-MOEDEP-0007`（"MTP 已冻结：接受率 0.023→0.632 仍输 decode"）同族** —— **"接受率上升 ≠ 端到端变快"**；**我们全线 `--no-draft --lookup 0` 与这条外部证据一致**。
- **⇒ 对本靶的直接含义**：**别把 MTP 当成"现成的下一块肉"**；真要上，必须配"**同批交错 ON/OFF decode + 防'接受率升而 decode 降'**"的判据（我们 P2-a 已是这个设计）。

## 7.2 投机的**两种载体**（上游都支持，A 级读码 + C 级文档）
- **`docs/BACKENDS.md:16-22` 逐模型列了投机载体**：
  | 模型 | 载体 |
  |---|---|
  | GLM 5.3 Flash · Qwen3.6 · Ornith · Tiel Coder | **draft block in the file**（**模型文件内置**） |
  | Gemma 4 | **its draft model, a file of 0.46 GB**（**独立文件**） |
  | **DeepSeek V4 Flash** | ★ **"repeats only: its draft model is not run yet"** |
  | **LFM2.5-24B-A2B** | ★ **"repeats only: there is no draft model"** |
- **`docs/GLM.md:197`**：`--draft-model` 是"**当 `--model` 没有块时**"的来源；「**A right guess saves less than a whole**…」
- **`docs/DEEPSEEK.md:119`**：DeepSeek 的 draft 件叫 **`dspark`（11 GB）**。
- **⇒ ★ 与本引擎的对应关系**：**完全同构** —— 我们的 `--draft-model FILE` 的 help 原文就是「**when `--model` has none**」（备选来源）；**本靶的块是内置的**（`nextn_predict_layers=1`）⇒ **我们不需要外部 draft 文件** ✓（另件已证）。
- **上游源码**：`src/spec/` = `draft_policy.cpp` · `draft_source.cpp` · **`suffix_drafter.cpp`**（★ **= prompt-lookup 的实现**，对应我们的 `--lookup`）· `guesses.hpp`；工具 `tools/glm_draft_fetch.py`（**取 GLM 的 draft 块**）。

## 7.3 默认值（C，`docs/GLM.md:255-259`）
- **`--window N`**：**GLM 默认 4**（其他模型 8）；含义「the known one and guesses after it: **the draft block's three at most**, and more than three only from a repeat of eight tokens or more; 1 = no guesses」⇒ ★ **"草稿块最多 3 + 由 8-token 重复补更多"** = **两种猜测源混用**（ours: `--window` 默认 8）。
- **`--draft-vocab N`**：**Qwen3.6 / Ornith / Tiel Coder = 131072**（前 131072 个 token），**其他模型 = 全词表**（ours: 默认 `-1` ⇒ 按架构 `guess_vocab()`，**我们已有同机制**）。

## 7.4 ★ 负面结论（重要）
- **"温度"相关：`grep -i temperature` 在 `docs/GLM.md` / `docs/DEEPSEEK.md` / `docs/BACKENDS.md` ⇒ 零命中** ⇒ **此前的"温度 1.0⇒MTP 6% / 0.3+top_p0.9⇒70%"说法在仓内公开件里找不到**（可能在 **issue 10/21** 或别的渠道，**本席未查 issue**）。
- **"投机与专家驻留如何共存"**：上游文档**没有专门论述**；**可读到的唯一线索**是 `docs/GLM.md:171`（"`--expert-profile FILE` is where the engine keeps count of which ones are used"）与 `:197`（**A right guess saves less than a whole**）⇒ **投机与驻留的耦合在本仓未披露** ⇒ **标"未披露"**。
- ⇒ **★ 对本靶的意思**：**"投机与专家驻留的交互"这件事，公开面没有可抄的答案** ⇒ 我们要自己测（正好是我们 P2-a 的设计）。
