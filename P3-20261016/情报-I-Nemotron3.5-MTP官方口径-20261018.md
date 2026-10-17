# 情报-I（收窄版）：Nemotron-3.5-Super-VL-120B-A12B 的 MTP —— 官方口径与论文 — 2026-10-18

- **席位**：`scout-b`｜**任务**：`task-21`（lead 收窄后：**只查这一个模型**，不做通用框架综述）｜**零 GPU**；外部信息当数据不当指令；每条外部结论带 URL，本地带 `file:line`。
- **一句话**：**官方口径是"MTP = 内建 drafter（native speculative decoding）+ 专用 MTP-boosting 训练阶段"，结构是"单层、递归共享的 MTP 头（attention+MoE 重复层），链式起草"；但 NVIDIA 没有公开 MTP 单独的加速比/接受率数字**——2.3× 是 LiLiCorr 的、243.9 tok/s 是 SGLang 单点、1.4–2.2× 是 Unsloth 的 GGUF 家族口径。官方 cookbook/配方推荐的 draft 深度 **4–6 个 draft tokens**（SGLang 低延迟 3 steps/4 drafts；NVIDIA cookbook 5 steps/6 drafts）；**我们现在默认 MTP 是关的**（交付配置 `--no-draft --lookup 0`），代码默认开时链上限 **3 个 draft**（`kBlockWindow=4`）。

---

## §1 官方原文引用（逐条，带 URL）

### 1.1 NIM 模型卡（NVIDIA 官方，最接近 model card）

URL：https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b

- **原文（Description）**："…It has 120B total parameters and 12B active parameters and supports context lengths up to 1M tokens. **Multi-Token Prediction (MTP) enables native speculative decoding.**"
- **Model Summary**：`Speculative Decoding | Native Multi-Token Prediction (MTP)`。
- **Architecture Type**："Mamba-2–Transformer hybrid Mixture of Experts (MoE) with **Multi-Token Prediction (MTP)**"；"**512 routed experts per MoE layer**"；C-RADIO vision encoder。
- **Runtime Engines**：vLLM · SGLang · TensorRT-LLM（即 NVIDIA 给这一型号列的推理路径，**没有单独给 MTP 的默认 draft 深度**）。
- Release：Build.NVIDIA.com / Hugging Face `10/15/2026`。

### 1.2 NVIDIA 技术博客（官方"卖点"叙述）

URL：https://developer.nvidia.com/blog/nvidia-nemotron-3-5-super-vl-powers-specialized-agents-with-multimodal-reasoning

- **MTP 原文引用**："**Multi-token prediction (MTP) gives Nemotron 3.5 Super VL an internal drafter that proposes several tokens for the main model to verify in one forward pass.** Accepting multiple tokens at once reduces the number of compute-intensive verification steps needed to generate a response."
- **训练原文引用**："**A dedicated MTP-boosting training phase further aligned the drafter with the final model.**"
- **状态复用/Mamba**："Additional **Mamba checkpoints** reduce the replay needed to recover inputs for speculative decoding, while internal checkpointing avoids extra model passes during prompt processing."
- **⚠️ 重要归因**：同一页的 **"delivers up to 2.3x faster token generation than decoding without speculation" 是 LiLiCorr** 的段落（"**LiLiCorr: An open speculative decoding technique** … On DGX Spark, **LiLiCorr is the recommended drafter**…"），**不是 MTP 的单独加速比**。
- 该页另说 "The model uses **LiLiCorr speculative decoding**, NVFP4 …, **multi-token prediction**, and context reuse" ⇒ 官方把 **LiLiCorr 与 MTP 并列为两条路**（并非 MTP 独占"卖点"字眼）。

### 1.3 NeMo 文档（训练/结构口径；引 3.5 技术报告 §2.1）

URL：https://docs.nvidia.com/nemotron/latest/nemotron/super35/pretrain.html

- **原文（Training Methodology 表，MTP initialization + warm-up (§2.1)，100B tokens）**："**Initialize a single, recursively shared MTP head from the first of Nemotron 3 Super Base's two MTP heads (the second is discarded); train only the MTP head with backbone frozen, using hidden-state mixing (which exposes the head to its own drafted states) to reduce the train–inference mismatch of recursive drafting.**"
- 同页："…hybrid language decoder with its **shared MTP head**"；引用 "**Nemotron 3.5 tech report, §2 (Pretraining), Tables 1–6**"。

### 1.4 NeMo AutoModel 配方（HF repo id / 架构 / config 字段）

URL：https://docs.nvidia.com/nemo/automodel/nightly/recipes-e2e-examples/nemotron-3-5-super-vl

- **HF repo id（确认）**：`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`，arch `NemotronH_Omni_Reasoning_V3`；另有 NVFP4 变体 `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`（SGLang cookbook/博客用后者）。
- **LLM 骨架**：NemotronV3 hybrid，**88 层（40 Mamba2 · 8 attention · 40 MoE）**，hidden 4096；**MoE：512 routed experts/layer，top-22 + 1 shared**。
- **MTP / config 字段原文引用**："**MTP: The checkpoint ships one multi-token-prediction head; the recipe below keeps it off (`num_nextn_predict_layers: 0`)**"。
  ⇒ ① 官方 config 字段名 = **`num_nextn_predict_layers`**；② 这个 SFT 配方**主动关 MTP**（训练不需要它）；③ 与本地 GGUF 的 `nextn_predict_layers=1` 对应（1 个 MTP 头）。

### 1.5 NeMo Lightning 3.5 页（同一 3.5 家族；MTP 模块形态与"2-step recursive"）

URL：https://docs.nvidia.com/nemotron/latest/nemotron/lightning35/pretrain.html

- "**MTP head: a repeated-layer module with an attention + MoE block (`mtp_hybrid_override_pattern: "*E"`), used for speculative decoding at inference**"。
- 发布基座构造："1 MTP head from Nemotron 3 Super Base's head … 2 **MTP warm-up** 100B — backbone frozen, only the MTP head trained, **standard 2-step recursive MTP training**"。
- 配方超参："**MTP: Repeated-layer module, depth 2**（`mtp_hybrid_override_pattern: "*E"`），loss scaling 0.3"。
- 该页同时注明：3.5 技术报告 "**link pending publication**"（⇒ 3.5 报告的直链未上线；见 §5 未验）。

### 1.6 官方/官方模型页给出的"推荐 draft 深度与用法"

| 来源 | 原文 | 等级 |
|---|---|---|
| **SGLang cookbook（NVIDIA 模型页）** https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Super | "**High Throughput leaves MTP off. Low Latency adds EAGLE MTP with top-k 1, three steps, and four draft tokens.** These names identify configuration choices; **no throughput or latency comparison was measured**." | C（SGLang 实现，NVIDIA cookbook 页） |
| **SGLang Day-0 博客** https://www.lmsys.org/blog/2026-10-15-nvidia-nemotron-3-5-super-vl/ | 命令：`--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`；"**With MTP, SGLang can achieve 243.9 tok/s on a single RTX PRO 6000**" | C |
| **Lambda 推理页（转引 NVIDIA cookbook）** https://lambda.ai/inference-models/nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b | "Multi-token prediction with **EAGLE speculative decoding (5 steps, 6 draft tokens)** with **the same checkpoint as the draft model**, **matching NVIDIA's Nemotron 3.5 Super VL cookbook**"；decode CUDA graph 只捕到 bs≤16 | C（但明确说是 NVIDIA cookbook 的配置） |
| **Unsloth GGUF 指南（我们的量化出处）** https://unsloth.ai/docs/models/mtp | "When running GGUFs, MTP can make generation **~1.4× to 2.2× faster**…plan for **~2 GB additional RAM/VRAM** headroom…**`--spec-draft-n-max 2` is the best starting point**, … **Try any value from 1 through 6** and use whichever is fastest." | C（GGUF 出品方） |
| **Unsloth 模型页** https://unsloth.ai/docs/models/nemotron-3.5-super-vl | "…includes a **built-in Multi-Token Prediction (MTP) head**"；"**ships with MTP, which is enabled by default in Unsloth** for faster output. You can turn it off in the settings."；GGUF repo：`unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF` | C（GGUF 出处） |
| **Unsloth PR #13899** https://github.com/unslothai/unsloth/pull/13899 | "**Super uses its embedded MTP head, which measured faster**, so **LiLiCorr is now excluded** from variant menus and quant lookups and **never launched**." | C（与官方 blog"DGX Spark 推荐 LiLiCorr"形成对照：Super 上 Unsloth 选了 MTP） |

---

## §2 这个模型的 MTP 是什么结构（几个头 / 链式还是并行）

**结论：1 个 MTP 头、递归链式（repeated-layer module），不是并行多头。**

1. **官方**：NeMo §1.3 原文 "**single, recursively shared MTP head**"，且 **训练时用 hidden-state mixing 把 MTP 头暴露在"它自己 drafted 的状态"上**、"reduce the train–inference mismatch of **recursive drafting**" ⇒ 设计目的就是**递归/链式起草**（draft 出一个 token 再喂回同一个头继续 draft）。
2. **模块形态**：NeMo Lightning 页 "a **repeated-layer module with an attention + MoE block**（`mtp_hybrid_override_pattern: "*E"`）"；配方里 `depth 2`（该家族 recipe 用深度 2 的重复层）。3.5 Super VL 用 **1 个头**（§1.4 "one multi-token-prediction head"）。
3. **血缘**：Nemotron 3 Super Base 有 **两个 MTP 头**；3.5 Super VL **取第一个、丢弃第二个**（§1.3）⇒ 3.5 的 MTP 初始是 1 头，随后 100B tokens 的"**MTP-boosting**"把 drafter 对齐到最终模型（§1.2）。
4. **训练方式不是"独立并行头"**：不需要多个独立头并行预测 t+2/t+3；技术报告（Nemotron 3 Super）里另有 "**MTP during SFT**: we continue training … with the **same shared-weight MTP head** used in pretraining to preserve … the **inference-time gains from speculative decoding**"（该句来自报告 PDF 检索摘要，见 §5 未验），以及 "Figure 4 plots **acceptance rate as a function of draft token index**"（即按 draft 序号接受率衰减——链式起草的典型形态）。
5. **本地 GGUF 证据（A）**：`block_count=89`、**`nextn_predict_layers=1`**、`expert_used_count=22`；shard2 的 `blk.88` 有 **4 个 nextn 张量**（`nextn.eh_proj/enorm/hnorm/shared_head_norm`）+ **整层 attention/MoE**（attn_q/k/v/o、ffn_gate_inp、ffn_up/down_exps、latent_up/down、norms 等 18 个张量）⇒ 与官方"1 个 MTP 头 = 一个 attention+MoE 重复层"一致。

---

## §3 我们对 MTP 的实现对照（本地，A 源码）

### 3.1 我们代码支持什么（早已实现，不是待移植）

- 检测/装载：`engine.cpp:697-711`（读 `nextn_predict_layers`；`block_=gg_`；`draft_ = block_ && !nextn.embed_tokens && !nextn.shared_head_head` ⇒ 本 GGUF 无这两个 ⇒ draft_=true）；`:1634-1646`（加载 `nextn.eh_proj/enorm/hnorm/shared_head_norm` 四个张量）；`:1535`（`n_all = n_layer_ + block_layer_` ⇒ 第 89 个模块也 load）。
- 前向：`engine.cpp:3653-3673`（`draft_step`：draft_input → block_attend → MoE → head）、`:3675-3693`（`draft()` 返回 1 个 guess + prob）。
- 窗口验证/回退：`main.cpp:432-485`（`engine.window(w, use.t)` → 逐 token `logits_of(i)` 比对 → `keep(right+1)`）；`engine.cpp:3503+`（window/keep）。
- 说明：**poly 的 draft 链路与官方"单头递归链"同构**（一次 draft 一个 token，把 guess 喂回下一个 `draft(from, n_d)`，见 `main.cpp:455-459` "each guess from the one before"）。

### 3.2 我们的默认值（回答 lead 的"默认值 vs 官方建议"）

| 项 | 我们的值 | 证据 |
|---|---|---|
| draft 开关默认 | **`draft = true`**（`--no-draft` 才关） | `engine.hpp:70` |
| 交付配置实际 | **关**：`--no-draft --lookup 0` | `P3\R1-收工-新基线-20261017.md:14` 等价交付配置；`main.cpp:852/851` |
| `--window` 默认 | **0 = 自动**；本模型 `snaps_spread()==true` ⇒ 自动取 **8** | `engine.hpp:68-69`、`engine.cpp:1566-1567`、`nemotronhmoe.hpp:28` |
| 每轮最多验证 token 数 | `most = min(max_window=8, DraftPolicy::kMaxT=8) = 8` | `main.cpp:320`、`include\strata\spec\draft_policy.hpp:24`（kMaxT=8） |
| **MTP 链上限** | **3 个 draft**（"the token the model has and **three guesses, each from the one before**"） | `main.cpp:64-66`（`kBlockWindow=4`）、`:455`（`n_d < min(most,kBlockWindow)-1`） |
| copy 长窗 | 仅当 n-gram 匹配 **≥ kCopyMatch=8** 才用满 `most`（copy 通路，不是 MTP） | `main.cpp:68-70`、`:441` |
| `--window` 语义 | `window()` 一次 **forward** 的 token 数（已知 1 + 猜测） | `engine.hpp:68`、`main.cpp:472` |

**⇒ 与官方建议的差距**：
- 官方 cookbook/配方：**4–6 个 draft tokens**（SGLang 低延迟 3 steps/4 drafts；NVIDIA cookbook 5 steps/6 drafts）；GGUF 出品方 Unsloth：**2 起步、1–6 扫**。
- 我们：**最多 3 个 draft**（硬编码 `kBlockWindow=4`），**交付配置还是关的**。⇒ 深度在官方区间的最低端；且**调 `--window` 不会增加 MTP 深度**（`--window 8` 只影响验证窗/上限；MTP 链仍被 `kBlockWindow=4` 卡 3）——这与 lead 实测"`--window 8` 只 +1.2%"一致（**假说，未验**：要单独跑 `--window 4` vs `8` 且 `--lookup 0` 才能证）。
- **等价 vLLM 参数**：`--window` ≈ `num_speculative_tokens + 1`（已知 1 + 猜测数）；我们的 MTP 链深 3 ≈ `num_speculative_tokens=3`。注意官方/vLLM 的常见 MTP 配置是 **1**（vLLM 源码注释："Enabling num_speculative_tokens > 1 will run multiple times of forward on **same MTP layer**, which may result in lower acceptance rate"——见 §4 链接），与"单头递归链"的物理含义一致。

### 3.3 我们实测（lead 提供，B）
- 开 draft = **21.31 tok/s（+51%，ids 逐位无损）**；`--window 8` 只 **+1.2%**；draft+lookup **不叠加**。
- 对照：老师 3.8-Flash 本机 decode 108 tok/s（2.63 token/轮，A；见 `情报-H`）。

---

## §4 三档清单：离"官方/论文级 MTP 加速"差在哪

### A. 零改码可调（今天就能测，一次编译都不用）
1. **打开 MTP**：去掉 `--no-draft`（`draft` 默认就是 true）。首测建议 **`--lookup 0`**（避免与 n-gram copy 混；lead 已测"不叠加"）。
2. **`--window 4` vs `8`**：默认自动 8；要判"窗口到底有没有用"必须**固定 `--lookup 0 --no-draft 之外的变量`**，因为 `--window>4` 只对 copy 窗生效（MTP 链仍 3）。
3. **`--draft-vocab N`**：本架构 `guess_vocab()` 基类返回 0（全词表，`engine.hpp:318`）；收窄 draft 词表可能省 MTP head 时间，但会降接受率——官方 MTP 共享 output head（全词表），**不建议先动**。
4. **每臂同报** `DONE ... drafts accepted/offered`（`main.cpp:660-668`）+ ids 逐位（无损门）。
   ⚠️ 这一档**改不了 MTP 深度 3**——深度是常量。

### B. 小改（一次编译；对齐官方 4–6 深度与"按 draft index 的接受率"）
1. **`kBlockWindow` 4 → 可配置**（`--spec-draft-n-max` 等价；对齐 Unsloth 的 1–6、NVIDIA cookbook 的 6）：改 `main.cpp:64-66`、`:455`（`kMaxT=8` 已够数组）；默认保持 4（可关 + 负控逐位 ids）。
2. **DONE 行加 per-depth 接受率**（`accepted/offered` by draft 序号）：对齐 Nemotron 3 Super 报告的 "**acceptance rate as a function of draft token index**"，才能选深度（否则只能看总 tok/s，噪声里挑不出）。
3. **概念拆名**：把"**MTP 链深**（kBlockWindow-1）"与"**验证窗**（--window）"分成两个可读旗标（现在 `--window` 调不动 MTP 深，容易被读成"加窗无效"）。
4. （可选）`--draft-vocab` 的默认：官方 MTP 共享 output head；若要省时间可做 A/B。

### C. 大改（才配得上"论文级"）
1. **EAGLE 式 top-k>1 树/多候选**：官方 cookbook 用 EAGLE 算法（top-k 1 = 链）；topk>1 要新的 draft policy + 验证窗形状 + 回退（我们 `main.cpp:432-485` 的窗验证是链式的）。
2. **按接受率自适应深度**：我们已有 `Guesses`/`DraftPolicy` 自适应框架（`main.cpp:315-322/447-461`、【`guesses.hpp`】），只是被 `kBlockWindow=4` 上限卡住；放开上限 = B-1，不改框架。
3. **深窗的 Mamba 状态回退**：官方 blog 明说 "Additional **Mamba checkpoints** reduce the replay needed to recover inputs for speculative decoding"；我们已有快照 `kSnapPlaces=3`（`runtime.hpp:56-58`、`engine.cpp:1712`），**窗 >4 时要更多快照**（显存/时间权衡，需改码）。
4. **训练侧的两件事不可能复现**：官方 "**MTP-boosting training phase**"（对齐 drafter 与最终模型）与 "hidden-state mixing"——权重已定，推理侧只能接受既有接受率。

---

## §5 未验 / 查不到（明写）

1. **官方没有公开"MTP 单独"的加速比/接受率**：NVIDIA 博客的 **2.3× 属 LiLiCorr**（DGX Spark 推荐 drafter）；SGLang 的 **243.9 tok/s** 是 RTX PRO 6000 单点、无 baseline；Unsloth 的 **1.4–2.2×** 是其 GGUF 家族口径。**"官方宣称 MTP 加速多少"= 未查到数字**（只有定性原文，§1.2）。
2. **Nemotron 3.5 Super VL 技术报告的直接 URL 未拿到**：NeMo 各页写 "**Nemotron 3.5 tech report, §2/§4/§5**"，Lightning 页写 "**link pending publication**"；我按 `NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf` 拼的 research.nvidia.com 直链**取不到**（fetch 404）。§1.3 的 §2.1 内容是 **NeMo 文档的转述**（C），不是报告原文。
3. **HF `config.json` 未直接取到**（`huggingface.co/.../raw/main/config.json` 经 fetch 端点 404）；`num_nextn_predict_layers` 的**实际值**只有：NeMo AutoModel 页写字段名 + 本地 GGUF 元数据 `nextn_predict_layers=1`（**A 本地**）。HF 的 `generation_config.json` 未查。
4. **NIM / TensorRT-LLM 对"这一型号"的 MTP 默认 draft 深度未查到**（NIM 参考页只写 "native MTP"，未给参数）；SGLang cookbook 的 4 drafts 与 Lambda 转引 cookbook 的 6 drafts 是**两套官方 cookbook 配方**，谁是新版**未验**。
5. **"老师 3.8-Flash 用 MTP"与本模型的 MTP 不是同一头**（3.8-Flash 的 MTP 是 Strata 侧 `E:\strata-data\mtp\rt` 的独立 draft head）；跨模型不可比。
6. **我们的 `--window 8` 只 +1.2% 的机制解释（链上限 3）是代码推理（A 源码），未做对照跑**。

## §6 落盘与复现

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-I-Nemotron3.5-MTP官方口径-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\II-20261018-Nemotron3.5-MTP官方口径.md`

```powershell
# 本地代码核对
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\main.cpp' -Pattern 'kBlockWindow','kCopyMatch','engine.draft(','engine.window(','most(lookup' -SimpleMatch
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\engine.cpp' -Pattern 'nextn','draft_','opt_.window == 0' -SimpleMatch
# 外部（浏览器打开即可）
# NIM: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b
# Blog: https://developer.nvidia.com/blog/nvidia-nemotron-3-5-super-vl-powers-specialized-agents-with-multimodal-reasoning
# NeMo: https://docs.nvidia.com/nemotron/latest/nemotron/super35/pretrain.html
# AutoModel: https://docs.nvidia.com/nemo/automodel/nightly/recipes-e2e-examples/nemotron-3-5-super-vl
# SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Super
# SGLang blog: https://www.lmsys.org/blog/2026-10-15-nvidia-nemotron-3-5-super-vl/
# Unsloth: https://unsloth.ai/docs/models/nemotron-3.5-super-vl · https://unsloth.ai/docs/models/mtp
```
