# 回执 · 情报I：Nemotron-3.5-Super-VL-120B-A12B 的 MTP 官方口径（task-21，收窄版）— 2026-10-18

## 1. 任务
`task-21`（lead 收窄：**只查这一个模型的论文与官方口径**，不做通用框架综述）：① HF model card / config 的 MTP 字段与推理用法；② NVIDIA 官方论文/技术报告的 MTP 章节（训练方式、宣称加速比/接受率、推荐配置）；③ NVIDIA 官方推理路径（NeMo/NIM/TensorRT-LLM）对本型号的 MTP 用法；④ GGUF 出品方（Unsloth）文档；⑤ 本地对照（`--window` 默认、链式上限 3 的由来）⇒ 回答"官方建议 draft 深度 vs 我们默认"。零 GPU、零编译、不改码；外部每条带 URL，本地每条 file:line。

## 2. 状态
**达成（5/5）**：官方原文引用 6 组（NIM 模型卡 / NVIDIA 技术博客 / NeMo super35 训练页 / NeMo AutoModel 配方 / NeMo Lightning 3.5 页 / SGLang cookbook+博客+Lambda 转引 / Unsloth 文档+PR）；结构判定 1 个头递归链式（附本地 GGUF 证据）；本地默认值与官方建议逐项对照；三档清单（零改码/小改/大改）；未验 6 条明写。红线：零 GPU、零编译、零改码；只写本回执与详细件。

## 3. 结论（一句话）
**官方对 MTP 的口径是定性的"内建 drafter（native speculative decoding）+ 专用 MTP-boosting 训练阶段"，结构是"单个、递归共享的 MTP 头（attention+MoE 重复层），链式起草"；官方没有公开 MTP 单独的加速比/接受率（2.3× 属 LiLiCorr、243.9 tok/s 是 SGLang 在 RTX PRO 6000 的单点、1.4–2.2× 是 Unsloth 的 GGUF 家族口径）；官方 cookbook/配方推荐 4–6 个 draft tokens（SGLang 低延迟 3 steps/4 drafts；NVIDIA cookbook 5 steps/6 drafts），GGUF 出品方建议 2 起步、1–6 扫；我们代码默认 draft=true、`--window` 自动 8，但 MTP 链被 `kBlockWindow=4` 硬卡 ≤3 个 draft，交付配置还把它关了（`--no-draft --lookup 0`）。**

## 4. 证据（要点）
- **NIM 模型卡**（https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b）："**Multi-Token Prediction (MTP) enables native speculative decoding**"；Model Summary = "Native MTP"；512 routed experts/MoE layer；Runtime engines = vLLM/SGLang/TensorRT-LLM。
- **NVIDIA 技术博客**（https://developer.nvidia.com/blog/nvidia-nemotron-3-5-super-vl-powers-specialized-agents-with-multimodal-reasoning）：MTP = "**internal drafter that proposes several tokens for the main model to verify in one forward pass**"；"**A dedicated MTP-boosting training phase further aligned the drafter with the final model**"；"Additional Mamba checkpoints reduce the replay needed …"；**2.3× 明确属 LiLiCorr**（"LiLiCorr is the recommended drafter on DGX Spark… up to 2.3x"）。
- **NeMo super35 pretrain**（https://docs.nvidia.com/nemotron/latest/nemotron/super35/pretrain.html）："**single, recursively shared MTP head from the first of Nemotron 3 Super Base's two MTP heads (the second is discarded)**"；"**hidden-state mixing (exposes the head to its own drafted states) to reduce the train–inference mismatch of recursive drafting**"；100B tokens；引 "Nemotron 3.5 tech report §2"。
- **NeMo AutoModel 配方**（https://docs.nvidia.com/nemo/automodel/nightly/recipes-e2e-examples/nemotron-3-5-super-vl）：repo id `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`；88 层（40 Mamba2+8 attn+40 MoE）、512 experts/top-22+1 shared；"**MTP: … one multi-token-prediction head; the recipe keeps it off (`num_nextn_predict_layers: 0`)**"。
- **NeMo Lightning 3.5**（https://docs.nvidia.com/nemotron/latest/nemotron/lightning35/pretrain.html）：MTP = "**repeated-layer module with an attention + MoE block (`mtp_hybrid_override_pattern: "*E"`)**"；"standard **2-step recursive** MTP training"；3.5 报告 "link pending publication"。
- **官方 cookbook 深度**：SGLang cookbook（https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Super）"Low Latency adds EAGLE MTP with top-k 1, **three steps, and four draft tokens**；High Throughput leaves MTP off；no throughput/latency comparison measured"；SGLang blog（https://www.lmsys.org/blog/2026-10-15-nvidia-nemotron-3-5-super-vl/）命令 **5 steps / 6 draft tokens**、**243.9 tok/s on RTX PRO 6000**；Lambda 转引 NVIDIA cookbook（https://lambda.ai/inference-models/nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b）**5 steps / 6 draft tokens + same checkpoint as draft**。
- **GGUF 出品方 Unsloth**：模型页（https://unsloth.ai/docs/models/nemotron-3.5-super-vl）"built-in MTP head… **enabled by default in Unsloth**… turn it off in settings"；MTP 指南（https://unsloth.ai/docs/models/mtp）"GGUF 上 MTP **1.4×–2.2×**、~2 GB 额外显存、**`--spec-draft-n-max 2` 起步、1–6 扫**"；PR #13899（https://github.com/unslothai/unsloth/pull/13899）"Super uses its **embedded MTP head, which measured faster**, so **LiLiCorr is excluded**"。
- **本地（A 源码）**：`engine.cpp:697-711`（检测 `nextn_predict_layers`，`draft_`）、`:1634-1646`（4 个 nextn 权重）、`:3653-3693`（draft 前向）；`main.cpp:64-66`（`kBlockWindow=4`，"the token the model has and **three guesses, each from the one before**"）、`:455`（链式循环 `min(most,kBlockWindow)-1`）、`:320`（`most=min(max_window,kMaxT=8)`）、`:432-485`（窗验证/回退）、`:660-668`（DONE 的 drafts accepted/offered）；`engine.hpp:68-70`（`window=0` 自动、`draft=true`）、`engine.cpp:1566-1567`（自动 8/4）、`nemotronhmoe.hpp:28`（`snaps_spread()=true` ⇒ 本模型自动 **8**）、`draft_policy.hpp:24`（`kMaxT=8`）。
- **本地 GGUF MTP 证据（A）**：`nextn_predict_layers=1`、`block_count=89`、`expert_used_count=22`；`blk.88` = 4 个 nextn 张量 + 整层 attention/MoE（与官方"一个 attention+MoE 重复层"同构）。

## 5. 未验
1. **官方无 MTP 单独的加速比/接受率数字**（只有定性原文）；2.3×=LiLiCorr、243.9=SGLang 单点、1.4–2.2×=Unsloth 家族口径。
2. **3.5 技术报告直链未拿到**：NeMo 各页只写 §2/§4/§5；Lightning 页写 "link pending publication"；拼出的 `NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf` 直链 fetch 404 ⇒ §1.3 内容是 NeMo 转述（C）。
3. **HF `config.json` / `generation_config.json` 未直接取到**（fetch 端点 404）；`num_nextn_predict_layers` 实际值只有本地 GGUF（=1，A）与 NeMo 页字段名。
4. **NIM/TensorRT-LLM 对本型号的 MTP 默认 draft 深度未查到**；SGLang cookbook 的 4 drafts 与 Lambda 转引的 6 drafts 是两套配方，新旧未验。
5. 老师 3.8-Flash 的 MTP 是独立 rt head（`E:\strata-data\mtp\rt`），与本模型 MTP 不同源，不可比。
6. 我们 "`--window 8` 只 +1.2% ⇒ 链上限 3" 是代码推理（A 源码），未做 `--window 4` vs `8` 的对照跑（零 GPU 约束）。

## 6. 落盘路径（绝对）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-I-Nemotron3.5-MTP官方口径-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\II-20261018-Nemotron3.5-MTP官方口径.md`
