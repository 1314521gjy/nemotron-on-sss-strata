# 回执 · 线 D：多样 prompt 台 + decode 判据表 v2（2026-10-18）

```
状态: 达成（零 GPU；3 类异构 prompt 已实体化并往返验证；判据表 v2 落盘；未起引擎）
席位: 验收（reviewer）· 任务: 线 D（计划-decode总攻-20261018.md §2 线 D）
时间: 2026-10-18 08:26 - 09:2x
```

## 交付物

| 绝对路径 | 内容 |
|---|---|
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\多样prompt台-设计-20261018.md` | 台子组成（4 条）· 异构性论证 · 反模板/反抄门（静态已测+运行期待跑）· tokenize 规程 · 跑法规格 |
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\decode判据表-v2-20261018.md` | D0 口径 + D1–D8 判据（每条含绿/红/负控/不能证明什么）+ 类 E 门 + 负控清单 |
| prompt 原文 + ids（ASCII 路径，已实体化） | `C:\sp3\bench\qa.txt` / `qa_ids.txt` · `code.txt` / `code_ids.txt` · `zh.txt` / `zh_ids.txt` · `alt_ids.txt`（alt 原文在 `C:\sp3\alt_prompt.txt`） |
| 工具（新） | `...\验收\prompt_gate.py`（P1/P2/P5 静态门）· `...\验收\decode_ids_gpt2.py`（GPT-2 byte-level 正确 detokenizer，中文必需） |
| **本件** `...\回执\RV-20261018-多样prompt台与decode判据v2.md` | 五段回执 |

## 判据读数（本席实测/复算）

| 项 | 读数 |
|---|---|
| 3 类 prompt 规格 | qa 英文算术推理 **218 tok**（ids sha16 `a3c36b3620193e49`）· code 代码审计 **240 tok**（`54bfd49e266dea28`）· zh 中文指令 **281 tok**（`8db2a0b5dcc2b6c4`）· alt **293 tok**（`ebb605f844af8fd5`）——全部 200–400 |
| tokenize 往返（必须门） | 用 `llama-tokenize --ids --no-bos` → `decode_ids_gpt2.py` → 再 tokenize：qa **218/218**、code **240/240**、zh **281/281**、alt **293/293** 全等 |
| 异构性 | 跨 prompt token-5gram Jaccard 最大 **0.0059**（code↔alt），其余 0.0000 |
| 反模板静态门（能红） | token 8-gram 最大重复：qa 1 / code 1 / zh 1 / alt 2，阈值 ≤2；**p6792 = 10 ⇒ 红**。distinct：0.550/0.617/0.555/0.539（阈值 ≥0.45）；**p6792 = 0.049 ⇒ 红**。段落 distinct：新台 100%；**p6792 9/53 = 0.17 ⇒ 红**。句子 distinct：新台 100%；**p6792 29/259 = 0.11 ⇒ 红** |
| 反抄 prompt 运行门 | 未跑（本轮零 GPU）；阈值 = 生成文本与 prompt 最长公共子串 ≤max(40 chars,20%) 且无 ≥40 字符逐字块；**p6792 历史 128-token 生成 = prompt 完全子串（623/623 chars=100%）⇒ 红例** |
| decode 判据新增 | D1 速度（3 跑中位 + 异构口径；单跑/p6792-only ⇒ 红）· D2 ids 逐位 ≥2 prompt · D3 文本逐字 · D4 确定性（`--expert-swaps 0` ×2 逐位同）· D5 全位置 max\|Δp\| + near-flip（>0.05 跳=红，margin<0.05 单列）· D6 多样+反模板/反抄 · D7 类 E 门 · D8 自洽总闸 |

## 红臂（怎么让判据变红；均已在盘上）

- **D6**：把 p6792 当"多样 prompt" ⇒ 8-gram=10、distinct=0.049、段落 9/53、语句 29/259、生成 100% 抄 prompt ⇒ 四条门同时红（本席复算）。
- **D2**：昨日已建负控——pos64 改 1 token ⇒ LCP=64；打乱 40 token ⇒ LCP=8、tok_ratio 0.4375 ⇒ 红。
- **D4**：R5 实测"同 config decode 概率逐跑不可复现"（pos1–44 有 37/45 处不同）⇒ 若无 D4，逐位门会假绿。
- **D5**：昨日 T11p max\|Δp\|=0.134 @pos103（margin 0.167）⇒ 判 YELLOW（near-flip），证明"文本逐位同"不能替代概率门。
- **D1**：单跑报提速（旧轮 T15 30.97 vs B3 22.09 的单跑对比）⇒ 本表不认，必须 3 跑中位。
- **D3**：用 `detok.py` 解中文 ⇒ 281→1280 token（工具错）⇒ 红。

## 关键决定

1. **台子含 4 条**（qa/code/zh + 既有 alt）：3 类新 prompt 覆盖语言×题型×答案形态三个维度；alt 作为第 4 条快检与历史基线（13.36 tok/s）衔接。
2. **p6792 明确降级为"历史对照/负控"**，不进 decode 结论口径；这直接落实昨晚"模板红利骗结论"的教训。
3. **反模板判据做成"静态（零 GPU 可判）+ 运行期（GPU）"两层**：静态门本轮就已能判红 p6792；运行期门给施工席起跑后用。
4. **判据 v2 以"能变红"为组织原则**：每条都配本机已测负控（p6792 / pos64 翻转 / R5 概率不可复现 / T11p near-flip）。
5. **中文 detokenizer 必须换**：本 GGUF 是 gpt2 byte-level BPE，旧 `detok.py` 只对 ASCII 正确——新增 `decode_ids_gpt2.py` 并逐条往返验证。

## 未验 / 阻塞

未验：新台子（qa/code/zh）的运行期读数全部未跑（本轮零 GPU）；D1 的 +5% 门与 D5 的 0.05 跳门尚未在 3 类新 prompt 上标定（第一批跑后按 MAD 校正）；中文**引擎侧** detokenizer 未核（验收侧标准已定）；`alt` 原文不在 `C:\sp3\bench\`（ids 已补落）。
阻塞：无。零 GPU / 未编译 / 未改码 / 未 spawn 子代理；写入仅 `验收\`、`回执\`、以及 lead 指定的 `C:\sp3\bench\`。
