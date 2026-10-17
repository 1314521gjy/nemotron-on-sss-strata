# 投机"能不能叠加"：引擎自带 lookup × 模型自带 MTP —— 读码定死 — 2026-10-16

- 席位：`scheduler` · **只读**（`grep`/读文件/`git show`）：**未跑 GPU、未改码**。
- 触发（新思路，2026-10-16）：**模型自带的 MTP 投机与引擎自带的投机是否可叠加** ⇒ 要判**能不能叠**。
- **★ 一句话结论**：**两个来源【可以同时被问】（代码本来就是这么写的），但【每一轮的窗口只能装一个来源的 token】⇒ 所以现在是"择优"不是"叠加"；要真叠加（先 lookup 白抄、再用 MTP 补满）必须改 3–4 处（全在 `main.cpp` 的 decode 循环 + `DraftPolicy::Pick` 的语义），不动内核 ⇒ 按既定框架这属"加构造"，不是"调配方"。**

---

## §0 四问速答

| # | 问题 | 判定 |
|---|---|---|
| **1** | 引擎的投机通路是**单来源**吗？模型 MTP 与 lookup 能否**同时启用**？ | **不是单来源**；**两个来源在同一轮里都能被问**（`main.cpp:439` 与 `:446/458` 各出一套提案）⇒ **可并存**（**但要满足一个抑制条件**，见 §2） |
| **2** | 若能并存，**窗口怎么拼**？ | **不拼** —— **两个提案 → 一次仲裁**（`policy.choose`），**窗口内容二选一**（`if (!use.lookup) std::copy(d, …)`，`:471`）⇒ **现在没有"前段 lookup + 后段 MTP"的拼接** |
| **3** | 若互斥，是**代码层互斥**还是**没接线**？ | **都不是互斥**；有**两条真实限制**：① `blocks` 的抑制条件（`:446`，**优化**而非互斥）；② **窗口单来源**（`:471`）⇒ ★ **要叠加 = 加构造**（改 3–4 处），**不是**"把开关打开就行" |
| **4** | 最自然的叠法在**现有代码里可实现吗**？动哪几处？ | **可实现，且不动内核**；**动 4 处**（§4）—— 但**前提是先把 MTP 接线补齐**（`G2b` 判"缺 3 处"） |

---

## §1 代码事实（A：`src/poly/main.cpp` 的 decode 循环 + `src/poly/guesses.hpp`）

**一轮（round）里发生的事，按行序**：

| 步 | 代码 | 说明 |
|---|---|---|
| ① **lookup 提案** | `offered = look && room ? repeats.propose(most - 1, w + 1) : 0;`（`:439`）· `match = repeats.last_match();`（`:440`） | **引擎自带**的 prompt-lookup 起草：从**历史重复的后缀**抄出 `offered` 条，并给出匹配长度 `match` |
| ② 本轮窗口上界 | `widest = match >= kCopyMatch ? most : std::min(most, kBlockWindow);`（`:441`）· `k = std::min(offered, widest - 1);`（`:442`） | lookup 填不满时窗口缩到 `kBlockWindow` |
| ③ **是否问 MTP block** | `blocks = engine.drafts() && room && !(k == widest - 1 && policy.lookup_rate(match) >= 0.9);`（`:446`） | ★ **唯一抑制条件**：**lookup 正好填满窗口、且该长度的历史命中率 ≥ 0.9 ⇒ 本轮不问 block**（"the block would mostly say the same, and asking it takes time" —— 注释原文） |
| ④ | `const bool ask = guesses.begin(blocks, policy);`（`:447`） | 由 `Guesses` 的**收益门**再判一次（见 `guesses.hpp:35-55` 的 `Gate`：拿"问了 block"与"没问"的实测速率比，落后的一方每 8/24/96 轮问一次） |
| ⑤ **MTP 提案** | `for (…; n_d < std::min(most, kBlockWindow) - 1 && guesses.wants(n_d); …) d[n_d] = engine.draft(from, n_d, &p);`（`:455-460`） | **模型自带**的 `nextn` 块：**逐条往后猜**（每条以上一条为输入），**条数上限 = `min(most, kBlockWindow) - 1`**；`guesses.takes()` 逐条判"这一条还值不值" |
| ⑥ **仲裁** | `use = policy.choose(1 + n_d, k, match);`（`:462`） | ★ **policy 同时拿到两边的提案**：`1 + n_d`（MTP 侧）与 `k, match`（lookup 侧）⇒ **返回 `{t, lookup}`** |
| ⑦ **窗口内容（二选一）** | `if (!use.lookup) std::copy(d, d + use.t - 1, w + 1);`（`:471`）· `engine.window(w, use.t);`（`:472`） | ★★ **`use.lookup` 为真 ⇒ 窗口用 lookup 抄来的 token；为假 ⇒ 窗口用 block 猜的 token** ⇒ **同一窗口不混装两个来源** |
| ⑧ 验收 | `logits_of(i)` / `keep(right + 1)` / `r.guessed += use.t - 1; r.guessed_right += right + …`（`:477-483`） | 两来源**共用同一套验收**（窗口就是一串 token + 它们的 logits） |

**窗口宽度的来源**：`most = lookup > 0 || e.drafts() ? std::min<int>(e.max_window(), DraftPolicy::kMaxT) : 1`（`main.cpp:320`）⇒ **只要 lookup 或 draft 任一启用，窗口就开到 `min(--window, kMaxT)`**（`--window` 默认 8）。

---

## §2 逐问判定

### Q1：单来源吗？能同时启用吗？
- **不是单来源**：**两种 drafter 在同一条代码路径里、同一轮里各出一套提案**（`:439` 与 `:455-460`）。
- **能同时启用**：`--lookup N`（默认 **3**，`0` = 从不）与 `--no-draft`（默认 **开**，即 block 可用）**互不排斥**。
- **但有两条"限流"**，都不是互斥：
  1. `blocks` 的抑制（`:446`）：lookup 填满窗口 + 该 match 长度命中率 ≥0.9 ⇒ **本轮不问 block**（省时间）；
  2. `Guesses::Gate`（`guesses.hpp:41-48`）：按实测速率决定"这一轮问不问 block"（落后方每 8/24/96 轮问一次）。
- ⇒ ★ **"同时问"是现状**；**"同时用"不是**（见 Q2）。

### Q2：窗口怎么拼？
- **不拼**：**两个提案 → 一次 `policy.choose` → 返回 `{t, lookup}`** ⇒ **`t` 是窗口宽度，`lookup` 是"用哪一家"的布尔**（`:462` + `:471`）。
- ⇒ **不是"先 lookup 再 MTP"的链**，而是**一次仲裁、整体取一家**。
- **"`--window` 8 与草稿 3"的关系**：`most = min(--window, kMaxT)` 是**窗口总宽**；MTP 侧上限 = `min(most, kBlockWindow) - 1`；lookup 侧上限 = `most - 1`。**GLM 文档说的"the draft block's three at most"** 是**上游 `strata` 引擎**的 `--window` 说明（本席另一件已引）；**我们这边由 `kBlockWindow` 常量决定 —— 该常量的取值本席【未核】**（标 **未验**）。

### Q3：互斥吗？代码层还是没接线？
- **不是互斥，也不是"没接线"**：**同轮双提案是现成的**。
- **真正的限制只有一条**：**窗口内容单来源**（`:471`）⇒ ★ **要"叠加"必须加构造**。
- **⇒ 按既定框架**：**先做"调配方/测量"**（把 `--lookup` 与 block 同时打开、看 policy 在两来源间的选择质量），**只有当实测指出"policy 的选择不够好"时**，才去加"拼接"构造。

### Q4：真叠加（先 lookup 白抄 + MTP 补满窗口）要动哪几处？
| # | 动什么 | 位置 | 说明 |
|---|---|---|---|
| 1 | **窗口内容改成拼接** | `main.cpp:471` | `if (!use.lookup) copy(d, …)` ⇒ 改成"前 `m` 条来自 `repeats`，后 `n_d` 条来自 `d`"（lookup 的 token 已在手） |
| 2 | **MTP 的起点接上 lookup 段末尾** | `main.cpp:455-458` | 现在 `from` 从本轮种子 `t` 起；拼接时要让链**接到 lookup 段的最后一个 token** |
| 3 | **`DraftPolicy::Pick` 的语义** | `main.cpp:452/461` + `strata/spec/draft_policy.hpp` | `{t, lookup}` 的 `lookup` 是**布尔** ⇒ 要能表达"两段拼接"（或加一个 `m` = lookup 段长度） |
| 4 | **记账口径** | `main.cpp:481-483` | `r.guessed/guessed_right` 要**按来源分别记账**（否则"叠加"的收益无法归因） |
- **不动内核**：窗口本质是"一串 token + 它们的 logits"，`logits_of()/keep()` 同一套 ⇒ **不需要新 kernel**。
- **代价/风险**：投机是**近似档**（改结果）⇒ 必须**单独成档 + `--top-probs` 逐位判据**（`跑法纪律 D6`）。
- **★ 前提（必须写）**：**现在开 MTP 是"半载"** —— `G2b-MTP通路判定-20261016.md` 判 **"机制齐、接线缺 3 处"**（缺 `block_attend()`、`L.ffn_norm` 从没被设 等）⇒ **"叠加"的评估必须排在"把 MTP 接线补齐"之后**，否则叠的是一个跑不完整的 block。

---

## §3 与已知外部证据的关系（**别把两件事混**）

| 证据 | 说什么 | 与本问的关系 |
|---|---|---|
| 同门 `docs/GLM.md:209` | **GLM 上 MTP 开/关 ≈ 无差**（33.7/31.4 vs 31.3/30.9/31.4 tok/s） | **那是"模型 MTP 单独开"**（很可能没调温度）；**本问是"两来源叠加"** ⇒ **不构成对叠加的否证** |
| `KP-MOEDEP-0007` | "MTP 接受率升仍输 decode"（草稿头常驻吃专家槽） | ★ **叠加会把这个风险放大**：**两个 drafter 都吃资源**（MTP 吃专家槽/时间；lookup 吃 CPU 与内存）⇒ **判据必须防"接受率升、decode 降"** |
| 此前的"温度 0.3 ⇒ 接受率 70%" || —— | **公开件里无出处**（本席另件已 grep 零命中）⇒ **不许当依据** |

---

## §4 未验 / 边界
1. **【A】** §1/§2 的全部行号与行为均**读码实测**（`main.cpp:436-484`、`guesses.hpp`、`main.cpp:320`）。
2. **【D】`kBlockWindow` 的取值未核**（决定 MTP 侧一次最多猜几条）；**`DraftPolicy` 的 `choose()` 内部打分公式未逐行读**（只读了它的调用点与返回类型）⇒ **"policy 在两来源间选得好不好"本席未判**。
3. **【D】** §2-Q4 的 4 处改动是**评估**（**未改码、未编译**）；改动面大小**未实测**。
4. **【D】** **本靶的 MTP 现在跑不完整**（`G2b` 的 3 处接线）⇒ **"叠加"能否收益，本席不下结论**。
5. 本件**未跑 GPU、未改码**；未跑重 CPU/磁盘扫描。

## §5 复现（只读命令）
```bash
P="…/PolyStrata/src/poly"
sed -n '436,484p' "$P/main.cpp"      # 一轮：两提案 + 仲裁 + 窗口二选一
sed -n '316,324p' "$P/main.cpp"      # most = min(--window, kMaxT)
grep -n "kBlockWindow\|kCopyMatch" "$P"/*.hpp "$P"/*.cpp | head
sed -n '35,60p;65,95p' "$P/guesses.hpp"   # Gate 与 Guesses（收益门）
```
