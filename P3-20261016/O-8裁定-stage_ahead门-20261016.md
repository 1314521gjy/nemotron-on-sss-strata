# 裁定：`--stage-ahead` 的门到底有没有意义 —— **scheduler 对，我的 O-8【撤回】**

> 分歧双方：**我（verifier）**的 `P5-收口复核` **O-8**：*"`read()` 期间 `shared_ahead_=true`（`:2935`）把门短路 ⇒ 阈值大小不生效，只有 0/>0 有差"*
> **vs scheduler** 的 `stage语义与sa门-读码`：*"prompt≈700、`n_calls==1` ⇒ 走逐调用 `eval()` ⇒ `n=700<1024` ⇒ `ahead=false`"*。
> **裁定：scheduler 对**（就本靶当前跑法）；**我的 O-8 结论撤回**；两者其实是**同一份代码的两个分支** —— 判定表见 §3。
> **只读 + 零 GPU**。**AI：deepseek-flash[1m]** ⇒ **自验，不构成独立验收**。证据 **A** = 本机读源码。

---

## §1 ★★ 决定性一行：`read()` 有**早退分支**，短路赋值在它**之后**

`Engine::read()`（`engine.cpp:2824`）在构建完 `calls_` 后（`:2835-2843`）**立刻**分岔：

```cpp
// engine.cpp:2855
if (n_calls == 1 || !mmq_ || all_cached() || n < opt_.stage_ahead) {
    ...
    for (int c = 0; c < n_calls; ++c) { ... eval(tokens + call.at, call.m, false); ... }   // :2870
    ...
    return true;                                                                          // :2882
}
turn = share_to(hi_);
...
shared_ahead_ = true;                                                                     // :2935  ← 在早退【之后】
```

⇒ **四个条件命中任一 ⇒ 逐调用 `eval()` 并 `return` ⇒ 永远到不了 `:2935`**。
⇒ **`shared_ahead_` 在我们这条路径上【恒为初值 false】**（`engine.hpp:627` `bool shared_ahead_ = false;`；只有 `:2996`/`:3001` 会把它复位）。⇒ 闸门 `:2197`
```cpp
const bool ahead = opt_.stage_ahead > 0 && (n >= opt_.stage_ahead || shared_ahead_);
```
里的 `shared_ahead_` **恒 false** ⇒ **门退化为纯阈值 `n >= opt_.stage_ahead`**。**scheduler 对。**

**✍ 补一条我犯错时漏掉的前提（`engine.hpp:627` 注释原文，A）**：
> `bool shared_ahead_ = false;   ///< read() runs **several calls**: each copies ahead, whatever its size`

⇒ **这个短路的语义前提就叫"read 跑多个调用"** —— 而 `n_calls == 1` 恰恰是它的反面 ⇒ **我拿着"多次调用"语义的开关去解释"单次调用"的路径** ⇒ **注释里写着的适用前提，我没读**。

---

## §2 ★ 我错在哪（方法论，值得记）
1. **只读了路径的一半**：读到 `:2935` 就下结论，**没往上看 80 行**（`:2855` 与 `:2935` 之间隔着一个 `return true`）。⇒ **同族于"抽样≠全量"**：**读到一个赋值点 ≠ 读到它的可达性**。
2. **把"存在该语句"当成"该语句会执行"** —— 判可达性必须看**分支**，不能只看**语句**。
3. **注释里的限定条件没读**（`engine.hpp:627` 的 *several calls*）。
⇒ **纪律（自用）**：**凡下"某开关不起作用/被短路"这类结论，必须先交出"它会在哪条分支被走到"**（给 `if` 而不是给赋值行）。

---

## §3 ★★ 两说各在什么条件下成立（判定表）

| 条件 | `n_calls` | 走哪 | `shared_ahead_` | **门** | 谁的说法成立 |
|---|---|---|---|---|---|
| **本靶现状**：prompt 700 · `--prompt-batch 1024` · 无 marks | **1** | **早退 ⇒ `eval()`** | **false** | **纯阈值 `n ≥ sa`** | ★ **scheduler** |
| 同上但 `n < sa`（如默认 sa=1024） | 1（或 >1） | 早退 | false | `n ≥ sa` ⇒ **假 ⇒ 关** | scheduler |
| **长 prompt**：某次 read 调用吃满 1024（见 §4 边界） | 1 | 早退 | false | `1024 ≥ 1024` ⇒ **真 ⇒ 开** | **两者都对**（我错在"阈值无关"这半句） |
| **多调用 read**：`n_calls > 1` **且** `n ≥ sa` **且** `mmq_` **且** 非 `all_cached()` | >1 | **进 staged 路径** | **true** | **被短路 ⇒ 阈值无关（只要 `sa>0`）** | ★ **我 O-8 的适用范围**（在当前跑法下**不可达**） |

⇒ **结论**：**O-8 描述的机制【真实存在但不可达】**（需要 `n_calls>1 且 n≥sa`，而本靶每次 read 的 `n ≤ read_rows_ ≤ --prompt-batch`，且无 marks 时 `n_calls` 只能为 1）。

---

## §4 ★ 精确门语义（写死，供 builder 用）
1. **门 = `opt_.stage_ahead > 0 && (n >= opt_.stage_ahead)`**（本路径，`:2197`）。
2. **`n` = 本次 `moe_rows()` 调用的 token 数**；在 read 早退路径下 = 该次 `eval()` 的 `call.m`。
3. **`n` 的上界**：`generate()` 一次给 read 的量 = `room = min(n_prompt − 1 − i, max_read())`（`main.cpp:372`），而 **`read_rows_ = rows_ < kStageRows ? rows_ : max(rows_, most/rows_*rows_)`**（`engine.cpp:1593`）⇒ **`n ≤ read_rows_`，本配置下 = `--prompt-batch` = 1024**。
   ⇒ ★ **`--stage-ahead > --prompt-batch` ⇒ 永久关**（`n` 永远够不到阈值）。
   ⇒ ★★ **默认 `sa = 1024` 与默认 `pb = 1024` 相等 ⇒ 门【只在"某次 read 恰好吃满 1024"时才开】** —— **这就是"默认值不适合本机"的精确根因**（队列件已列；此条给它一个**充要条件**）。
4. ★★ **边界要按 `n_prompt − 1` 算，不是 `n_prompt`**：`generate()` 把 **prompt 的【最后一个 token】单独交给 `eval()`**（`main.cpp:373` `m = last ? 1 : …`）⇒ **read 实际只看 `n_prompt − 1` 个 token**。
   ⇒ **`n_prompt = 1024` ⇒ read 的 `n = 1023` ⇒ `1023 < 1024` ⇒ 关**；
   ⇒ **`n_prompt = 1025` ⇒ `m = min(1024, 1024) = 1024` ⇒ 开**。
   ⇒ **⇒ 实验 B 的门在 1025，不是 1024（差一），`--prompt≥1024` 这个说法会踩反**。

---

## §5 ★ 对 P0-c 的更正（**我上一条建议撤回**）
- ❌ 我此前说：**"扫 `--stage-ahead` {128,256,512,1024} 可能是空转"**（理由：阈值无关）—— **理由错，撤回**。
- ✅ **正确的结论**：
  1. **阈值【有意义】**（它就是唯一的门）；
  2. 但 **对 700-token prompt，`sa ∈ {1..699}` 三档（128/256/512）【结果相同：都开】** ⇒ **扫 128/256/512 仍是空转 —— 原因不是"阈值无关"，而是"三档都在门的同侧"**；
  3. **有意义的点是【跨过 n】** ⇒ 建议 P0-c 改跑 **`{0, 128, 1024}`**（即"关 / 开（安全侧）/ 开（等于 n，边界）"），**若要细扫只值得加边界点 `699 / 700`**。
- ★ 一句话给 builder：**判"门"只需两点 —— `sa ≤ n` 与 `sa > n`**；扫满一档是浪费窗口。

## §6 ★ 实验 B（长 prompt 验门）的判据 —— 我把它写成可执行的一刀
**目的**：证明"门按**本次调用的 n** 判"，并证伪"默认 sa 永远等于关"。
- **配置**：**不加 `--stage-ahead`**（用**默认 1024**——这才是被测对象）；其余用新 canonical；冻结 profile。
- **两臂（唯一变量 = prompt 长度）**：
  | 臂 | prompt token 数 | 预测（read 的 n） | 预测 `ahead` | 预测 prefill |
  |---|---|---|---|---|
  | **B-off** | **≤1024**（如现有 700） | 699 | **假** | 慢档 |
  | **B-on** | **≥1025**（建议 **1200**） | 1024 | **真** | **跳回"开"的那一档** |
- **判据**：**B-on 的 prefill 显著高于 B-off**（差 > `max(2·s_round, 5%)`，各 ≥3 跑 + 弃预热）；**且 B-on 的 prefill 应落在"用 `--stage-ahead 128` + 700-token prompt"那一档附近**（同侧 ⇒ 同一机制）。
- **负控（能变红）**：**两臂都改回 `--stage-ahead 0`** ⇒ **两臂都应落到慢档、差消失** ⇒ 若差仍在 ⇒ **说明"差"不是门造成的**（判据失效）。
- **★ 别让第三件事混进来**：**`--prompt-cache-every` 保持默认**（见 §7）。

## §7 ★ 一条**交叉影响**警告（给 P3 实验）
**「marks」会改变 read 的调用切分**：`--prompt-cache-every E`（默认 2048）会在 read 内切出 checkpoint（`conversations.hpp:106-127` `inside()`）⇒ `calls_` 按 mark 分段 ⇒ **`n_calls > 1`**。
- 但**只有当 `n ≥ sa` 时才进 staged 路径**（`:2855` 第三项仍是 `n < sa ⇒ 早退`）。
- ⇒ **若 `sa` 落在 `(call.m, n_prompt−1]` 区间内**（如 `sa=300`、mark 把 read 切成 256 一段）⇒ **staged 路径里每段 `n=256 < 300` ⇒ 门【关】**，而无 marks 时 `n=699 ≥ 300` ⇒ **门开** ⇒ **★ 加 `--prompt-cache-every` 会把重叠关掉**。
- ⇒ **给 P3 的纪律**：**P3（`--prompt-cache-every 128/256`）与 P0-c（`--stage-ahead` 扫档）【不许同跑】**；P3 跑完若发现 prefill 塌，**第一嫌疑就是这条**（**而不是"复用变慢了"**）。

---

## §8 未验 / 边界
- **只读、零 GPU、未跑**；§3/§6 的"走哪条分支"是**读码推得**（A），**未用一次真实运行确认**（如打印 `n_calls`/`shared_ahead_` 的临时读点）⇒ **若你要一锤定音**：让 builder 加一个**只读打印**（`n_calls` · `n` · `shared_ahead_` · `ahead`，env 门控、默认可 revert）跑两臂即可**当场实证**。
- **我未核**：`most`（Session 的窗口）> 1 时 `read_rows_ = max(rows_, most/rows_*rows_)` 是否会把 `read_rows_` 抬到 > `rows_`（`--lookup`/draft 开时可能）⇒ **本靶 canonical `--no-draft --lookup 0` ⇒ `most=1` ⇒ `read_rows_ = max(1024, 0) = 1024`**（推得，未实跑）。
- **我未核**：多卡（`next_`）时早退分支的行为（本靶单卡，不涉及）。
- ★ **本件与 `P5-收口复核` 的 O-8 冲突：以本件为准**（O-8 那条**已撤回**，`P5` 件内已加更正指针）。
