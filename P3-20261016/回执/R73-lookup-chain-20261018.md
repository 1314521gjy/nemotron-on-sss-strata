# R73 · `--lookup-chain` 接线（`STRATA_LOOKUP_CHAIN`）— 判否 — 2026-10-18

- 席位 builder4；实现 `main.cpp +~58`（默认 0，`=0` 逐位等价旧行为）；与 R72 同一次编译（`build_asyncchain.log` rc=0，exe sha 前缀 `2511900F7C6B`）。
- 上游依据（读码纠正）：真正机制在 `E:\strata\src\program\generate.cpp:3267-3277` + `:3949-3955`（`sfx.propose()` → `sbuf[0]==drafts[0]` agreement 门 → `DraftPolicy::choose`）；poly 早已有 `SuffixDrafter`/`DraftPolicy`，本件补的是 agreement 门 + 链式拼接（窗口 = `[t, lookup0(=d0), lookup1..]`，`<= min(most, kBlockWindow)=8`；链跑里不取纯 lookup 窗口以隔离）。
- 配置：alt 293 + `--draft-vocab 65536 --window 8` + `--lookup 0` + R71 T1 同环境（swaps 0、`p_decode_only`、`PROFILE_DECODE_ONLY=1`、`SWAP_DIRECT_FROM_PIN=1`、reserve 3000）；1 跑/次；断言/峰值同 R72。
- 日志：`C:\sp3\R73-{C0,C1,C2,C3}-20261018.log` · 汇总 `C:\sp3\R73-chain-summary-20261018.txt`/`.json` · 驱动 `C:\sp3\run_lookup_chain.ps1`。

## 四跑表

| 臂 | 配置 | tokens/round | offered/accepted | chain fired | chained tokens | decode tok/s | 峰值 MiB |
|---|---|---|---|---|---|---|---|
| C0 | `=0` | 3.02 | 107/107 | 0 | 0 | 14.77 | 31,575 |
| C1 | `=1` | 3.10 | 112/112 | **2** | 14 | 14.68 | 31,539 |
| C2 | `=1` 重复 | 2.65 | 102/102 | **1** | 7 | 14.86 | 31,500 |
| C3 | `=0` CTRL | 3.02 | 111/111 | 0 | 0 | 13.81 | 31,549 |

- **tokens/round 中位：base 3.02 → chain 2.88 = −4.8%**；**chain 只触发 3/214 轮（1.4%）**（C1 的 answer summary：`fired in 2 rounds, chained 14 tokens`；C2：`1 round, 7 tokens`）。
- decode 配对 base 14.29 → chain 14.77 = +3.4%，但 CTRL C0 vs C3 = **−6.5%**（>5%，作废）⇒ decode 不作数。
- ids：C1 vs C0、C2 vs C1、C0 vs C3 全不同（同配置 C0/C3 也不同 ⇒ MTP 非确定性，按 class-E 记录，不作门）。
- 峰值 31,500–31,575 ≤31,700 ✓（余量 ~125 MiB，未破门）。

## 判决

**判否（不进交付候选）**：agreement 门在我们负载上几乎不成立（`lookup[0]==MTP d[0]` 的轮次仅 1.4%），链式窗口因此对 tokens/round 无可测正收益（中位 −4.8%，属噪声/轻微负）；不是实现错，是条件稀有。**代码留树默认 0**（`=0` 逐位等价旧行为）。
**后续建议**（若要再试）：放宽 agreement 门（例如允许多数前缀一致）或把链式窗口用于 `lookup_rate` 高的匹配桶；但当前判据下判否。


## 为什么判否（一句话，以后不重试）

**agreement 门（lookup[0]==MTP d[0]）在本负载只触发 3/214 轮 = 1.4%，链式窗口对 tokens/round 无可测正收益（中位 −4.8%）⇒ 条件稀有而非实现错误，除非换负载/放宽门，否则不重试。**

## 幂等 / 可复现说明

**不是幂等机制**：同一 =1 配置重复跑（C1 vs C2）tokens/round 3.10 vs 2.65、ids 不同；连同一 =0 配置（C0 vs C3）ids 也不同 —— MTP/链式的窗口与 accept 数依赖墙钟时间与负载，**ids 不可复现（class-E）**，所以任何以 ids 为门的验收都不能用这个模式；本批 ids 只作记录。
