# R76d · `--max-context` 上限（16K → 1M）是否免费 — 实测通过 — 2026-10-18

- 席位 builder4；**零改码**探针；同一 exe 两档 max-context 各 2 跑、交错 M0→M1→M1→M0。
- 配置：p6792 交付形态，`--max-new 128`，其余同交付（BORROW=1、PIN=59.1、M2=1、ring2、reserve 3000、`p_deliver`）；1 跑/次；1 Hz 峰值。
- **运行 exe 的 sha（实测时）**：`C:\sp3\sp_b1prod.exe` = `7C0717994B0D…`（注意：与 R75 时 Lead 给的 `97A1875E71D6` 不同，builder3 在 R75 后更新过该 exe；两臂同 exe，不影响本对比）。
- 日志：`C:\sp3\R76-{M0,M1}-20261018.log` · 汇总 `C:\sp3\R76-maxctx-summary-20261018.txt`/`.json` · 驱动 `C:\sp3\run_maxctx_probe.ps1`。

## 四跑表

| 臂 | `--max-context` | prefill tok/s | decode tok/s | 峰值 MiB |
|---|---|---|---|---|
| M0 run1 | 16,384 | 1663.1 | 28.50 | 31,283 |
| M1 run2 | 1,048,576 | 1702.7 | 28.42 | **31,286** |
| M1 run3 | 1,048,576 | 1706.4 | 28.48 | 31,270 |
| M0 run4 | 16,384 | 1704.9 | 28.20 | 31,282 |

- prefill M0 avg 1684.0 → M1 avg 1704.6 = **+1.2%**（同批噪声内）；decode 28.35 → 28.45 = **+0.4%**。
- **峰值：31,283 → 31,278 = −5 MiB**（判据"差 ≤64 MiB"**通过**；远小于 256 MiB 的"提前映射"红线）。
- **ids M0 == M1 逐位同**（128/128），臂内也同。

## 结论

**`--max-context 1048576` 是零成本上限**：KV 由 VMM 按需映射（`kv_alloc` 只 reserve 地址、`cover()` 才 map 行），峰值不增、prefill/decode 不退化、ids 同 ⇒ **已采纳进交付脚本**（Lead 已改 `--max-context 16384 → 1048576`）。

附注（读码结论，供将来真跑长上下文时注意）：`nemotronhmoe.cpp` 的 `chunk_ = clamp(kScoreFloats/(n_head*max_context), 1, rows_)` 在 max_context 极大时会缩到 1、`score_` 缓冲 `chunk*n_head*max_context` 会涨到 ~130 MB——本轮 p6792 短跑未触发（实际 attend 行数小），**真跑到 ~1M 时要复查这两处**。
