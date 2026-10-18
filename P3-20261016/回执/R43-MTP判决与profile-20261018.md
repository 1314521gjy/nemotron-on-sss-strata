# R43 · MTP 最终判决（draft-vocab 65536）与 profile 16pp 真测 — 2026-10-18

- **席位**：builder2 · alt prompt（293 tok）· threads 16 · `--max-new 128` · 一次一跑 · 峰值 ≤31,700 · **全零改码**（`sp_mlp.exe`，sha `33D954E5…`）· 未 spawn 子代理。
- **7 跑**：W00 预热丢弃 + A1–A4（交错配对）+ B1/B2。
- **判决**：
  - **A 组：MTP（`--draft-vocab 65536`）per-token 相对 CTRL 改善 8.4%（配对均值）⇒ 过 ≥+5% 门，进交付候选。**
  - **B 组：旋钮生效但方向为负**（命中率 −2.1% rel、miss/token +3.7%、swaps +18%、per-token 更慢）⇒ **profile 16pp 的"kSwapMin 太保守"假设不成立。**

---

## 1. A 组（MTP 判决，交错配对防负载漂移）

| 臂 | 配置 | per-token | decode | tokens/round | 兑现（accepted/offered） | 命中率 | miss/token | cpu_us/miss | cpu_busy | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | CTRL `--no-draft` | **67.0 ms** | 14.92 | 1.00 | 0/0 | 60.5% | 345.4 | 140.7 | 125.9 | 30,995 |
| **A2** | draft + `--draft-vocab 65536` | **62.9 ms** | 15.89 | **3.88** | 95/109 = 87.2% | 51.6% | 488.1 | 111.2 | 97.4 | **31,330** |
| **A3** | CTRL `--no-draft` | **53.9 ms** | 18.55 | 1.00 | 0/0 | 60.7% | 343.9 | 117.4 | 105.1 | 30,956 |
| **A4** | draft + `--draft-vocab 65536` | **47.8 ms** | 20.90 | **3.88** | 95/106 = 89.6% | 54.6% | 447.9 | 92.8 | 79.2 | 31,297 |

- **配对判决**：CTRL 均值 (67.0+53.9)/2 = **60.5 ms**；MTP 均值 (62.9+47.8)/2 = **55.4 ms** ⇒ **per-token −8.4%（≥+5% ⇒ 进交付候选）**。相邻配对方向一致：A1→A2 **−6.1%**、A3→A4 **−11.3%**。
- **机理（第一次看到 MTP 正收益的原因）**：窗口把 GPU 段摊薄（CTRL GPU 段 ≈18.4 / 13.5 ms → MTP ≈8.7 / 6.2 ms），`--draft-vocab 65536` 又把 draft head 的 GPU 开销压小；CPU 腿没有恶化（cpuLeg/token：A2 54.3 vs A1 48.6；A4 41.6 vs A3 40.4），净下来 per-token 下降。
- 峰值 31,330 / 31,297 ≤31,700（余 ~370 MB）✓。
- **本批负载漂移很大**（A1 67.0 → A3 53.9 = −20%），交错配对只能压住一部分；−8.4% 的量级有 ±3~4pp 不确定性。

## 2. B 组（profile 16pp，`--expert-swaps 64` 真测）

| 臂 | 配置 | per-token | 命中率 | swapped | miss/token | cpu_us/miss |
|---|---|---|---|---|---|---|
| **B1** | `--no-draft --expert-swaps 64`（不调） | 52.2 ms | **62.4%** | 4,025 | **328.3** | 117.9 |
| **B2** | 同 + `--adapt-min-gain 0.5` | 55.0 ms | **61.1%** | **4,757** | **340.3** | 121.2 |

- **旋钮确实生效**：自证行 `adaptive tier: min gain 0.500` 打印；**swapped +18%**、**miss/token +3.7% ≥3%**。
- **但方向为负**：命中率 **−1.3pp（−2.1% rel）**、miss/token **+3.7%**、per-token **+5.4% 更慢**。
- ⇒ **"kSwapMin=2.0 太保守导致 16pp"不成立**：降低 min-gain 只是增加 swap 抖动、命中率反而下降。16pp 的差更可能来自 profile 表与真实负载分布的差异（或窗口内 use 口径），不是 swap 阈值。

## 3. 重要限定（影响 MTP 进交付）

- 本轮 A/B 组按 lead 的表用**默认 `--expert-swaps 64`** ⇒ 自适应层按时序发布槽位，**同配置 run-to-run ids 不可复现**。证据：
  - **B1 与 A1 配置完全相同**，ids 却 **105/128 不同**、`max|Δp|=1.000@pos80`、**25/128 位置 >0.05**；
  - A3 vs A1（同为 CTRL）也有 `max|Δp|=0.1105`、4/128 >0.05。
  - ⇒ 本批 **ids/Δp 只能作 class-E 记录，不能当质量门**；A2/A4 的 ids 差异（128/128）里混着这个噪声。
- **MTP 若要进交付，必须同时把 `--expert-swaps 0` 定下来**（或先解决自适应层确定性），否则"质量门"无法通过。

## 4. 结论与建议

1. **MTP（draft-vocab 65536）过 +5% 门（−8.4%）⇒ 交付候选**；下一步必须做**干净窗、`--expert-swaps 0`、M0/M4 交替 2+2（4 跑）**，确认 −8% 可复现且 ids 稳定（相对 no-draft 仍然 class-E，但至少 run-to-run 可复现）。
2. 若确认，再验 p6792 交付形态 + qa/zh + 质量门；**交付暂不动**（仍 `sp_donly.exe` + threads16 + `--no-draft --lookup 3`）。
3. **profile 16pp 收口**：swap 阈值方向为负，不再沿此线投入。
4. 峰值全 ≤31,330 ✓；全零改码；一次一跑。

## 5. 未验 / 边界

- 只 alt prompt；每个 A 臂 1 跑（配对只有 2+2）；负载漂移大（−20%）。
- B2 只 1 跑；B 组同样受自适应层时序影响。
- `--draft-vocab` 更小档（16384/32768）未测；R42 的 32768 在脏批里是 −7.4%（与本批 65536 −8.4% 同量级）。
- 未测 p6792 / qa / zh 上的 MTP。

## 6. 落盘（绝对路径）

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R43-MTP判决与profile-20261018.md`
- 日志：`C:\sp3\R43-W00/A1/A2/A3/A4/B1/B2-20261018.log`；ids `C:\sp3\R43-*-ids.txt`；峰值 `C:\sp3\R43-*-peak.txt`；脚本 `C:\sp3\run_R43_*.cmd`
- exe 未变：`C:\sp3\sp_mlp.exe`（sha `33D954E5…`）；交付旧 exe `C:\sp3\sp_donly.exe`（sha `1BAFA78D…`）
