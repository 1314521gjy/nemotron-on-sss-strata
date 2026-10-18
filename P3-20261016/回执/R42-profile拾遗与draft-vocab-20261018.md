# R42 · profile 拾遗（adapt-min-gain）与 draft-vocab 减负 — 2026-10-18

- **席位**：builder2 · alt prompt（293 tok）· threads 16 · `--expert-swaps 0` · `--max-new 128` · 一次一跑 · 峰值 ≤31,700 · **全零改码**（用现 exe `sp_mlp.exe`，sha `33D954E5…`）· 未 spawn 子代理。
- **6 跑**（批首 W00 预热丢弃 + M0–M4）。
- **判决**：**M1/M2 判红（旋钮未生效——被 `--expert-swaps 0` 关掉，非旋钮无效）；M3/M4 过"per-token 不变差"门（−7.4% / −12.5% vs M0），但本批有负载下降趋势，只能算"不再变差"，需干净窗复验。**

---

## 1. 读数（同批）

| 臂 | 配置 | decode | per-token | `experts from VRAM` | miss/token（use 口径） | cpu_us/miss | cpu_busy | cpuLeg/token | peak MiB | ids vs M0 |
|---|---|---|---|---|---|---|---|---|---|---|
| W00 | 预热丢弃 | 11.34 | 88.2 ms | 41.2% | 516.9 | 134.4 | 124.5 | 69.5 ms | 31,111 | — |
| **M0** | CTRL：`--no-draft --lookup 0` | **10.93** | **91.5 ms** | **41.2%** | **516.9** | 139.9 | 128.9 | 72.3 ms | 31,016 | — |
| **M1** | M0 + `--adapt-min-gain 1.2` | 11.52 | 86.8 ms | **41.2%**（不变） | **516.9**（不变） | 133.3 | 122.9 | 68.9 ms | 31,003 | **0/128 ✓**（Δp=0） |
| **M2** | M0 + `--adapt-min-gain 0.5` | 11.15 | 89.7 ms | **41.2%**（不变） | **516.9**（不变） | 137.4 | 126.8 | 71.0 ms | 31,023 | **0/128 ✓**（Δp=0） |
| **M3** | draft + `--window 8` + **`--draft-vocab 32768`** | 11.81 | **84.7 ms** | 29.0% | 694.6 | 108.9 | 97.8 | 75.6 ms | **31,370** | 128/128 不同 |
| **M4** | draft + `--window 8` + **`--draft-vocab 65536`** | **12.48** | **80.1 ms** | 32.2% | 705.9 | 100.0 | 90.1 | 70.6 ms | **31,367** | 127/128 不同 |

## 2. 判据

### M1/M2（profile 拾遗）——**判红（旋钮未生效，非旋钮无效）**
- **命中率 41.2% 完全不变、miss/token 516.9 完全不变、ids 逐位同（Δp=0）**，自证行 `adaptive tier: min gain 1.200 / 0.500` 已打印。
- **原因（读码）**：`adapt_min_gain` 只在**自适应层 swap 候选选择**里用（`engine.cpp:1436` `r >= opt_.adapt_min_gain`、`:1451` `in.recent < swap_ratio*out.recent + adapt_min_gain`）；而 `Engine::start_swaps()` 开头 `if (swap_ || cached_ == 0 || opt_.expert_swaps == 0) return;` ⇒ **本批按令的 `--expert-swaps 0` 把整条 swap 路径关掉了，旋钮没有作用面**。
- **要真正测 profile 拾遗（16pp）**：必须 **`--expert-swaps >0`（默认 64）**，届时 ids 可能漂（自适应层按时序发布槽位，R27 已证同配置不可复现）⇒ 用"命中率 + miss/token + per-token + Δp"判，不用 ids 硬门。建议另批 3 跑。

### M3/M4（draft-vocab 减负）——**过门，但被负载趋势污染**
- `--draft-vocab` **被接受**（32768 / 65536 都正常跑完），GUI 无报错；tokens/round **4.57 / 2.91**（兑现 90.9% / 82.4%）。
- **per-token 84.7 / 80.1 ms vs M0 91.5 ms = −7.4% / −12.5%** ⇒ 过"不再变差（≤+1%）"门。
- **CPU 腿没有变差**（M3/M4 75.6/70.6 ms vs M0 72.3 ms）⇒ 窗口带来的额外 CPU 工作被 **vocab 子集把 draft head 的 GPU 开销压下去**抵消（R40 全词表时 draft 是 **+0.9% 更差**）。
- **但**：本批有**负载下降趋势**（M0 cpu_us/miss 139.9 → M4 100.0；W00 88.2 → M4 80.1 ms），M3/M4 的绝对优势有一部分来自机器变轻，不能直接读成"净赚 7–12%"。⇒ 结论只到 **"MTP 不再有额外开销"**；要量化收益需**干净窗 M0/M4 交替 2+2**。

## 3. 质量（class-E 记录）

- M1/M2：**ids 与 M0 逐位同（0/128）、top-probs Δp=0**（无任何行为变化）。
- M3/M4：**ids 128/128、127/128 不同**；max|Δp|=**0.9998@pos34**（两者）、mean 2.59e-2 / 3.51e-2、**9/128 / 10/128 位置 >0.05**——窗口 batch 的舍入路径变化（与 R40 同型），不是 draft-vocab 特有。

## 4. 结论与建议

1. **M1/M2 这条测试设计自相矛盾**：`--expert-swaps 0` 让 `adapt-min-gain` 无作用面。**profile 16pp 的假设（`kSwapMin=2.0` 太保守）尚未被检验**，需 swaps>0 的批次。
2. **draft-vocab 是本轮唯一正向信号**：MTP 的额外开销被 vocab 子集压掉（per-token 至少不再变差，方向与 R40 相反）。建议下一批在**干净窗**做 `M0 vs M4` 交替 2+2（4 跑）确认；若 M4 稳定 ≤ M0，MTP 就具备进交付的前提（且它比全词表 draft 更省 VRAM/带宽）。
3. **交付不动**（`sp_donly.exe` + threads16 + `--no-draft --lookup 3`）。
4. 峰值 M3/M4 = 31,370 / 31,367 ≤31,700 ✓；M0–M2 = 31,016–31,023。

## 5. 未验 / 边界

- 只 alt prompt；M1/M2 各 1 跑；M3/M4 各 1 跑；本批机器负载由重转轻（cpu_us/miss 139.9→100.0），**跨臂绝对时间受趋势污染**。
- `--draft-vocab` 未做"低于 32768"的下探（如 16384）——如果再压 head，可能需要更小 vocab 才见更大收益（另批）。
- `--adapt-min-gain` 的正经测试（swaps>0）未做。

## 6. 落盘（绝对路径）

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R42-profile拾遗与draft-vocab-20261018.md`
- 日志：`C:\sp3\R42-W00/M0/M1/M2/M3/M4-20261018.log`；ids `C:\sp3\R42-*-ids.txt`；峰值 `C:\sp3\R42-*-peak.txt`；脚本 `C:\sp3\run_R42_*.cmd`
- exe 未变：`C:\sp3\sp_mlp.exe`（sha `33D954E5B5E662B400EAF3AE8D3941A323B82B207FE55CF73956FE6CA4D60DF6`）；交付旧 exe `C:\sp3\sp_donly.exe`（sha `1BAFA78D…`）
