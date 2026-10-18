# R47 · `--host-core` / `--aux-cpus` A/B（漏做项 TOP-1）— 2026-10-18

- **席位**：builder2 · alt prompt（293 tok）· `--threads 16` · `--no-draft --lookup 0 --expert-swaps 0 --max-new 128` · 一次一跑 · 峰值 ≤31,700 · **零改码**（`sp_detwin.exe` sha `13CA1A03…`）· 未 spawn 子代理。
- **5 跑**：W 预热丢弃 + A/B/C/D。
- **判决：判红（大幅负收益）。`--host-core first/last`、`--aux-cpus auto` 在本机都让 per-token 慢 56–64%，`cpu_us/miss` 从 99.3 升到 152.5–159.2 ⇒ "已验无收益"，收口此项。ids 全臂与 A 逐位同 ✓。**

---

## 1. 读数（同批）

| 臂 | 配置 | per-token | vs A | decode | cpu_us/miss | cpu_busy | miss/token | peak MiB | ids vs A |
|---|---|---|---|---|---|---|---|---|---|
| W | 预热丢弃 | 68.7 ms | — | 14.55 | 106.0 | 98.6 | 516.9 | 30,961 | — |
| **A** | `--threads 16`（CTRL，off/off） | **64.5 ms** | — | 15.51 | **99.3** | 93.1 | 516.9 | 30,942 | — |
| **B** | `--host-core first` | **105.9 ms** | **+64.3%** | 9.44 | **159.2** | 140.7 | 516.9 | 31,033 | **✓ 0/128** |
| **C** | `--host-core last` | **101.8 ms** | **+57.9%** | 9.82 | **157.0** | 145.3 | 516.9 | 31,022 | **✓ 0/128** |
| **D** | `--host-core first --aux-cpus auto` | **100.8 ms** | **+56.4%** | 9.92 | **152.5** | 139.9 | 516.9 | 31,018 | **✓ 0/128** |

## 2. 自证行原文（钉核确实生效，不是旗标没被接受）

- A：`strata-poly: threads: --host-core off --aux-cpus off: host and workers not pinned, aux off`
- B：`strata-poly: threads: --host-core first --aux-cpus off: host cpu 0 (physical core 0 of 16), 15 workers on 23 cpus, aux off`
- C：`strata-poly: threads: --host-core last --aux-cpus off: host cpu 23 (physical core 15 of 16), 15 workers on 23 cpus, aux off`
- D：`strata-poly: threads: --host-core first --aux-cpus auto: host cpu 0 (physical core 0 of 16), 15 workers on 21 cpus, aux 2 cpus: 1 15`

## 3. 判据

| 判据 | 结果 |
|---|---|
| 采纳门 per-token ≥5% 且 ids 逐位同 | **✗ 全负（+56.4%~+64.3%）**；ids ✓（B/C/D 与 A 均 0/128） |
| 判红：ids 漂 或 `cpu_us/miss` 不降 | **`cpu_us/miss` 反升 +54~60%**（99.3 → 152.5–159.2）⇒ 判红 |
| `--host-core` 报错/不支持 | 无报错，自证行显示钉核成功 |

## 4. 为什么钉核反而大幅变慢

1. **本机是 16 物理 / 24 逻辑，`--threads 16` = host + 15 workers**：off 时 OS 让 16 个线程自由落在 24 逻辑上；一旦 `--host-core first/last`，host 被钉在**某一个物理核**，15 workers 被限制到 23/21 个 CPU 的掩码上——**host 线程本身要跑的活（launch、路由等待、CPU 专家分发）被固定在一个物理核上，且该核还要与 worker/外部进程抢**。
2. **外部负载敏感**：本机常驻 AweSun + 多个 DSH/qoderwake；钉核后无法利用空闲逻辑核 ⇒ 惩罚远大于 R24 在安静窗口测到的 −11.5%（`--threads 24` 时），这次 `--threads 16` 下是 +56~64%。
3. **`cpu_us/miss` 上升**说明是 CPU 腿实打实变慢（不是 GPU 侧），与"host 钉核 + workers 被挤到受限掩码"一致。
4. **老师的跑法不可移植**：`q38_perf.log` 的 `--host-core first + 15 workers + 16 线程` 是在他们的负载/调度环境里；本机实测同一形态负收益 56–64% ⇒ **不应照抄**。
5. 与 R24 的结论一致（当时已把 `Options::host_core` 默认从 `first` 改成 `off`，交付用 off）。

## 5. 结论

1. **`--host-core off`（现行默认）就是本机最优**；`first/last`、`--aux-cpus auto` 均无收益且大幅负收益 ⇒ **漏做项关闭（已验无收益）**。
2. 交付脚本**保持不加**这些开关（与 lead 审计一致）。
3. 峰值 30,942–31,033 全 ≤31,700 ✓；ids 全臂逐位同（钉核只改调度、不改数值）。
4. 唯一已采纳的收益仍是 **threads 16**（alt +28% / 交付形态 prefill 1938.8 / decode 26.25）。

## 6. 未验 / 边界

- 只 alt prompt；每臂 1 跑（负收益 56–64% 远超噪声，无需重复）。
- 未测 `--host-core <n>` 指定其它物理核、也未测 `--aux-cpus` 显式列表（按 off 已最优的结论，没有继续价值）。
- 本批机器有外部常驻负载（与 R24 安静窗口不同），所以绝对 per-token 与 R24 的 −11.5% 不可直接比，但**方向一致（钉核=负收益）**。

## 7. 落盘（绝对路径）

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R47-host-core与aux-cpus-20261018.md`
- 日志：`C:\sp3\R47-W/A/B/C/D-20261018.log`；ids `C:\sp3\R47-*-ids.txt`；峰值 `C:\sp3\R47-*-peak.txt`；脚本 `C:\sp3\run_R47_*.cmd`
- exe 未变：`C:\sp3\sp_detwin.exe`（sha `13CA1A03DAA4E0CCFC9F460DA5ABA3CAFE4EA5ACF66BDFBB0F1A29EE11DCAF59`）；交付旧 exe `C:\sp3\sp_donly.exe`（sha `1BAFA78D…`）
