# 情报-AI · CPU 读 + PCIe DMA 并发带宽（"预取/换腿"生死判据）— 2026-10-18 · builder3

**判决（按 lead 预注册判据）：比值 (C+P)/max = 1.358 ≥ 1.25 ⇒ DRAM 有余量、预取有窗口（理论量级 +20~25%）。**
但须附带一个测法 caveat（见 §3）：本测试的 CPU 腿绝对带宽（21.2 GB/s）低于引擎实测的 pinned sidecar 读聚合（41.3 GB/s），比值方向可信、绝对上限仍需按引擎读法复核。

## 1. 测法

`C:\sp3\staged-dram-pcie-concurrent.cu`（零引擎改动，单文件；`C:\sp3\staged-dram-pcie-concurrent.exe`）：
- **C 腿**：16 线程从 **2 GiB `cudaHostAlloc` pinned 缓冲**顺序 `memcpy` 读（每线程独立 span，1 MiB 块），持续 4 s；
  首版曾用 `ReadFile()` 读 sidecar（page-cache 路径，C 单独仅 **11.44 GB/s**），因不代表 pinned 读被替换。
- **P 腿**：64 MiB pinned → 64 MiB device，`cudaMemcpyAsync H2D` 在 non-blocking 流上连续排队，4–7 s；
- **C+P**：同一墙钟窗口两腿同时跑，各自计自己的字节数与耗时。
- 峰值：context + 64 MiB device + 2 GiB pinned host（显存「<2 GB」，满足断言）。

## 2. 读数（`C:\sp3\evidence\concurrent-bw-pinned-20261018.txt`）

| 档 | 读数 |
|---|---|
| **C 单独**（16 线程 pinned 直读） | **21.22 GB/s**（85.10 GB / 4.01 s） |
| **P 单独**（H2D DMA） | **22.98 GB/s**（159.79 GB / 6.95 s；与历史 ~23.2 GB/s 一致） |
| **C+P 并发** | C = **17.64**、P = **13.57**，合计 **31.21 GB/s** |
| **比值 (C+P)/max(C,P)** | **1.358** |
| 判据 | **≥1.25 ⇒ PASS：DRAM 有余量、预取窗口（理论 +20%+）** |

- 首版（ReadFile CPU 腿）对照：C 11.44 / P 22.84 / C+P 28.40，比值 1.243 ⇒ 落在「1.10–1.25 微弱窗口」；该版 CPU 腿走 page-cache 路径，**不作结论**，仅留档 `C:\sp3\evidence\concurrent-bw-20261018.txt`。

## 3. Caveat（必须与结论一起读）

1. 本测试的 CPU 腿绝对带宽 21.2 GB/s **低于**引擎里 pinned sidecar 读的实测聚合 **41.3 GB/s**（R69 ledger `down=2009.4ms … 41.3 GB/s`）。两者读法不同（引擎 16 worker 读 pinned 大块 + 交织计算），**比值的"方向"（DRAM 有 >25% 余量）可信，绝对总量（31 GB/s）不能直接当引擎上限**。
2. 若 CPU 腿按引擎读法跑到 41 GB/s，PCIe 腿并发时总带宽的实际上限应重新测（可能仍在 45–60 GB/s 区间，也可能被 DRAM 控制器瓶颈压住）——**建议下一步：把本测试的 CPU 腿换成引擎同款读法（16 worker 读 pinned sidecar 块，不计算），复测并发比值**，约 1 编译 + 1 跑。
3. 该结果**只说明"同时读+搬"有带宽余量**，与 R72 的"CPU 专家腿单腿带宽受限（37.5 µs/use）、PCIe 分流无窗口（≤5%）"不矛盾：R72 问的是"换腿能否省 CPU 腿时间"（答案否），本测试问的是"并发时 DRAM 是否还有余量"（答案：有）。

## 4. 对两条线的结论

- **RouterLookahead 预取 / ForesightSwap（把 miss 搬到 VRAM 让 GPU 算）**：**有窗口（比值 1.358）**，可以立项进入下一步设计；先按 caveat 2 用引擎读法复核一次并发比值，再决定工程量。
- **"边搬边算"在 CPU 腿原地算的形态**：仍无窗口（R72：CPU 腿本身带宽受限，PCIe 搬入额外占 DRAM）。
- 证据：`C:\sp3\evidence\concurrent-bw-pinned-20261018.txt`；程序 `C:\sp3\staged-dram-pcie-concurrent.cu`；构建日志 `C:\sp3\build_concurrent-bw.log`。
