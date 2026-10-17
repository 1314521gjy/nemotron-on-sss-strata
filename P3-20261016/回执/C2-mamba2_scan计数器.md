# 回执 · C2 —— `mamba2_scan_kernel`（prefill）ncu 计数器 — 2026-10-17 09:4x

- **任务**：对 C1（=N1 sm48 rep）里占比 top-1 的 `mamba2_scan_kernel` 采 ncu 计数器，判瓶颈是 **①「state 搬 shared（带宽）」** 还是 **②「chunked scan（并行度/延迟）」**。
- **状态**：**达成**。
- **结论一句**：**都否 —— 既不是 DRAM 带宽（0.84%），也不是并行度不足那么简单**：**真因 = `mamba2_scan` 的 128 个 block 各自【串行扫 700 步序列】，warp 数只剩 1/调度器，暴露的延迟里 80.6% 是【L1TEX/global 长记分板】，且访问【严重非合并（4.1/32 字节）】**。⇒ **②"chunked scan"方向对（要造并行度）**，但 **①的"state 进 shared"同样命中那 80.6% 的长记分板**（两者互补，非二选一）。

## 1. 命令与口径（A 级）

- **命令**：`ncu.exe -k regex:mamba2_scan --launch-count 3 -f -o C:\ncu\m2scan --target-processes all --set full C:\sp3\strata-poly.exe <canonical: pb1024 sa128 sm48 cpu-share0.55 swaps64 tail-skip0 threads24 vram-reserve2500 no-draft lookup0 expert-profile p_run.txt>`（cwd/exe/`-o` 全 ASCII）。
- **命中**：**3 个 launch，Duration 38.62 / 37.08 / 35.64 ms** ⇒ **全是 prefill 大调用**（非 39 µs 的 decode 小调用）✓。
- **★ 报数口径**：**ncu 的【时间】不进份额**（replay 改时序；日志里 prefill 显示 13.5 tok/s 即被拖慢）。份额只认 nsys（`回执\N1挂死…` 同批的 `n1_sm48` 分段）。

## 2. 计数器（3 次 launch 一致，取第 1 次）

| 项 | 值 | 读法 |
|---|---|---|
| **Duration** | **38.62 ms** | 单次 prefill scan |
| **Grid × Block** | **(128,1,1) × (64,1,1)** | **仅 128 block / 80 SM = 0.08 wave/SM** |
| Registers/thread | **47** | |
| Static smem/block | 4.11 KB | |
| **Compute (SM) Throughput** | **2.98%** | 算力几乎全闲 |
| **Issue Slots Busy / SM Busy** | **1.10%** | |
| Executed IPC | **0.04** | |
| **Memory Throughput（SOL）** | **84.81%** | = **L2 Cache Throughput 84.81%**（不是 DRAM！） |
| **DRAM Throughput** | **0.84%** | **显存带宽几乎没用** |
| L1/TEX Cache Throughput | 56.82% | |
| Mem Busy / Max Bandwidth / Mem Pipes Busy | 49.3% / 84.81% / 2.98% | |
| L1/TEX Hit Rate / **L2 Hit Rate** | 51.46% / **98.91%** | 工作集**在 L2 里**（非缺数据） |
| Memory Throughput(绝对) | **4.77 GB/s** | 极低 |
| **Achieved Occupancy** | **6.87%** | Theoretical 79.17% |
| **Waves Per SM** | **0.08** | |
| Achieved Active Warps/SM | **3.30** | |
| **Active Warps Per Scheduler** | **1.00**（上限 12） | |
| **No Eligible / One-or-More Eligible** | **98.67% / 1.33%** | |
| Issued Warp Per Scheduler | 0.01 | 每 **75.4 cycle** 才发一条 |
| **Warp Cycles Per Issued Inst** | **75.25** | |
| **最大 stall 类型** | **scoreboard 依赖 L1TEX(global/local/surface/tex) = 60.6 cyc = 80.6%** | **长记分板（访存延迟）** |
| 访存合并 | **global load 4.1/32 字节 · store 4.0/32（72% sector 未命中 L1TEX）** | **非合并（stride）** |

## 3. 判读（对队长的 ①②③）

- **③（`stall_barrier` 第一）**：**否** —— 最大 stall 是**长记分板**，不是 barrier。
- **①（state 搬 shared）**：**部分命中**。最大 stall 正是 **L1TEX/global 长记分板（80.6%）**，而 **L2 命中 98.91%** ⇒ 数据在 L2、**痛在"每次访问都要等"**。把**递推 state 放进 shared** 正好砍这条延迟依赖。**但**：**DRAM 只 0.84%**，所以**不是"显存带宽"问题** —— 任务单原述"若这条高⇒带宽问题"里的"这条"是 **L2 吞吐 84.81%**，是**被少数 warp 的非合并小请求顶上去的**，不是真带宽上限。
- **②（chunked scan）**：**方向对**。`Grid=(128)` = 每 head 一个 block，**每 block 串行扫 700 步**（序列依赖）⇒ **并行度只有 128 block（0.08 wave/SM、1 warp/调度器、No-Eligible 98.67%）**，**无法隐藏任何延迟**。**chunked（分段+前缀）** 才能把 block 数拉起来。
- **★ 我的判读（供你定夺；标"待你判"）**：**主因 = 并行度被序列依赖锁死（②）**；**次因 = 每步访问 state 走 global/L1TEX 且非合并（① + 合并优化）**。**两者互补**：②造并行度、①+合并把每条访存延迟/扇区数压下去。**单独做①（若只搬 shared 不增并行）可能只吃掉那 80.6% 的一部分**；**单独做②不治非合并访存**。
- **另有一条便宜线索**：**非合并访存（4.1/32 字节）** 框内 ncu 估 **~49%/53% 局部加速**（load/store）—— 是①/合并路线的直接靶子。
- **覆盖/残差**（nsys sm48 混合榜）：**top-8 覆盖 ≈91.3%，残差 ≈8.7%**（`mamba2_scan` 一项即 51.4%）。

## 4. 未验
- **①/② 各自的真实增量**：需接线后的 A/B（正控+负控），**未做**。
- **decode 侧 `mamba2_scan`（41 µs）** 的计数器：未采（占比小）。
- **ncu × CUDA graph** 共存：本次 `--set full` 下 replay 正常出数（**未破坏**），但**未两档对照**。

## 5. 落盘（绝对路径）
- 本件：`…\P3-20261016\回执\C2-mamba2_scan计数器.md`
- `C:\ncu\{m2scan.ncu-rep, m2scan_report.txt, m2scan.log}` · `C:\ncu\cell.ncu-rep`（A2）
- 相关：`C:\nsys\n1_sm48.{nsys-rep,sqlite}` · `C:\nsys\split.py`
