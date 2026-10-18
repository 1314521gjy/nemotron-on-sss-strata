# R39 · PCIe direct（零拷贝原地读）A/B — 2026-10-18

- **席位**：builder2 · alt prompt（293 tok）· threads 16 · `--expert-swaps 0 --max-new 64` · 一次一跑 · 峰值 ≤31,700 · 未 spawn 子代理。
- **结论**：**direct 模式生效（自证行 + DMAs=0/GB=0）**，且**比 dma 同 frac 好 +16.9%**（13.06 vs 11.17）——但**绝对提升只有 +5.6%（中位），未达 ≥15% 采纳门**；`cpu_us/miss` 仍 +7.4%（未降）⇒ **"多一次搬运"不是全部病因，剩余成本是 CPU/GPU 争抢与每层同步**。**不采纳进交付**，direct 保留为 opt-in（默认仍 dma）。
- exe：sha256 **`2356C878A2230BB5C0E8E3BF9CCAA41F8040536922B1131F90BE88ECB88CEEA7`**（`build-p3\strata-poly.exe`，副本 `C:\sp3\sp_direct.exe`）；旧交付 exe 未变（`sp_donly.exe`）。

---

## 1. 实现（本包唯一改动，默认值未动）

| 文件 | 改动 |
|---|---|
| `src\poly\engine.hpp` | `Options::pcie_mode`（默认 `"dma"`）；`Stats::pcie_direct`；`pcie_direct_`；`exp_up_dev_/exp_down_dev_`（pinned 区域的**设备别名**） |
| `src\poly\engine.cpp` | ① `--pcie-mode dma|direct|auto` 校验；② pinned 区域在 direct/auto 时加 **`cudaHostRegisterMapped`**（只 `ReadOnly` 时设备地址取不到）；③ 启动探测 UVA + `cudaHostGetDevicePointer`，把每个 pinned 层的 up/down **设备别名**缓存下来（实测设备地址与宿主地址不同：dp=0x205000040 vs host=0x2441FC62040 ⇒ 必须用设备别名，不能直接用宿主指针）；④ direct 时不分配 staging/stream/event，split 的专家**零拷贝**：`pcie_src_up/down = 设备别名 + e*stride`，不计数 DMA；⑤ split 的 `Cached` 加 `down` 指针，kernel 用 `c.down ? c.down : c.slot + slot_down()` |
| `src\poly\main.cpp` | `--pcie-mode` 解析/usage；decode 行打印 `(DMAs, GB, N direct)` |

- 编译：删 poly `*.cpp.obj` 后一次主构建 EXIT=0（`C:\sp3\build_r39-20261018.log`）；调试 direct 探测另修 2 次（`build_r39b/c`，同一包内）。**默认 `--pcie-mode dma`、`--pcie-frac 0` 未动。**

## 2. 读数（同批；环境仍偏慢：E0 12.26 vs 干净窗 16.63，但批内可比）

| 臂 | 配置 | decode | vs E0 | miss/token | pcie experts/token | pcie GB/token | DMAs | direct | cpu_us/miss | cpu_busy | peak MiB | ids vs E0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E0** | `--pcie-frac 0` | 12.26 | — | 496.7 | 0 | 0 | 0 | 0 | 129.0 | 118.9 | 31,020 | — |
| **E1** | 0.2 **dma** | 11.17 | **−8.9%** | 508.7 | 97.37 | **0.297** | 12,268 | 0 | 141.2 (+9.5%) | 122.1 | 31,021 | 64/64 |
| **E2** | 0.2 **direct** | **13.06** | **+6.5%** | 422.6 | 76.06 | **0.000** | **0** | **4,792** | 138.5 (+7.4%) | 124.1 | 31,001 | 43/64 |
| **E3** | 0.35 **direct** | 12.26 | 0.0% | 344.5 | 152.17 | **0.000** | **0** | 9,587 | 152.5 (+18.2%) | 134.6 | 31,002 | 1/64 |
| **E4** | E2 重复 | 12.84 | +4.7% | 422.6 | 76.06 | 0.000 | 0 | 4,792 | 142.7 | 127.5 | 30,997 | ==E2 ✓ |

**E2 中位 = (13.06+12.84)/2 = 12.95 ⇒ +5.6% vs E0**（采纳门 ≥15% **未达**）。
**同 frac 下 direct vs dma：13.06 vs 11.17 = +16.9%**（direct 确实省掉了搬运/等待）。

## 3. 关键判据（lead 的三条）

| 判据 | 结果 |
|---|---|
| 启动自证行 `--pcie-mode direct: … zero copy, no staging slots or copies` | **✓ 已打印**（E2/E3/E4） |
| `pcie_us/miss`≈0、staging DMAs≈0 | **✓**：DMAs=0、GB=0.00、`direct` 计数 4,792 / 9,587；`pcie_us/miss` 引擎口径字段恒 0（本包未改该打点），以 DMAs=0 为证 |
| `cpu_us/miss` 不应再 +29% | **+7.4%（E2）/ +18.2%（E3）**：比 dma 的 +9.5% 与 R37-D2 的 +29% 小很多 ⇒ **搬运成本确实省了**；但**仍未下降** ⇒ 剩余成本不是搬运，而是 **GPU 专家与 CPU 专家在同一层时间窗里互相争抢 + 每层串行同步**（GPU 份额越大，CPU 侧剩余 miss 的有效并行度越低：E3 miss −30.6% 而 decode 0%） |
| decode ≥15% | **✗ +5.6%** ⇒ 不采纳 |

**质量（类 E 记录，不判）**：E1 64/64 ids 变、max|Δp|=0.9999@pos34、5/64 >0.05；**E2 43/64 变、max|Δp|=0.9999@pos36、11/64 >0.05**；E3 1/64 变、max|Δp|=0.078@pos26、5/64 >0.05。均非无损（专家从 CPU 挪到 GPU 改变舍入路径，近分翻 token）。
**负控 ✓**：**E0（新 exe、默认 dma、frac 0）ids 与旧 exe `sp_donly` 的 R37-D0 逐位同（64/64）** ⇒ 新增代码在默认路径上无副作用。

## 4. 结论与建议

1. **direct 生效、且优于 dma**（同 frac +16.9%），但**离 2× 很远**：alt 上 +5.6%（中位），未达 15% 门。
2. **病因判定**：direct 把"staging 拷贝 + event 等待"消掉后，`cpu_us/miss` 仍升（+7.4%）⇒ 剩余瓶颈是 **CPU 腿与 GPU 腿的串行/争抢**（每层 CPU 专家算完才轮到该层残余；GPU 份额 ↑ ⇒ CPU 端 miss 摊销的固定同步成本 ↑）。**这条不是可用杠杆**，除非动"每层串行依赖"（线 C）。
3. **交付不动**：`--pcie-mode` 默认 dma、`--pcie-frac` 默认 0；direct 仅 opt-in（交付脚本仍 `sp_donly.exe`）。
4. 若 lead 想再压这条线：E3 显示 frac 0.35 已经"GPU 抢 CPU"（−30.6% miss 但 decode 0%）⇒ **份额甜点应在 0.1–0.2**；但按采纳门与"2× 杠杆"预期，建议**不投入**。

## 5. 未验 / 边界

- 环境仍偏慢（E0 12.26 vs 干净 16.63，−26%）；absolute 只在批内可比。
- 只 alt prompt；E2 只有 2 跑（中位）；E3 1 跑。
- `pcie_us/miss` 是引擎既有占位字段（恒 0），direct 的 PCIe 读带宽/延迟没有单独打点（只有 DMAs=0）。
- 未在 qa/p6792 上测 direct；未测 `--pcie-mode auto`（direct 可用时 auto = direct，逻辑同 E2）。

## 6. 落盘（绝对路径）

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R39-pcie-direct-20261018.md`
- 日志：`C:\sp3\R39-E0/E1/E2/E3/E4-20261018.log`；ids `C:\sp3\R39-*-ids.txt`；峰值 `C:\sp3\R39-*-peak.txt`；脚本 `C:\sp3\run_R39_*.cmd`
- exe：`C:\sp3\sp_direct.exe`（sha `2356C878…`）；交付旧 exe 未变 `C:\sp3\sp_donly.exe`（sha `1BAFA78D…`）
- 构建：`C:\sp3\build_r39-20261018.log`（+ `build_r39b/c` 调试）
