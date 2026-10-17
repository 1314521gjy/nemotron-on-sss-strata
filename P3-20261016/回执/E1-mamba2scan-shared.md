# 回执 · E①：`mamba2_scan` state→shared（`STRATA_M2_SCAN_SMEM`）— 2026-10-17 10:0x

- **任务**：把 `mamba2_scan` 的递推 state 从 global/L1TEX 搬进 shared（队长批准的第一步，一次一变量）。
- **状态**：**达成**（改码 + 编译 + 正控 + 负控 + N-A + 正确性 + ≥3 跑全做完）。
- **结论一句**：**`mamba2_scan` 单 kernel Duration 38.62→3.36 ms（≈11.5×），端到端 prefill 348.2→728.8 tok/s（+109%）；`--top-probs`/ids 与旧逐位一致（N-A 能变红）；decode −2.4%（rows=1 摊不开 load/store，机制清楚）。**

## 1. 改动（只动存放位置）

- `src/poly/kernels.cu`：`mamba2_scan_kernel` → **`template<bool SMEM>`**；SMEM 分支把每线程 `d_state` 宽的 state 行 load 进 **dynamic shared**（布局 `i1*(d_state+1)`，**+1 padding 消 32 路 bank 冲突**），**循环内读写 shared、末尾写回 global**。
- **算术顺序逐字不变**（同一 `st` 升序、同一表达式）；**未新增 `__syncthreads`**；**launch grid/block 不变** `<<<n_head=128, head_dim=64>>>`（只多一个 smem 尺寸参数）。
- **旗标**：**env `STRATA_M2_SCAN_SMEM=1`**（默认关；未碰 main.cpp/engine 选项管线）。**不碰 §7 四处** ✓。
- **★ 回退响亮**（按队长令）：若 `need > 设备 optin 上限` ⇒ **once 打印 `WARNING: STRATA_M2_SCAN_SMEM=1 but dynamic shared needs N KiB > device limit M KiB - FALLING BACK ...`**。**本靶 33.02 KB < 48 KB ⇒ 走 fast path，未触发回退告警。**
- **编译**：清 `kernels.cu.obj` → `cmake --build build-p3 --target strata-poly` ⇒ `[79/79] Linking` EXIT=0；`strata.exe`/`mmvq_il_parity.exe` **未动**。

### ★ sha（重要口径）
- 旧 exe = `978d64bc01d5b77f3bb3318a3000eee83351ca7cacc3c38d3614638898700bb7`（存为 `C:\sp3\sp_old.exe`）。
- 新 exe（**最终交付**）= **`2719c3dfb27e790712c2af8b1cdbe4b3a0687fd0de7143de12356af38a5ea759`**。
- **★ MSVC 嵌构建时间戳 ⇒ sha 不可复现**：同一源码连编两次得 `64d3bff4…` 与 `2719c3df…`。**用功能对拍（ids/top-probs）而非 sha 相等来验"回退干净"**（已验：N-A 残留无、ids/md5 回到基线）。**读数以最终 sha `2719c3df…` 为准。**

## 2. 正控（PASS）

| 项 | 旧 | 新 SMEM on | 倍 |
|---|---|---|---|
| **`mamba2_scan` Duration（ncu）** | 38.62 / 37.08 / 35.64 ms | **3.36 / 2.50 / 3.36 ms** | **≈11.5–15×** |
| Compute (SM) % | 2.98 | **40.50** | |
| L2 Cache Throughput % | 84.81 | **2.68** | |
| DRAM Throughput % | 0.84 | 3.47 | |
| Memory GB/s | 4.77 | 18.80 | |
| Registers/thread | 47 | 40 | |
| Dynamic Shared | 0 | **33.02 KB** | |
| Grid×Block | 128×64 | 128×64（不变） | |
| Achieved Occupancy % | 6.87 | 6.66（仍低 ⇒ ② 是下一门） | |

## 3. 负控（PASS —— 关掉回旧读数）

- **ncu（新 exe、env off）**：Duration **35.70 / 36.43 ms** · L2 **85.1%** · Compute **2.99%** · **Dynamic Shared 0** · Occupancy 6.87% ⇒ **回到旧读数** ✓。
- **端到端 env off**：prefill **348.2**（见 §5），与旧 `332.8~350` 同档 ✓。

## 4. 正确性（P1 ids / P2 top-probs）—— PASS（在 `--prompt-batch 8` 下）

- 口径：同 prompt（`tok700.txt`）、同 `--max-new 8`、greedy、冻结 profile、**不带 `--dump-*`**；`--top-probs 5`。
- **ids**：**OLD == NEW-OFF == NEW-ON = `1531,6330,1307,23716,1395,1605,1261,6330`** ✓
- **top-probs（8 步）md5**：**OLD == NEW-OFF == NEW-ON = `45a83075b381166782fc2cc1e839af98`** ✓ ⇒ **逐位一致**（6 位有效数字内）。
- **★ N-A 负控（判据能变红）**：把 SMEM 分支临时改 `st < d_state-1`（少加 1/128）⇒ **ids 变 `1531,48365,2655,1584,1605,2156,1317,3180`、top-md5 变 `42b5547c…`** ⇒ **判据有鉴别力** ✓。**临时改动已 revert**（源码无 `N-A TEMP`/`d_state - 1` 残留；回退后功能对拍回基线）。
- **★ 为什么用 pb8 而非 canonical pb1024**：**canonical `--prompt-batch 1024` 下 logits 是 NaN**（ids 全 0、top-probs 全 nan），**旧 exe 同样** ⇒ **该路径无法用于正确性对拍**（pre-existing，另案查；不影响性能可比性）。见此件 §6。

## 5. 性能（≥3 跑 + 预热 1 丢弃；canonical，pb1024）

| 臂 | prefill ×3 | **中位** | MAD | 极差 | decode ×3 | 中位 |
|---|---|---|---|---|---|---|
| **NEW-ON** | 725.6 / 731.7 / 728.8 | **728.8** | 0.40% | 6.1 (0.84%) | 54.58 / 55.00 / 55.75 | **55.00** |
| **NEW-OFF** | 347.9 / 348.2 / 348.9 | **348.2** | 0.09% | 1.0 (0.29%) | 56.76 / 56.13 / 56.36 | **56.36** |

- **prefill：348.2 → 728.8 = +109%（2.09×）** ⇒ **E1 通过**。
- **decode：56.36 → 55.00 = −2.4%** ⇒ **略超噪声**。**机制**：decode `rows=1`，shared 版每 call 多一次 128 load + 128 store（摊不开）⇒ 轻微变慢。**E2 严格照判据算"在 s_round 外下降"** ⇒ **如实标出，交队长判**（不改旗标默认关）。
- **E4 读数链自检**：kernel 份额 67.9%→≈8.7%、端到端 +109% ⇒ **读数链一致（未断）** ✓。
- 专家：**36 swapped / 0 left out / 命中 97.1%** 与旧一致 ⇒ **无静默退化** ✓。

## 6. ★ 重要：`--prompt-batch 1024` 路径 NaN（pre-existing，另案）

- `--prompt-batch 1024`（canonical）⇒ **logits NaN ⇒ ids 全 0、top-probs 全 nan**；**旧 exe（978d64bc）完全相同**。
- 同一 700-token prompt 改 `--prompt-batch 8` ⇒ **正常有限**。短 prompt + pb8 也正常。
- ⇒ **NaN 绑定 `--prompt-batch 1024` 的批量 prefill 路径**，**非本改引入**。**性能数在新旧同一坏路径上对比仍成立；但"这数算得对不对"是另一件事**（队长已令：先收 ①，再查 NaN）。

## 7. 未验 / 待
- **NaN 根因**（首选假设：pb=1024 > prompt 700 ⇒ 尾部填充行未初始化；4 跑矩阵在办）。
- **decode −2.4%** 是否值得"仅 prefill 开"的自动策略（未做，需另批）。
- **②（并行度）**：① 后 kernel 仅 ~3 ms / prefill ~8.7% ⇒ 收益上界缩小；队长倾向 NaN 定案后再判。

## 8. 落盘（绝对路径）
- 本件：`…\P3-20261016\回执\E1-mamba2scan-shared.md`
- 代码：`PolyStrata\src\poly\kernels.cu`（template + env 旗标 + 响亮回退）
- exe：`PolyStrata\build-p3\strata-poly.exe`（sha `2719c3df…`）· `C:\sp3\{sp_old.exe, sp_new.exe, sp_na.exe}`
- ncu：`C:\ncu\{m2scan_smem.ncu-rep, m2scan_neg.ncu-rep, *_report.txt}`
- 运行日志：`C:\sp3\{c8_old,c8_off,c8_on,na,rev_check,perf_on_1..3,perf_off_1..3,perf_warm}.log`
- 编译：`C:\sp3\{build_e1.log, build_na.log, build_rev.log}` · `…\P3-20261016\build_sp.cmd`
