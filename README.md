> ## ⚠️ WIP — 施工中，暂无成品方案，有思路可参与研究
> 本项目**处于施工阶段**；现有读数与结论均为**过程证据**，不构成交付承诺。
> 有思路 / 想一起研究，欢迎参与（issues / 讨论；仓内 `P3-20261016/` 有完整路线与判据）。
> 主结论只对「RTX 4080 SUPER 32 GB + 96 GB RAM + i7-13790F」这一套硬件成立；
> **换显卡前请先读 §4「其他显卡的适配方向」**——参数不是照抄就有同样效果。

# Nemotron on SSS-Strata

本仓是 2026-10-15→18 **Nemotron-on-Strata** 实验的**研发路线、判据、注意事项与回执存档**；
源码树 `PolyStrata/` 与本文档**同在本仓**（2026-10-18 单仓合并；上游历史保留在 `PolyStrata/.git-upstream`）。

目标：把 **NVIDIA Nemotron-3.5-Super-VL-120B-A12B（Q4_K_M，70.1 GB）** 跑在**单张 32 GB 消费卡**上——
专家权重 pin 在内存（59.1 GiB）、按路由把用到的专家流式搬进 VRAM、未驻留专家由 CPU 分担。

---

## 1. 当前交付参数（**唯一推荐**，本机实测）

```bat
set STRATA_PIN_ARENA_GIB=59.1
set STRATA_M2_SCAN_SMEM=1
set STRATA_MAX_PROMPT_BATCH=8192
set STRATA_SCRATCH_TILE=1
set STRATA_STAGE_RING2=1
set STRATA_PROMPT_BORROW=1
set STRATA_SWAP_DIRECT_FROM_PIN=1
set STRATA_PROFILE_AUTO_LEARN=1
set STRATA_AHEAD_EXACT=1

sp_seeded.exe --model <gguf> --tokens-file <ids> \
  --max-context 1048576 --prompt-batch 6912 --stage-ahead 128 --stage-mib 2048 \
  --cpu-share 0.55 --expert-swaps 64 --tail-skip 0 --prompt-cache-every 8192 \
  --threads 16 --no-draft --lookup 3 --max-new 128 --vram-reserve-mib 3000 \
  --expert-profile C:\sp3\p_autolearn.txt
```

| 指标 | 值 | 起点（10-17 末） |
|---|---|---|
| prefill | **1,754.5 tok/s** | 652.6（+169%） |
| decode（模板型 prompt） | **41.32 tok/s** | 22.3（+85%） |
| 专家命中率 | 76.4–76.7% | 58.4% |
| miss/token | 207.7 | 373.3 |
| 显存峰值 | 31,362 MiB（门 31,700） | — |
| 输出一致性 | ids 与旧基线**逐位相同** | — |

**异构（问答/代码类）prompt：decode ≈18–20 tok/s。** 模板型负载会显著更快——**别把模板数字当通用值**。

## 2. 关键机制（都默认关，交付脚本显式打开）

| env / flag | 作用 | 本机实测 |
|---|---|---|
| `STRATA_PROMPT_BORROW=1` | 预填借用专家缓存尾部槽给 staging | decode +11.3% |
| `STRATA_SWAP_DIRECT_FROM_PIN=1` | 换槽直接从 pinned 映射 DMA（免 host 拷贝） | +4.4~9.9% |
| `STRATA_AHEAD_EXACT=1` | 预取容量按专家实际布局（gated 4.43 → 实际 2.95 MB） | prefill +3.2% |
| `STRATA_PROFILE_AUTO_LEARN=1` | 首跑自动建 decode-only 专家表、次跑自动加载并继续学 | decode +38.4%（首跑略慢） |
| `--max-context 1048576` | VMM 按需映射；上限 ≠ 预分配 | 峰值仅 +3 MiB |

**专家表**：随包 `P3-20261016/交付/p_shipped.txt`（170,796 B / 16,900 行）。安装时复制为可写工作表：

```bat
copy p_shipped.txt p_autolearn.txt
```

## 3. 目录

- `P3-20261016/` —— P3 阶段主台账、判据、情报、施工单、`回执/`（逐轮交付回执）、`验收/`、`交付/`（随包专家表与交付物清单）
- `P0-20261016/`、`P2-20261016/` —— 前置阶段（基座裁决 / 单元格校准）
- 根目录 —— 项目书、协作规则总纲、交接件、计划件、机制消化件
- `P3-20261016/git/` —— 建仓留痕（改动前 status、提交后 show --stat、GIT 台账）
- **`P3-20261016/下一棒交接-20261018-收工.md` = 唯一权威状态件**（全部读数、判否清单、纪律）

## 4. 其他显卡的适配方向（**换卡必读**）

参数不是"越大越好"，几乎每个都对应一个**本机实测过的悬崖**。换卡请按规则重算，不要照抄。

### 4.1 先算三个硬量

| 量 | 怎么算 | 本机值 |
|---|---|---|
| 专家库总量 | 专家数 × 每专家字节（Q4_K_M = 3.0966 MB） | 20,480 × 3.0966 MB = **63.4 GB** |
| VRAM 门 | 卡容量 − 1024 MiB（给桌面留 1 GB） | 32,768 − 1,068 = **31,700 MiB** |
| 每 token KV | 8 个 attention 层 × n_kv × head_dim × (K+V) × 2B | **8 KiB/token** |

### 4.2 参数随卡怎么变

| 参数 | 规则 | 为什么 |
|---|---|---|
| `--threads` | **= 物理核数**（不是逻辑核数） | 本机 16 物理 / 24 逻辑；用 24 时 miss/token 不变但 `cpu_us/miss` 96→128–157 µs ⇒ **decode −55%**（超线程互抢悬崖） |
| `STRATA_PIN_ARENA_GIB` | `min(RAM − 非专家部分 − 8 GiB, 专家库总量)` | 96 GB RAM ⇒ pin 59.1 GiB；RAM 更小就少 pin 几层，否则 `REFUSED` 或压死系统 |
| `--vram-reserve-mib` | **≈ 卡容量 9%** | 3000；2600 只涨 1% 驻留但峰值贴门（余 261 MiB），2400 直接崩 |
| `--stage-mib` | 会自动钳到 **512 专家**（≈1512 MiB），更大无效 | 一层最多 512 个被路由专家，一轮装得下即够 |
| `--max-context` | 可直接开到任务上限（VMM 按需） | 峰值只随**实际用到**的长度长：128K 花 1 GiB、1M 花 8 GiB。**长上下文是从专家槽里扣的** |
| `--prompt-batch` `--prompt-cache-every` | 有 **43 GB 重复预填墙** | 本机 6912 / 8192；更大卡可试 8192/8192，盯 prefill 的重复字节 |
| `--expert-swaps` | 64（1..64） | 32 时 −5.1% |

### 4.3 换卡时先看两件事

1. **VRAM 能不能装"够用"的专家**：槽数 ≈ (VRAM 门 − 稠密权重 ~5 GB − staging 1.5 GB) / 3.0966 MB。
   本机 ≈ 6,746 槽 = 覆盖 32.9% ⇒ 命中 76%；**24 GB 卡约 4,600 槽（覆盖 22%）⇒ 命中掉到 ~60%**，decode 明显变慢——
   **这是容量问题，不是参数问题**。
2. **内存带宽**：CPU 未驻留专家是**带宽受限**（本机 16 线程聚合 ≈41 GB/s，单流上限 ≈32 GB/s）。
   换平台时 `--threads` 与内存通道数要一起看：**通道减半 ⇒ 命中再高也没用**。

### 4.4 精度的适配

- 本仓实测均基于 **Q4_K_M**（up=Q4_K 1.548 MB + down=IQ4_NL 1.548 MB = 3.0966 MB/专家）。
- 换更低位宽（IQ3_S ≈2.37 MB、Q2_K ≈1.79 MB）同显存能装更多专家（+31%/+73% 槽），但**有损**；
  **本仓所有"位精确/无损"结论届时全部失效，必须重测 ids 与文本质量**。
- KV 量化（q8/q4）：16K 上下文只省 64 MiB（≈21 槽）**不值得**；只有 **≥128K** 才值得（1M 时 q8 省 4 GiB ≈ +1,322 槽）。

## 5. 已在"判否清单"（**别再试**，都有实测读数）

`MTP 投机`（受控 −1.2%；历史上三次 +21% 均**不可复现**）· `shared expert 独立流`（+1.7%，低于门槛）·
`snap 快照 3→1`（有窗口时 keep() 重算不位精确）· `GPU 侧 routing`（CUDA/Host `expf` 无位一致保证）·
`RouterLookahead / ForesightSwap`（相邻层路由重合率 **4.23% ≈ 随机 4.30%**）· `KV 进内存`（依赖模型自带的 QSA indexer，我们没有）·
`pcie 分流`（direct 模式 −23%）· `tail-skip`（有损）· `per-layer 槽配额`（0 增益）·
`专家组预取/缓存`（带宽净亏）· `head alias`（真杠杆其实是预取分母）· `ReLU² down 重排`（T 内核延迟受限）·
`异构图模式`（alt +8.6% 但 qa −13.8%，不通用）

完整 55+ 条与逐条读数见 §3 交接件。

## 6. 使用注意

1. 本仓只收**文档**与**小脚本**；日志、模型、exe、构建产物、实验数据一律不入库（`.gitignore` 有逐类清单）。
2. 结论的证据等级与"怎么变红"写在各自文档里；**未验条目标 `未验`**。引用行号前先核对 `sha256`（源码在动）。
3. 源码 `PolyStrata/` 原本是独立 git 仓库，现已并入本仓（上游历史在 `PolyStrata/.git-upstream`）。
4. **任何涉及专家表的 A/B 必须冻结表**（`STRATA_PROFILE_AUTO_LEARN=0` + 四臂同一只读表 + 自证 `saved=0`），
   否则表会在批次里漂移，产生**假阳性**（本仓踩过一次：看似 +21.9%，受控后是 −1.2%）。
