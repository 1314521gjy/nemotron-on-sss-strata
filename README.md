> ## ⚠️ WIP — 施工中，暂无成品方案，有思路可参与研究
> 本项目**处于施工阶段**；现有读数与结论均为**过程证据**，不构成交付承诺。
> 有思路 / 想一起研究，欢迎参与（issues / 讨论）。
> 主结论只对「RTX 4080 SUPER 32 GB + 96 GB RAM + i7-13790F」这一套硬件成立；
> **换显卡前请先读 §4「其他显卡的适配方向」**——参数不是照抄就有同样效果。

# Nemotron on SSS-Strata

把 **NVIDIA Nemotron-3.5-Super-VL-120B-A12B（Q4_K_M，70.1 GB）** 跑在**单张 32 GB 消费卡**上：
专家权重 pin 在内存（59.1 GiB）、按路由把用到的专家流式搬进 VRAM、未驻留专家由 CPU 分担。
引擎是本仓 `PolyStrata/`（上游 PolyStrata 的 fork + 本仓改动）。

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

rem A. 异构 / agent 负载（问答、代码、多轮对话）—— 日常用这套
sp_seeded.exe --model <gguf> --tokens-file <ids> \
  --max-context 1048576 --prompt-batch 6912 --stage-ahead 128 --stage-mib 2048 \
  --cpu-share 0.55 --expert-swaps 64 --tail-skip 0 \
  --threads 16 --no-draft --lookup 0 --max-new 128 --vram-reserve-mib 3000 \
  --expert-profile C:\sp3\p_autolearn.txt

rem B. 模板型负载（重复模板 / 可被 n-gram 命中的续写）—— 只把 --lookup 换掉
    --lookup 3
```

### ⚠️ 两条硬规矩（都是踩出来的）

1. **绝对不要设 `--prompt-cache-every`**：它会按 N 个 token 打"会话检查点"供跨请求复用。
   我们是 **mamba 混合架构，递归状态无法像纯注意力 KV 那样存取** ⇒ **任何超过该边界的提示、以及之后的请求都会输出崩**
   （实测：7,306 token 正常 / 11,066 崩 / 去掉后 11,324、12,679、**20,191** 全部正确）。
   ⇒ 下表基准读数是 **6,792 token（< 8,192）** 上测的，所以当时没暴露；**通用配置一律不设这个旗标**。
2. **`--lookup` 要按负载选**：模板型 = `3`（n-gram 能抄到，窗口被填满）；异构/agent = `0`（猜不中，白花钱）。
   异构负载上关掉投机实测 **+9.6%**。

### 基准读数（模板型负载，6,792 token prompt）

| 指标 | 值 | 起点（10-17 末） |
|---|---|---|
| prefill | **1,754.5 tok/s** | 652.6（+169%） |
| decode（模板型 prompt） | **41.32 tok/s** | 22.3（+85%） |
| 专家命中率 | 76.4–76.7% | 58.4% |
| miss/token | 207.7 | 373.3 |
| 显存峰值 | 31,362 MiB（门 31,700） | — |
| 输出一致性 | ids 与旧基线**逐位相同** | — |

**异构（问答/代码类）prompt：decode ≈18–22 tok/s、prefill ≈1,100–1,400 tok/s**（长提示跨过缓存冷启动后）；
模板型负载会显著更快——**别把模板数字当通用值**。

## 2. 关键机制（都默认关，交付脚本显式打开）

| env / flag | 作用 | 本机实测 |
|---|---|---|
| `STRATA_PROMPT_BORROW=1` | 预填借用专家缓存尾部槽给 staging | decode +11.3% |
| `STRATA_SWAP_DIRECT_FROM_PIN=1` | 换槽直接从 pinned 映射 DMA（免 host 拷贝） | +4.4~9.9% |
| `STRATA_AHEAD_EXACT=1` | 预取容量按专家实际布局（gated 4.43 → 实际 2.95 MB） | prefill +3.2% |
| `STRATA_PROFILE_AUTO_LEARN=1` | 首跑自动建 decode-only 专家表、次跑自动加载并继续学 | decode +38.4%（首跑略慢） |
| `--max-context 1048576` | VMM 按需映射；上限 ≠ 预分配 | 峰值仅 +3 MiB |

**随包专家表** `交付/p_shipped.txt`（170,796 B / 16,900 行）。安装时复制为可写工作表：

```bat
copy 交付\p_shipped.txt C:\sp3\p_autolearn.txt
```

## 3. 目录

| 路径 | 内容 |
|---|---|
| `PolyStrata/` | 引擎源码（fork 自上游；上游历史在 `PolyStrata/.git-upstream`） |
| `交付/` | 随包专家表 `p_shipped.txt` + `交付物清单-20261018.md` |
| `README.md` | 本文件（参数 / 适配 / 判否清单 / 署名） |

> 阶段过程文档（项目书、台账、回执、情报、验收、协作与调度规则等）**不入库**，按项目惯例另存仓外归档。

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
| `--lookup` | 模板型负载 3；**异构负载 0** | 异构负载上 n-gram 猜不中，投机白花钱（实测关掉 +9.6%） |

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
`异构模式`（alt +8.6% 但 qa −13.8%，不通用）

## 6. ⚠️ 接入新模型到 strata 的通用检查表（**每次适配都会踩**）

> 血泪总结：同一台机器、同一个引擎，**batch 模式完全正常**，一上 `--serve`（给 agent/DSH 用）就
> "<unk> 循环 / 复读 / 乱码 / illegal memory access"。逐项排查后发现是下面这一类坑的叠加。
> **新模型接进来，把这张表从头跑一遍**，别等 agent 跑起来才查。

### 6.1 头号坑：**会话检查点/前缀复用 vs 递归状态**（本仓 2026-10-18 实测）
- **症状**：serve 下第一条长提示正常，**之后的请求全崩**（`<unk>` 重复 256 次、乱码、偶发 `illegal memory access`）；
  同一个提示、同样的采样参数，**batch 模式却完全正常**。
- **根因**：`--prompt-cache-every N` 会每 N 个 token 打"会话检查点"供跨请求复用。
  **mamba/hybrid 模型（40 mamba + 8 attn + 40 MoE）的递归状态无法像纯注意力 KV 那样存/取**，
  越过检查点的提示或后续请求拿到的是错的递归状态 ⇒ 输出崩。
- **判据**：**跨过边界才崩**。本仓实测：7,306 token 正常（< 8,192）、11,066 token 崩；把
  `--prompt-cache-every` **去掉**后，11,324 / 12,679 / **20,191** token 全部正确。
- **修法**：serve 配置**不要** `--prompt-cache-every`（长提示靠单次 `read()` 读完即可）。
  batch 模式若喂 >N 的长提示同样会中招——交付基准 6,792 < 8,192 才一直没暴露。
- **验证口径**：必须测"**连续多个长请求**"（第 2、3 条），只测第一条会假绿。

### 6.1b 二号坑：**借槽（`STRATA_PROMPT_BORROW`）在"一次进程跑多个请求"下污染缓存**
- **症状**：`--serve` 下**第一个请求正常、之后全崩**，输出 `<unk>`(id 0) 连续 256 次（logits 被坏缓存算成 NaN ⇒ argmax 恒 0）；
  长提示（28K）+ 长输出时最明显。
- **根因**：借槽把专家缓存**尾部 1,025 个槽**在预填期间借给 staging、预填后再回填。
  它只适合"**单次跑完就退出**"的 batch 模式（本仓交付基准 decode +11.3% 就在这个模式下测的）；
  **serve 进程连续处理多个请求时，借出/回填的槽在请求之间不一致** ⇒ 缓存里是错的专家权重。
- **判据**：同一进程连发 3 个 28K 请求 —— 带 borrow 时输出全 `<unk>`；**去掉 borrow 后 3 个请求全正常**
  （本机实测：prefill ≈1,200–1,400 tok/s、decode ≈17–18 tok/s、输出语义正确）。
- **修法**：**serve 配置不要 `STRATA_PROMPT_BORROW`**；batch 单跑交付可保留。

### 6.2 硬编码形状/静默截断（…"读取尾部不存在"）
- 同类坑：引擎里可能藏着**写死在参考模型上的常量**（另一条线 `cyber` 的记录：`idx_top_k = 2048` ⇒
  `selection_width ≤ 2,051` ⇒ 超过 2,051 token 的尾部"对模型不存在"；实测 2,045 ✅ / 2,085 ❌）。
- **查法**：新模型的 GGUF 元数据 vs 引擎里的常量（`idx_top_k`、`K`、`n_ff`、`n_embd` 的假设），
  并做**长度阶梯测试**（512 / 1K / 2K / 4K / 8K / 16K / 32K），看在哪一档输出语义开始错。

### 6.3 门控/激活函数的实现差异（数值崩塌）
- 同类坑（`cyber` 记录）：GDN（SSM 门）参考实现是 **SiLU**、引擎写成 **sigmoid** ⇒
  首位 token 错、32 token 文本全是 `整是inatom.manNEnenenene…`；改 5 处 kernel 后与参考逐字符相同。
- **查法**：每个新架构的**门控、归一化、位置编码、rope scaling** 都要与参考实现对一遍；
  症状常是"能跑但输出乱码"，不是崩溃。

### 6.4 停止/EOS 与 chat template
- 引擎若不读模型的 `eos_token_id`（只用硬编码集合）⇒ 该停不停、一路复读到 `max_new`。
- **chat template 必须从 GGUF 里导出**（`tools/strata_tokenizer.py --gguf … --out …`），
  别用引擎/服务端的默认模板（那通常是别的模型族的 `<|im_start|>` 风格）。
- **判据**：短问答的 ids/文本与参考一致；长提示不出现 `<unk>`(id 0)。

### 6.5 会"丢上下文"的省显存开关
- `--kv-resident N`（只把 N 格 KV 交给注意力、其余 mask）⇒ 不是"腾显存"，是**直接丢上下文**（`cyber` U8）。
- 同类：任何"按需驻留/分页"的开关，先确认**缺页时是阻塞读还是静默当作没有**。

### 6.6 接入验收顺序（照这个顺序测，省一轮）
```
① batch 模式：短 → 长（到目标长度）→ 连续多轮           ← 先确认引擎本身
② serve 单请求：短 → 长（必须超过所有"检查点/分块"边界）
③ serve 连续请求：长 → 长 → 长（第 2 条起才暴露复用类 bug）
④ 采样：greedy 与 sampled 各一遍（有的退化只在贪心下出现）
⑤ 与参考实现比对 ids/文本（前 N 个 token + 完整文本）
```

---

## 7. 来源与署名

| 层 | 来源 |
|---|---|
| **模型** | NVIDIA **Nemotron-3.5-Super-VL-120B-A12B**（Q4_K_M GGUF，70.1 GB） |
| **引擎** | **PolyStrata**（上游 `VecSzn/PolyStrata` 的 fork；本仓只做移植、接线与调参） |
| **引擎协议** | 上游 **`strata --serve`** 的行协议（`src/program/generate.cpp`：`GEN`/`GENI` → `T` 流）；HTTP/OpenAI 兼容层由 `PolyStrata/serve/server.py` 提供 |
| **tokenizer** | 由本仓 `tools/strata_tokenizer.py --gguf <gguf> --out <dir>` 从 GGUF 导出（含模型自带 chat template） |
| **加速机制** | 借槽 / 双缓冲 ring / 投机窗口 / 专家表学习等均**源自上游与老师线（strata）**；本仓做适配、实测与逐条判否 |

## 8. 使用注意

1. 本仓只收**项目必须内容**（源码 / 交付物 / 本说明）；日志、模型、exe、构建产物、实验数据一律不入库（`.gitignore` 有逐类清单）。
2. 结论的证据等级与"怎么变红"写在提交信息与判否清单里；**未验条目标 `未验`**。引用行号前先核对 `sha256`（源码在动）。
3. **任何涉及专家表的 A/B 必须冻结表**（`STRATA_PROFILE_AUTO_LEARN=0` + 同一只读表 + 自证 `saved=0`），
   否则表会在批次里漂移，产生**假阳性**（本仓踩过一次：看似 +21.9%，受控后是 −1.2%）。
