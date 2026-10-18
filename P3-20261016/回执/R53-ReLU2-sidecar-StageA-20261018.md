# R53 · ReLU² down neuron-major sidecar（Stage A）— 2026-10-18

- **席位**：builder3（fresh）· **Stage A：达成，A1 PASS** · **Stage B：未动（等 lead 批复）** · 零 GPU、零编译、未改 `PolyStrata` 一个字节、未 spawn 子代理。
- **任务书**：`计划-decode总攻-v2-20261018.md` Phase 3 / lead 派单（ReLU² down 重排）· 情报依据 `情报-X-ReLU2零跳过专项-20261018.md` · 实测依据 `回执\R51-ReLU2探针与真计时-20261018.md`（s=0.9206）。
- **一句话**：sidecar 已全量落盘（**31,708,938,240 B = 31.709 GB = 29.531 GiB**，20,480 专家 × 1,548,288 B，与 down 原字节数**逐字节同尺寸**）；**A1 可逆：8 随机 + 2 边界专家 sidecar→原布局逐字节 0 差异；负控（翻 1 位）已变红**；A4 的 mmap/WS/RAM 已量。Stage B（引擎只读接入 + 自证行 + ids 门）未开始。

---

## 0. 架构联动口径（A0 + lead 的 5 条）— 2026-10-18 追加，**Stage B loader 按这节**

> 触发：lead「这是架构调整，不是单旋钮问题」+ 需求指定的 RAM 预算（70.1+31.7 > 96 GB）。本节全部有盘上证据；A0 的数据文件 = `E:\sp3-sidecar\a0-down-byteranges.json`（40 条 down 张量区间，逐条 shard/绝对偏移/字节数）。

### 0.1 口径结论：**半替换（CPU decode 的 down 相替换；GPU 路径保留 GGUF 原布局）**

sidecar 只服务 **CPU 漏算（miss）路径的 down 相**；GPU 驻留/换入（fill/swap）、prefill staging、pcie direct **一律继续读 GGUF 原 down**。代码证据（本树 `PolyStrata\src\poly\engine.cpp`）：

| GPU 侧读 down 的位置 | 布局要求 | 行号 |
|---|---|---|
| `fill_expert_cache()`：VRAM 槽 = gate\|up\|down，down 用 `cudaMemcpy(L.exp_down + e*L.d_bytes, d_bytes)` **原样**拷入 | 原布局 | :1146-1159 |
| `stage()`：每专家 up/down 两次 `cudaMemcpyAsync`，down 从 `L.exp_down + e*d_bytes` **原样** DMA；pin 路径直接从未收集映射 DMA | 原布局 | :2664-2669 |
| `moe_card()`（小批）：down 指针 = `table[...] + 2*L.gu_bytes`（槽内 down 段），mmvq 按 `d_bytes` 行距读 | 原布局 | :2626-2628 |
| `moe_rows()`（prompt batch）：`mmq_->mul(sbuf + down_at, d_type, out_w, ff, L.d_bytes, ...)` 从 staging 按 d_bytes 行距读 | 原布局 | :3171 |

⇒ 若 GPU 也吃 neuron-major，要改 **① 小批 `native_mmvq_ids` 的 down 内核 ② prompt-batch 的 MMQ down 内核 ③ 槽布局 ④ staging 布局**（Stage C+，不在本包）。**因此本包/Stage C 的"替换"只对 CPU decode 成立；GGUF 原 down 区不能被删，也不能阻止 GPU 路径读它。**

### 0.2 A0 数据：GGUF 字节地图 —— "哪些字节可以不被读"

实测（两个分片 header 逐张量求和，零 GPU）：**tensor payload = 70,095,370,240 B（文件 70,103,292,320 B）**。

| 类别 | 张量数 | 字节 | GB | 占 payload |
|---|---|---|---|---|
| down_exps 主 40 层 | 40 | 31,708,938,240 | 31.71 | **45.24%** |
| up_exps 主 40 层 | 40 | 31,708,938,240 | 31.71 | 45.24% |
| 层内稠密（attn/mamba/norm） | 480 | 2,638,065,664 | 2.64 | 3.76% |
| shared expert | 82 | 1,134,895,104 | 1.13 | 1.62% |
| MTP 层 88（up+down+其他） | 13 | 1,624,729,600 | 1.62 | 2.32% |
| globals（token_embd/output/norm） | 3 | 742,408,192 | 0.74 | 1.06% |
| router | 41 | 343,932,928 | 0.34 | 0.49% |
| latent 投影 | 82 | 193,462,272 | 0.19 | 0.28% |

- **可被 sidecar 替代的读取 = down 主 40 层 31,708,938,240 B = 文件字节的 45.23%**（清单在 `a0-down-byteranges.json`）。
- **"原 down 不需要常驻"= 成立（对 CPU decode）**：decode 稳态 CPU 只读 up（GGUF）+ 活神经元 down（sidecar）；原 down 页可被 page cache 挤出，只在 (a) 加载 fill 5,721 专家 (b) prefill staging ≈42 GB/次 (c) swap 时按需读。
- **"原 down 从文件里消失"= 不成立**：GPU 路径还需要它（见 0.1 表）。
- 校验：up+down 主 40 层之和 = 59.06 GiB，**正好等于交付配置 `STRATA_PIN_ARENA_GIB=59.1` 钉住的 59.06 GiB** ⇒ lead 说的"WS 67.1 GB" = 59.06 pinned + ~8 GB 稠密/scratch，口径对上。

### 0.3 RAM 预算（回答 lead 的 101.7 GB）

| 项 | 读数 | 来源 |
|---|---|---|
| GGUF 文件 | 70.10 GB（65.29 GiB） | 实测 |
| sidecar | 31.71 GB（29.53 GiB） | 实测 |
| 现状 pin arena | 59.06 GiB（up+down 全 40 层） | R35-DP 日志 `pinned 40 layers (59.06 GiB)` |
| 现状进程 WS | 67.1 GB（本次 R54 实测 wsMax=67,395 MB） | R54 |
| **若 sidecar 整文件 touch/pin** | 59.06 + 29.53 + ~8 ≈ **96.6 GiB > 96 GiB ⇒ 必抖（禁止）** | 算账 + A4 实测 touch=+29.53 GiB |
| **本包口径** | sidecar **按需 mmap、不 pin、不整文件 touch**；常驻热集 = distinct 漏算专家 × 261 KB | — |
| 热集上界（全 distinct，不可能） | 496.7 miss/token × 128 token × 0.261 MB = 16.6 GB | R49 miss/token |
| 热集现实估计 | **2–8 GB**（专家重复使用；Stage B 打点实测） | 待测 |
| ⇒ RAM 可行区间 | **59.06 + ~8 + 2–8 + OS ≈ 70–78 GiB < 96 GiB ✓** | 算账 |

- **Stage B 必测三条（判红口径）**：① 进程 WS 增量；② `\Memory\Available MBytes` 最低值（要求 ≥8 GiB）；③ CPU down 相有效带宽是否仍 ≈31 GB/s（掉到 <20 判"page fault 抖动"，红）。
- **可选的省 RAM 变体（本包不做，另立实验）**：只 pin up 半（59.06 → 29.53 GiB），代价 = prefill staging 从 down 段直接 DMA 退化为 host gather；R54 已实测 staging 是敏感项（P1 −11%），故不建议在本包动。

### 0.4 lead 的 5 条联动，逐条回答

1. **GPU 路径也要 down**：现读**原布局**（0.1 表四行证据）。GPU 也吃 sidecar 需改 4 处（小批 mmvq down、MMQ down、槽布局、staging 布局），**代价大、不在本包**；本包 GPU 路径一行不动。
2. **prefill staging 混合来源**：**本包不存在混合来源**——staging 完全不用 sidecar，仍整块从 GGUF DMA。若 Stage C+ 要 down 来自 sidecar：要么 host 做 T→原 反变换（20,480 专家/次 × 1.55 MB ≈ **+6 s/次 prefill**，超过 prefill 墙 3.5 s，不可接受），要么改 GPU 内核。⇒ **staging 保持 GGUF 源**。
3. **slot 布局/字节数**：本包一行不动。将来若要改，清单 = ① `Layer::slot_up()/slot_down()`（engine.hpp:139-140）② `expert_slot_bytes()`（:1069）③ `fill_expert_cache()`（:1146-1159）④ swap/follow 的 pinned 拷贝（:1355-1362）⑤ `stage()`（:2636-2696）⑥ `moe_card()` down（:2626）⑦ `moe_rows()` 的 mmq down（:3171）⑧ `exp_down_dev_` pcie direct 映射（:1808-1821, :2332-2340）。
4. **几何**：sidecar manifest（`downT.json`）按 **2688 神经元 / 84 组每行 / 2048 B 每组** 写死并已 A1 逐字节验；GPU 侧原布局不涉及。
5. **位精确**：走 **块序（block-order），不是 rank-1**。读码依据（本机 llama.cpp pin，build-p3 `GGML_NATIVE=ON` ⇒ 走 AVX2/FMA 路径）：`ggml_vec_dot_iq4_nl_q8_0`（`E:\llama-cpp-pin\ggml\src\ggml-cpu\arch\x86\quants.c:3920-3964`）是 **even/odd 块两路 `accum1/accum2` + 8 lane FMA + `hsum_float_8`（同文件 :43 的固定顺序）**；块内 int32 lane 和与顺序无关（0 值恰好贡献 0）。⇒ 位精确做法：84 块按原序成对处理、块内只 gather 活神经元的 int32 lane 和、float 结构照抄（`_mm256_fmadd_ps` 的两路 accum + 同一 hsum）。**rank-1 逐神经元 float 累加会改 float 加法顺序 ⇒ 不位精确**，若 Stage C 为性能走它必须另立 ids 门。**任务书里"神经元升序 ⇒ 位精确"要补这句：升序只在 int32 域充分；float 域必须保持块序。**

---

## 1. 几何：排表三处数字是 helgard 的 NVFP4 几何，已按我们实测订正（**关键，后续 Stage C 内核按这节**）

| 项 | 排表/任务书 | **实测（GGUF 逐张量核对，A 级）** | 依据 |
|---|---|---|---|
| down 量化 | IQ4_NL | **IQ4_NL，type=20，块 = 32 值 / 18 B**（2 B fp16 scale + 16 B nibbles） | 两个分片 head，`ffn_down_exps.weight` 全 41 个张量 type=20 |
| down 形状 | — | **`[2688, 1024, 512]`**（n_in=2688 神经元连续，n_out=1024 行，512 专家） | GGUF header，`PolyStrata\src\poly\engine.cpp:774-777` 的同一形状断言 |
| 每神经元 | 512 B | **512 B**（1024 行 × 4 bit；字节 b 低 nibble = 行 2b，高 = 行 2b+1） | 同 helgard T 布局，nibble 对行做配对 |
| scale 区 | **168 组 × 1024 B/组**（16 神经元/组） | **84 组 × 2048 B/组**（**32 神经元/组** = 一个 IQ4_NL 块；1024 行 × 2 B fp16） | IQ4_NL 块几何；总量仍 84×2048 = 172,032 B = 2688×1024×0.0625 B |
| "active=N/**1024**" | 自证行样例 | 应为 **N/2688**（1024 是输出行数，不是神经元数） | `expert_feed_forward_length=2688`、`moe_latent_size=1024` |
| down_exps 张量 | 20,480 专家 / 31.7 GB | **41 个张量**（40 主层 + **1 个 nextn/MTP 层 88**）；`block_count=89`、`nextn_predict_layers=1`、主模型 `n_layer_=88` ⇒ **sidecar 只收 40 主层 × 512 = 20,480 专家**，与 31.7 GB 一致；层 88 只在 `--draft` 时加载，不在本包范围 | GGUF 元数据 + `engine.cpp:700-702` |
| 分片 | "第 2 分片" | **两个分片都有**：分片1 23 个层（1..50）、分片2 18 个层（52..88）；侧车按层升序拼接 | `split.no/count=1/2`、两分片 tensor 列表 |

- sidecar 每专家块 = `[values 1,376,256 B | scales 172,032 B] = 1,548,288 B = 378 × 4096`（**页对齐**，为 Stage B 的 mmap/register 留好）。
- sidecar 顺序 = 主模型 `ffn_down_exps` 张量按层升序、每张量 512 专家按张量内序；`offset(层序, 专家) = (层序×512+专家) × 1,548,288`。层→分片→绝对偏移全部写进 manifest。

## 2. 工具与产物（绝对路径）

| 件 | 路径 | 说明 |
|---|---|---|
| repack 工具（源码） | `C:\sp3\repack_down.py` | Python 3.12 + numpy 2.5.4；**零编译**（不碰 `build-p3`）；GGUF 只读（头解析用 `PolyStrata\tools\gguf_reader.py`，从不写 GGUF） |
| **sidecar** | `E:\sp3-sidecar\downT.bin` | 31,708,938,240 B；20,480 专家；总尺寸 = down 原字节数 |
| manifest | `E:\sp3-sidecar\downT.json` | 几何常量、nibble/scale 约定、40 层（layer/shard/tensor 绝对偏移/sidecar 偏移）、分片大小、构建时间 |
| 干跑残留 | `E:\sp3-sidecar\_dry.bin`、`_dry.json` | 1 层 512 专家（0.79 GB）的预跑，供复核；可删 |

- 写盘吞吐：40 张量 **195.7 s / 31.71 GB（162 MB/s）**，全程顺序读写、无中断、退出码 0。
- 工具自带两条校验：`--write` 后文件尺寸必须 == 40×512×1,548,288；`--verify N --negctl` 做反变换逐字节比对 + 负控。

## 3. A1 可逆（**本包硬门：过**）

判据：随机抽 **8** 个专家（`random.Random(7)`，跨层）+ 2 个边界（首层专家 0、末层专家 511），从 sidecar 反变换（`from_T`）回原布局，与 GGUF 原字节**逐字节**比对。

| 专家 | 结果 |
|---|---|
| layer 1 expert 0（边界） | **0 diff**（1,548,288 B 全同） |
| layer 87 expert 511（边界） | **0 diff** |
| layer 45 expert 154 | **0 diff** |
| layer 56 expert 49 | **0 diff** |
| layer 10 expert 96 | **0 diff** |
| layer 52 expert 59 | **0 diff** |
| layer 72 expert 219 | **0 diff** |
| layer 5 expert 88 | **0 diff** |
| layer 61 expert 428 | **0 diff** |
| layer 10 expert 246 | **0 diff** |

- **汇总：10/10 专家、0 差异、0 字节；exit=0。**
- **负控（判据能变红，已实测）**：把重建字节第 1000 字节翻 1 位 ⇒ 比对报告 `detected=True, diff_bytes=1` ⇒ 校验器不是恒绿。
- 边界：只抽 10/20,480 专家（任务书要求 8，本包超额 2 个边界）；**未做全量 20,480 反变换**（31.7 GB 往返，非本包要求；Stage C 换内核前应以 1 跑 ids 门兜底）。

## 4. A4 内存/盘（零 GPU 实测）

| 量 | 读数 |
|---|---|
| sidecar 实际大小 | **31,708,938,240 B = 31.709 GB = 29.531 GiB**（== down 原字节数） |
| mmap + 全页 touch 后进程 WS 增量 | **+29.53 GiB**（= sidecar 全量常驻；7.7M 页） |
| 系统 avail RAM（touch 前 → 后） | **82.97 → 53.92 GiB**；**余量 53.92 GiB ≥ 8 GiB ✓**；进程退出后回到 82.8 GiB（页缓存可回收） |
| 盘余量 | E: 500.6 → **468.8 GB 左右**（写后实测见 §6 的 sidecar 大小；E: 总量 2 TB） |

- **未验**：`cudaHostRegisterReadOnly` 锁 31.7 GB 的可行性与代价（Stage A 只做 mmap；Stage B 才接 pin 路径，届时按 `engine.cpp:1013-1064` 的预算检查先断言 RAM）。

## 5. Stage B 未动（如实报，等批）

- 没有改引擎、没有编译、没跑 GPU；所有默认值未动。
- Stage B 计划（照 task 书）：sidecar mmap 加载（env `STRATA_RELU2_DOWN=0|1`，默认 0）→ relu² 后 `h[i]!=0` 升序建活神经元索引（**n_ff=2688**）→ CPU down 路径打自证行 `relu2-down: sidecar on, active=N/2688, bytes_read=X (vs 1548 KB)` → **不换内核**（内核留 Stage C）。A2/A3 需要 GPU 跑，随 Stage B 一并做。

## 6. 落盘（绝对路径）

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R53-ReLU2-sidecar-StageA-20261018.md`（Stage B 回执另立 `R56-ReLU2-StageB-20261018.md`）
- sidecar：`E:\sp3-sidecar\downT.bin`（31,708,938,240 B）
- manifest：`E:\sp3-sidecar\downT.json`
- 工具：`C:\sp3\repack_down.py`
- 干跑残留：`E:\sp3-sidecar\_dry.bin`（792,723,456 B）、`E:\sp3-sidecar\_dry.json`

## 7. 未验 / 边界

1. **A2/A3/bytes_read/峰值**：属 Stage B（GPU），本包未做。
2. 只抽 10 个专家验证可逆（任务要求 8）；全量反变换未做。
3. 收益仍按 R51 实测 s=0.9206 推：down 字节可省比例 = 活神经元 7.94%×512B + 活块（P=1−0.9206^32=93.0%）×2048 B ⇒ **每专家 ≈ 261 KB / 1548 KB = 16.9%**（与情报-X 的 16.9% 一致；i.i.d. 假设）；**实际 gather 效率与 ids 门 Stage C 验**。
4. 层 88（MTP）不在 sidecar；若 `--draft` 开启需扩 sidecar（+0.79 GB，工具改 `--include-nextn` 即可，本包没做）。
5. sidecar 只是"把零变连续"的字节重排，**不改 down 数值**；Stage C 的求和顺序仍是风险点（任务书已预注册硬门 = ids 逐位同）。
