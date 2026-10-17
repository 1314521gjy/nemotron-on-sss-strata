# 情报-D：Strata 架构摸排 + ring 三问 + prefill 剩余杠杆 — 2026-10-18

- **席位**：`scout-b`（情报B）｜**任务**：`task-13`｜**只读**：零 GPU、零编译、零写 `src`；未跑任何 exe（含 `C:\sp3\strata.exe`）；`E:\strata`/`E:\rlab` 只读；只写本件与回执。
- **路径前缀**：`PS` = `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata`；`UP` = `…\upstream-0.1.44.1`（与 `PS` 平级，机制文件 sha256 相同，行号通用）；`P3` = `…\P3-20261016`。
- **证据等级**：`A(源码)`｜`A(日志)`=读本机历史实跑日志（非本席跑）｜`B`=队友件/交接件｜`C`=外部文档｜`D`=推算/规格｜`未验`=没查到或没跑过。
- **★ 先纠任务单的一处版本前提（A(日志)）**：任务单写"老师那次 = 原厂 0.1.43.1"，但 **`dec-pool2.txt:42` 自报 `session is up (engine 0.1.40.2)`**；`C:\sp3\strata.exe` + `C:\sp3\help.txt` 才是 **0.1.43.1**（`C:\sp3\BUILD.json` = `"version": "0.1.43.1"`，A(文件)）。⇒ **老师原始那次 = 0.1.40.2；0.1.43.1 是我们复跑用的**。本件的三方证据按这个分：help/默认值用 0.1.43.1，日志行为用 0.1.40.2 自己的日志。

---

## §0 三句话（先回答目标问题）

1. **ring 是上游的机制，不是"没有"**：上游 prefill 的**已排序多槽流水**（`PS\src\prefill\prefill.cpp:132-134` 原文）是 0.1.30 起、0.1.39b 改成字节预算的**默认路径**（`:270-314`；无关闭开关，只有 A/B env）。它的用途一句话：**从 chunk≥1024 token 起，每层所有非驻留专家按固定顺序流过 N 个槽，拷贝引擎在 attention/计算段继续工作**。
2. **老师那次（0.1.40.2）ring 开了（B+ 级）**：四条同代日志里，**chunk≥1024 的跑 `experts streamed` = 24576−resident（逐位相等）**，chunk=220 的跑只有 3562（routed-only）——这正是 `stream_all` 分支（**唯一使用 `m.ring` 的分支**）的签名。**具体 N 槽未验**（generate 模式不打印槽数，只有 serve 打，`PS\src\program\generate.cpp:7570`）。
3. **"wait for rings" 不是专家环、也不是 KV 环**：它是 **decode session loop 的逐层门铃（GPU→host flag）**，源码在 `PS\src\core\session.cpp:686-723`（"poll for the ring" / "ended without ringing"），打印在 decode/verify 表（`generate.cpp:13935-13940`）。判据见 §2.2。

---

## §1 A1：上游 ring 的确切形态（A(源码)）

### 1.1 一句话语义（原文）

`PS\src\prefill\prefill.cpp:132-134`：
> *"from this chunk size on, every non-resident expert of every layer streams in a fixed order through a ring_slots()-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the attention halves instead of waiting for each layer's routing."*

⇒ 三要素：**固定顺序**（不依赖本层 routing）、**N 槽**、**拷贝引擎与计算重叠**。

### 1.2 逐项形态表

| 项 | 事实 | 证据 |
|---|---|---|
| **启用条件（代码）** | `stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr`（`prefill.cpp:2341`）；`ring_slots(T)` 对 `T < stream_all_min()` 返回 `STAGE` | `prefill.cpp:2341`、`:292-314` |
| **默认开/关** | **默认开**：无 CLI 旗标；`STRATA_PREFILL_STREAM_AHEAD` 默认非 0（`stream_ahead_enabled()`）、`STRATA_RING_BYTES` 默认非 0（`ring_bytes_on()`）；`stream_all_min()` = **1024** token（0.1.30 起，`:158-162`） | `prefill.cpp:125-131`、`:282-285`、`:164` |
| **可关闭的 env（A/B）** | `STRATA_PREFILL_STREAM_AHEAD=0`（退回 routed-only 上传）、`STRATA_RING_BYTES=0`（回 0.1.39 的 ring/loan/chunk 列表）、`STRATA_PREFILL_RING=N`（直接覆盖槽数，`=8` 即 STAGE ⇒ routed-only） | `prefill.cpp:124-127`、`:270-285`、`:292-313` |
| **N 几槽** | 由 `ring_slots(T)` 决定：①`STRATA_PREFILL_RING` 覆盖；②否则 pinned-share 规则：**pinned≥0.9 ⇒ 1024（fused）/384**，否则 **96**（`:308`）；③#583 字节预算：`ring_bytes()/MAXBLOB()`（`:286-291`），受 `ring_cap()` 限制：**fused 且非 native = 1024，其余 = 512**（`:257-259`）；④实测锚点：Q2_0 pack **384 槽（96→384 提升 1153→1294 tok/s，`:214-217`）**；**IQ3_S（2.54 MiB blob）pack = 199 槽**（`include\strata\prefill\prefill.hpp:107-110`）；fused Q2_0 **512→1024 槽**（4K prompt 1512→1605 tok/s，`prefill.cpp:232-239`） | `prefill.cpp:230-314`、`prefill.hpp:80-121` |
| **每槽多大** | **一个整专家 blob**：`MAXBLOB()` = `expert_layout().max_blob`（`prefill.cpp:121`）；IQ3_S = 2.54 MiB ⇒ 199 槽 ≈ 505 MiB；**本靶 Q4_K_M 若套用 = 3,096,576 B/槽**（我们每专家帧，A(日志) `pair_D_pb6912.log` 口径），字节预算 384 槽×Q2_0_BLOB(1.3824 MB) ⇒ **≈171 槽 ≈ 505 MiB**（D 推算，未验） | `prefill.cpp:121/263-291`、`prefill.hpp:107-110` |
| **槽从哪分配** | 从 **prompt 缓冲的 borrowed 区**（`Prefill::init(..., void* borrow, uint64_t borrow_bytes)`，`prefill.hpp:68-73`）：`bytes_needed` 里 `ring_slots × MAXBLOB`（`prefill.cpp:1923-1926`、`:1936`）⇒ **ring 的显存是从"借来的专家缓存槽"里切的**；无 borrowed 区时自己 `o.take` 一整块（owned 一次分配，`prefill.cpp:1319-1335`） | `prefill.cpp:1318-1335`、`:1922-1936` |
| **复用/事件纪律** | **每槽一对事件**：`m.copied[sl]`（该槽 DMA 完成，拷贝流上 record）与 `m.used[sl]`（该槽最后一个读者完成，计算流上 record）；**写者复用槽前等 `used`**（`cudaStreamWaitEvent(m.copy, m.used[used_of[sl]])`，`:2439/:2487/:3806/:3812`），**读者用前等 `copied`**（`:3103/:3972/:4000`）；用完 record `used`（`:3862/:3899/:3986`）。事件创建 `:1127-1128`。⇒ **最多 N 个 DMA 在飞，槽号 `k % m.ring` 轮转**（`:3985/3996`） | `prefill.cpp:1127-1128/2436-2496/3800-3820/3980-4004` |
| **另有两个"ring"（同名不同物，别混）** | ①**host 缓冲 ring** `kRing=16`（`STRATA_STAGER_RING`，`:422-458`，files 模式 4×threads `:1153`）；②**反量化专家 ring** `constexpr int DQ = 2`（"dequantized-expert ring (FP16 gate/up + down)"，`:315`，分配 `:1307`，用于 FP16 prompt 路径 `:3932-3951`） | `prefill.cpp:315/422-458/847-848/1307/3932-3951` |
| **分层 split 的 ring** | `PS\include\strata\program\split_ring.hpp:20-34`：`STRATA_SPLIT_RING` 覆盖，否则"≥75% 常驻 ⇒ 96 槽"；`split_lend_slots` = 按字节算"尾部 k 槽"（供 RAM 副本与 borrow 用同一套算术）；测试 `PS\src\program\split_ring_test.cpp:57-79` | `split_ring.hpp:6-34`、`split_ring_test.cpp:29-83` |

### 1.3 为什么它有用（上游实测，C/文档）

- `prefill.cpp:214-217`：Q2_0、8192-token chunk，**96 槽 1153 → 384 槽 1294 tok/s**（"the next layer's experts arrive during its attention half"）。
- `prefill.cpp:232-239`：fused 路径下 **512 槽 +19-22%（4K）/ +11-12%（32K）** 比 384 槽。

---

## §2 A2：老师那次（0.1.40.2）ring 到底开没开？

### 2.1 三方证据交叉

| 证据源 | 事实 | 判读 |
|---|---|---|
| **(i) help/默认值** | 0.1.40.2 help（`E:\strata\_staging\help-0.1.40.2.txt`）与 0.1.43.1 help（`C:\sp3\help.txt`）**都没有 ring / stream-ahead 的 CLI 开关**（grep `RING`/`STREAM_AHEAD` ⇒ 0 个旗标；只有 `--prefill auto` `:139/141`）。⇒ ring 是**默认路径**，要关只能设 env；日志不回显 env | **默认开**（A(文件)+A(源码)） |
| **(ii) 日志字段（决定性）** | `experts streamed` 与 `24576 − resident` 的关系：**chunk≥1024 的三条全部逐位相等**，chunk=220 的一条远小于唯一集 | **`stream_all`（= ring 的唯一使用者）在跑** |
| **(iii) 代码启用条件（0.1.44.1 源码）** | `stream_all = m.ring > STAGE(8) && T ≥ 1024 && src != nullptr`（`prefill.cpp:2341`）；`ring_slots` 有 override/pinned/byte-budget 三条路，**没有一条默认返回 STAGE**（pinned≥0.9 ⇒ 384/1024，且 ≥16 下限 `:312`）⇒ chunk 6912 必然进 ring 分支 | **必然开**（A(源码)，但版本=0.1.44.1） |

**日志表（A(日志)，`E:\strata\_staging\` 四条同代 0.1.40.2 跑）**：

| 日志 | chunk/tokens | `experts streamed` | `resident` | 唯一非驻留 = 24576−resident | 判 |
|---|---|---|---|---|---|
| `dec-pool2.txt:50` | 6912 / 6792 | **15535** | 9041 | **15535** | **相等 ⇒ stream_all/ring 开** |
| `trace-work.log:51` | 6912 / 6792 | **15623** | 8953 | **15623** | 相等 ⇒ 开 |
| `trace-long32k.log:50` | 8192 / 32767（4 chunks） | **62844** | 35460（累计） | 每 chunk **62844/4 = 15711**；35460/4=8865 ⇒ 24576−8865=**15711** | 每 chunk 相等 ⇒ 开（4 chunk 各流一遍全非驻留） |
| `trace-tiny.log:50` | **220** / 220 | **3562** | 7855 | 24576−7855=**16721 ≫ 3562** | 不等 ⇒ **routed-only（chunk<1024）**，反证阈值规则 |

⇒ **A2 判决：老师那次的 ring 开了（B+；日志实证 + 默认无开关 + 代码条件）**；**具体 N = 未验**（见 §2.3 探针）。
补一条弱旁证：老师 borrow **2387 槽 / 4.55 GiB**（`dec-pool2.txt:46`）vs 我们 0.1.43.1 复跑 borrow **1903 槽 / 3.62 GiB**（`C:\sp3\q38_repro_6792.log:37`）——差 484 槽 ≈ 1 GiB，**可能**是 ring 槽/python 缓冲记账差；但两者版本不同（0.1.40.2 vs 0.1.43.1），**不足以定 N**（未验）。

### 2.2 ★ "wait for rings" 是什么（要求给判据）

**判决：decode session loop 的逐层门铃（GPU→host 的 flag），既不是专家环、也不是 KV 环。**

判据（每条都能变红）：
1. **出处**：该字符串只有一个来源 —— `PS\src\program\generate.cpp:13935-13940` 的 decode/verify 汇总行：`"verify window  wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round"`，分子是 `ver.ms_wait`（**verifier 等每个窗口/layer 的 ring**），分母是 **decode rounds**（`dec-pool2.txt:71` 同族 `:66` "171 tokens, 64 rounds"）⇒ **它属于 decode 表，不属于 prefill**。
2. **机制**：`PS\src\core\session.cpp:658-723` —— *"pre[0] IS LAUNCHED BEFORE THE LOOP SO EVERY ITERATION HAS THE SAME SHAPE: poll the ring that …"*（`:658`）、*"---- poll for the ring. THE DRIVER CALL IS NOW A FALLBACK, NOT THE MECHANISM."*（`:686`）、`if (mid_graph) ++gr.rings_mid_graph`（`:716`）、*"layer … ended without ringing"*（`:723`）、`:718-719` 记 `ms_to_ring`。⇒ 这是**每层 `pre[l]` 图跑完/到某点时向 host 打的一个 flag（门铃），host 轮询到它才发 `post[l]`**。
3. **同名不同物反证**：专家 ring 的全部状态是 `prefill.cpp` 里的 `m.ring` 槽 + `m.copied/m.used` 事件（§1.2），**没有任何 host 轮询/“wait”语义**；而 KV 在上游叫 **cells**（`--kv-resident N cells`，help:26），**不叫 ring**。⇒ 三个"ring"里，`wait for rings` 只可能是门铃。
4. **可观测性反证**：`generate.cpp:14152-14156` 另有 *"rings seen MID-GRAPH … / ring latency …"*（token graph 侧），也全部是 decode/图捕获路径。

⇒ `dec-pool2.txt:71` 的 `wait for rings 12.378 ms/round` = **decode 每 round 等每层门铃的累计**，**与 prefill 的专家 ring 无关**。先前"可能是 KV"的猜测（`P3\对拍-我方与老师-差异总表-20261017.md:72`）**作废**（只加不改：该件不动，本件为唯一新结论）。

### 2.3 若还要定"老师那次 N 槽"：一次探针的计划（零改码、一次跑）

- 用 **`C:\sp3\strata.exe`（0.1.43.1，A(文件) `BUILD.json`）** 或 0.1.40.2 原 exe，同 config 跑 6792-token prompt，**只改 env**：
  - 臂 A：不设（默认）→ 记录 `experts streamed`、`prompt path borrows N slots (X GiB)`、prefill tok/s；
  - 臂 B：`STRATA_PREFILL_RING=8`（= STAGE ⇒ **强制 routed-only**，`prefill.cpp:311`）→ `experts streamed` 应**从 15,535 掉到 routed-only**、borrow 应**减少 ring 字节**、prefill 应**变慢**（上游另件实测 ring 96→384 差 +12%）；
  - 臂 C（可选）：`STRATA_PREFILL_RING=96` vs `384` vs `1024`，看 borrow/耗时单调。
- **哪条日志字段能定**：`experts streamed`（唯一集 vs routed-only）+ `prompt path borrows N slots`（含 ring 字节）+ prefill tok/s 三列同报。**call 数/槽数不会打印，不要等它出现。**
- 说明：0.1.40.2 源码本机**没有**（`E:\strata\_staging` 只有 0.1.36 源码 zip + 0.1.38/0.1.39/0.1.43/0.1.43.1 二进制解包），所以"0.1.40.2 的代码条件"只能用 0.1.44.1 代理 + 0.1.40.2 日志行为实证（已做）。

---

## §3 A3：它 prefill "为什么不像用 ring"？

**结论：不是没开，也不是走了 borrow 替代；是 (a) generate 模式不打印 ring、(b) ring 槽从 borrow 里切所以只在 borrow 数值里、且 (c) borrow 和 ring 是"资金 vs 流水"的关系，不是二选一。** 证据强度：**A(源码) + B+(日志行为)**；N 未验。

1. **不打印**：`generate` 模式的 prefill 汇总只有 `streamed/resident/host/PLE`（`generate.cpp:13271-13276`，即 `dec-pool2.txt:50` 那一行）；**ring 槽数只出现在 serve 的日志**：`"the prompt path borrows %lld CUDA0 cache slots"` / `"lend sizing: … (ring %lld slots at chunk %lld …)"`（`generate.cpp:7554-7572`）。⇒ "日志里没有 ring" ≠ "没用 ring"（这正是先前判据过松的地方，`P3\对拍-…差异总表:56` 已自我更正）。
2. **ring 在 borrow 里**：`Prefill::init` 的缓冲区（含 ring）全部从 `borrow` 区切（`prefill.hpp:68-73`、`prefill.cpp:1211-1213`、`:1319-1335`），`bytes_needed` 把 `ring_slots×MAXBLOB` 算进借出量（`prefill.cpp:1923-1926/1936`）⇒ **borrow 4.55 GiB 里已含 ring 的字节**；ring 越大，borrow 越大、VRAM 常驻越少（老师 resident 9041 vs 我们复跑 10109，可能就是这笔账）。
3. **borrow 不是替代路径**：borrow = "prompt 缓冲借用专家缓存槽、prompt 后回填"（`generate.cpp:548-550`、`:13167-13210`）；ring = "这堆 prompt 缓冲里的 DMA 流水怎么排"。两者是**上下层关系**，同时存在。

---

## §4 B：strata 的完整运行流程（program 线）

### 4.1 入口有哪几条

| 入口 | 代码 | 说明 |
|---|---|---|
| `strata generate`（CLI 文本生成） | `PS\src\program\generate.cpp:1799 int main`；`if (!o.serve)` 分支 `:6063` | 老师 2997 跑的就是它（`dec-pool2.txt:1` 的 `strata generate`） |
| `strata serve`（OpenAI/Anthropic 服务） | `:7231 if (o.serve)`（serve 主循环 `/ :12600-13000` 区间） | 带 conversation-cache / 多请求 / 多卡窗口 |
| `--batch` / `--batch-groups`（serve 多槽） | help `C:\sp3\help.txt:100-106`；`PS\src\program\batch_groups.hpp`；主循环 `generate.cpp:10172+` | 多请求吞吐，不是单 prompt 计算 |
| 测具入口 | `--gpu-only-full` `:6337`、`--gpu-stages` `:6149`、`--graph-only`、`--stage-timing` `:5896-5924`、`--stats` `:14019` | 只测、或打印后退出 |
| 独立小工具 | `PS\CMakeLists.txt:306 strata-plan`（`src\plan\plan_main.cpp`）、`:245/248/324/327/330/334`（strata-gguf/dequant/device/load/concurrent/overlap） | 计划/加载/并发小件 |
| **我们的 `strata-poly`** | `PS\CMakeLists.txt:1818-1830`（**独立 exe**，不链 program/prefill/engine） | 另一条引擎线 |

### 4.2 五步流程骨架（每步的显存/内存动作 + file:line）

**① load（磁盘→宿主内存，本质上"不拷贝"）**
- 解析 argv：`generate.cpp:1874-2280`（`--pack/--native/--ple-gguf/--expert-profile/--expert-cache/--prefill/...`）。
- 打开专家源：`PS\src\core\expert_source.cpp:818 FileExpertSource::open`（pack 有 experts.bin）或 `:1122 open_gguf`（native pack）——**`MapViewOfFile`/`mmap` 只读映射**（`:947/:974/:1154/:1173`），专家 blob 留在**文件页缓存**（宿主内存，不占私有 RAM 也不算 VRAM）；`advise_pairs`（`:1517`）可提前给页缓存提示。
- 权重装载：`strata::core::WeightTable`（`src\core\weights.cpp`）；日志 `generate.cpp:3038`（"native pack: … largest blob …"）、`:3162`（"N MiB of weights loaded"）。
- PLE 表：`--ple-gguf` 第二个 GGUF（`:3038-3060` 区间；`src\ngram\ple_reader.*`），运行时按需读 SSD（`dec-pool2.txt:82` 的 ple io 行）。
- **显存动作：无**（除内核/图上下文的固定开销）；**内存动作：映射+页缓存**。

**② 常驻槽填充（VRAM arena + 可选 RAM 副本）**
- 专家缓存定尺寸：`--expert-cache auto` ⇒ 用 `cudaMemGetInfo()−reserve` 算槽数（`generate.cpp:5133-5198` 日志；`xcache.open/open_sized` `:5377-5378`；`src\core\expert_cache.cpp`）。
- 填 VRAM：`--expert-profile` 的排名表 ⇒ 逐槽 `admit` + `fill_slot_blocking`（`generate.cpp:5487-5515`；日志 "pre-filled N of M slots"）。
- **RAM 副本**：`pin_cache_complement`（`generate.cpp:7086-7099` → `src\core\expert_source.cpp:2634`）把"GPU 不驻留的补集 + 借出尾部"拷进**页锁定宿主 RAM**（`--resident-budget-gib` 控制多少，`generate.cpp:460-463`；unbuffered 读 `:4845-4853`）。
- **显存动作：arena（cudaMalloc 或 VMM 段，`src\core\vmm.cpp`）= 槽数×blob；内存动作：可选 pinned complement（= 老师 `dec-pool2.txt:30` 的 "loaded 46.84 GiB"）**。

**③ prefill（prompt 一批算完）**
- 借槽：从缓存尾部借 `k` 槽（`generate.cpp:13167-13210`；`Prefill::bytes_needed` 定价，含 ring）；`prefill.init`（`:13208` → `PS\src\prefill\prefill.cpp:1084`）在借区里布局全部 prompt 缓冲；`prefill.run`（`:13233`）按 `--prefill` chunk 逐个跑。
- 计算：`Prefill::run` 内（`prefill.cpp:2341-2513` 发拷贝；`:3760-4012` 逐层消费）——**拷贝流 `m.copy` 把专家 blob 搬进 ring 槽；计算流 `m.cs` 等每槽 `copied` 事件后跑 MMQ/FP16 GEMM；每槽用完 record `used`**（§1.2）。
- 收尾：回填借出的槽（`generate.cpp:13243-13260` "lent slots refilled"）、弹性 KV 开池（`kvg_start/kvg_ensure` `:13229/:13282`）。
- **显存动作：借来的槽在 prompt 期不放专家（= 驻留掉 k 槽）+ ring 在同一借区；内存动作：blob 从 pinned RAM 或 mmap 直 DMA（`host_res.data()` 传给 `Prefill::init`，`generate.cpp:13208-13210`）**。

**④ decode（逐 token / verify 窗口）**
- verify 框架：`strata::core::Verify`（`src\core\verify.cpp`）→ `session_loop`（`generate.cpp:13357` → `PS\src\core\session.cpp:590-800`）。
- 每层：launch `pre[l]` 图（带 CUDA graph，`session_capture` `:205-266`）→ **host poll 该层的"ring"门铃**（`:686-723`）→ launch `post[l]`；token graph 捕获/回放 `session_capture_token :851-914`、`session_run_token :925-980`。
- 专家：VRAM 槽命中在卡上算（R4），miss 交 CPU 池/PCIe（`--pcie-frac`），自适应层换入（`--expert-swaps`，`generate.cpp:8922-9339`）；投机：MTP（`src\core\mtp.cpp`）+ `src\spec\*`（draft_policy/suffix_drafter）+ `--lookup-chain`。
- **显存动作：每层 pre/post 图 + KV cells（`--kv-resident`，默认按 ctx 定）+ verify 窗口缓冲；内存动作：CPU 专家池算 miss、换入换出走 PCIe**。

**⑤ 会话存盘（checkpoint / park / session save）**
- conversation cache：`generate.cpp:8138 strata::core::ConversationCache conversations(...)`；每读 N token 取一次 `checkpoint_at`（`:8438-8479`，把 `SessionState`——gdn/ple/tails/dead/block_pos——+ ids 拷到宿主 blob，~118 MB/个）；`--prompt-cache-every` 默认 **16384**（`:647`），`--prompt-cache` 默认 6 个（`:632`）。
- 停车（park）：`:8219-8258`（把会话的 running checkpoints + K/V 行拷到宿主，`--conversation-cache-mib/slots` 控制容量）。
- session 存盘/恢复：`:10471-10627`（`session_save_checkpoints` / `conversation_file.cpp` / `conversation_snapshot.cpp`）。
- **显存动作：无（释放/回滚）；内存动作：checkpoint 副本（每 8.4k token 级 118 MB）+ 磁盘文件**。

### 4.3 "框架件"清单：program 线独有、poly 线没有（A(源码)）

**poly 的 include 面（`grep '#include "strata/' PS\src\poly` 全量 32 命中）只有**：`strata/poly/kernels.hpp`、`strata/kernels/native_mmvq.hpp`、`strata/artifact/gguf_reader.hpp`、`strata/spec/draft_policy.hpp`、`strata/spec/suffix_drafter.hpp`、`strata/core/conversation_memory.hpp`。⇒ 下列骨架 **poly 不 include、不链接**（`PS\CMakeLists.txt:1818-1830` 链接面：`strata_poly_mmq + strata_kernels + strata_spec + ggml-*`；`strata_engine/strata_prefill` **不在**）：

| # | 框架件 | 上游归属 | poly 有没有 | 作用（一句话） |
|---|---|---|---|---|
| 1 | **Prefill 引擎**（`gemm.cu/prefill.cpp/wmma_gemm.cu/kernels.cu`） | `CMakeLists.txt:612-615`；`src\prefill\*` | **无**（0 include/链接；poly 自己的 `moe_rows` 是行式引擎） | prompt 批量计算 + borrowed 缓冲 + ring |
| 2 | **ExpertCache**（槽表、per-layer 准入、VMM 弹性） | `src\core\expert_cache.cpp`；`CMakeLists.txt:491-497` | **无**（poly 自研 arena `engine.cpp:1053 fill_expert_cache`） | 专家 VRAM 常驻与准入策略 |
| 3 | **FileExpertSource / 补集 RAM 副本 / resident-budget** | `src\core\expert_source.cpp`；`generate.cpp:460-463/7086-7099` | **无**（poly 只有 `STRATA_PIN_ARENA_GIB` 原地钉 `engine.cpp:1000-1046`） | 专家在"文件/页缓存/pinned RAM/VRAM"四层间组织 |
| 4 | **SessionLoop + 门铃 + token graph** | `src\core\session.cpp:205-980` | **无**（poly 有 per-layer graph `engine.cpp:2951-3012`，但无 session loop/门铃） | decode 逐层 pre/post 与 host 解耦 |
| 5 | **Verify / 投机窗口** | `src\core\verify.cpp` + `src\spec\*` + `src\core\mtp.cpp` | **半**：链了 `strata_spec`（draft_policy/suffix_drafter 被 include），**无 verify/MTP** | 投机解码与验证 |
| 6 | **conversation cache（program 版）** | `src\program\conv_cache.hpp` + `src\core\conversation_*` | **无**（poly 自研 `src\poly\conversations.hpp`） | 前缀复用/停车/会话恢复 |
| 7 | **layer split / 多卡** | `src\core\layer.cpp` + `generate.cpp:5600-5700` 区间 | **无**（poly 有 `Options::layer_split` 骨架 `engine.hpp:78/engine.cpp:272`，单靶未用） | 多卡分层的 stage/窗口 |
| 8 | **batch / batch-groups** | `src\program\batch_groups.hpp`；`generate.cpp:4477-4552` | **无** | serve 多请求并行 |
| 9 | **pipeline windows** | `generate.cpp:3507-3542`；`src\core\verify.cpp` 窗口 | **无** | 多卡窗口流水 |
| 10 | **PLE / n-gram 表** | `src\ngram\*`；`--ple-gguf` | **不适用**（Nemotron 无 PLE） | prompt-embedding 表 |
| 11 | **VRAM elastic / 分段专家缓存** | `src\core\vmm.cpp`；`generate.cpp:797` | **无** | 弹性显存 |
| 12 | **计时/测量框架** | `--stage-timing` `generate.cpp:5896-5924/14019`；`--gpu-stages` `:6149`；`--gpu-only-full` `:6337`；`--stats` `:14120+` | **无** | 分段计时/真 GPU 地板/print-exit |

---

## §5 C：prefill 剩余杠杆（新靶子：compute 2.941 s/77.5% · 气泡 0.400 s · 未藏 0.289 s；DMA 45.23 GB@23.2 GB/s、85.2% 重叠）

### 5.1 逐项核查表（"能否用 + 为什么 + 要不要编译 + 重复造轮子"）

| # | 上游结构/框架 | 能否用于我们 | 为什么 | 要编译？ | 重复造轮子核查（三列：现成件 / 前提 / 我们前提差异） |
|---|---|---|---|---|---|
| 1 | **`--gpu-stages`**（三段捕获测具） | **不能直接压 compute/气泡** | 它是**测具**："capture the layer as three graphs (mixer / ffn+router / post) and time them from OUTSIDE the capture … **Prints and exits**"（`C:\sp3\help.txt:162-164`；代码 `generate.cpp:2205/6149`）。poly 无 session 三段图可测 | 要用=移植（编译） | 现成件=上游测具；前提=program 线 session 图；我们=poly 无；差=实现面不同 ⇒ **只能借"分段计时"思想** |
| 2 | **`--gpu-only-full` / `--graph-only`** | **不能加速** | 测具：`--gpu-only-full` = "replay pre+post for all 48 layers + LM head, no pool — THE TRUE PER-TOKEN GPU FLOOR"（help:151-154；代码 `:1979/:6337`） | 要=编译 | 同上；口径思想可借（我们已有 nsys 实测），**不必移植** |
| 3 | **`--stage-timing` + `--stats`** | **有间接用（测量）** | 上游按段报 **kernel-count shares**（`generate.cpp:5907-5924` 命名 16 段：gr_read/attn/gr_write/gr_read(ffn)/moe_route/moe_finish/…/gdn 8 段），并明说"未捕获段的 event interval 含 host gap，要先 `--gpu-only-full`"（help:149-155）；poly 无 | 要用=移植（编译） | 现成件=上游测具；前提=program 线分段；我们=poly 无 ⇒ **要拆我们 compute 2.941 s 的段构成，最便宜仍是 nsys；若要程序内计数需移植** |
| 4 | **`--trim-stage-weights`** | **不适用** | 只对显式 `--layer-split` 生效（help:111；`generate.cpp:3073-3084`：非多卡会告警/忽略） | 否（不适用） | 现成件=多卡省显存；前提=多卡；我们=单卡 4080S ⇒ **无差异可借** |
| 5 | **`--pipeline-windows`** | **不适用** | 只在**两卡 layer split 的 serve** 上把窗口交叠（help:107-110；`generate.cpp:3507-3542`，需要 `stages` 且 `batch==0` 等条件） | 否 | 现成件=多卡窗口流水；前提=≥2 stage；我们=单卡 ⇒ 不适用 |
| 6 | **`--batch` / `--batch-groups`** | **不适用（对当前单请求靶）** | serve 多请求并行的吞吐杠杆（help:100-106；`batch_groups.hpp`）；单 prompt prefill 没有"多请求"可并行 | 否 | 现成件=serve 多槽；前提=serve 负载；我们=CLI 单 prompt ⇒ 不适用（**若转服务场景再议**） |
| 7 | **PF_FUSED**（`moe_fused.cu` / `moe_fused_iq.cu`） | **对我们的 Q4_K_M：不适用（CUDA）** | 上游 **CUDA prompt 的 K-quant（Q4_K/Q5_K）默认走 FP16 路径；fused 的 K-quant 只在 gfx11（`STRATA_PF_FUSED_KQ=1`）**（`PS\docs\QUANTS.md:37-38`）；且 3.8 靶本机实测 fused 仅 **+5%**（B/`P3\回执\长prompt-3.8对拍:16`） | 要=大移植（内核族+布局） | 现成件=上游融合 int8 内核；前提=Q2_0/IQ native pack + gfx11/opt-in；我们=Q4_K_M + sm_89 + poly 不链 prefill ⇒ **不匹配，不是重造但不值得先做** |
| 8 | **FP16/BF16 prompt 计算路径**（上游 `Prefill` 默认：BF16 权重+BF16 激活的张量核 GEMM，`include\strata\prefill\gemm.hpp:1-8`；FP16 专家路径 + `DQ=2` 反量化 ring，`prefill.cpp:315/1307/3932-3951`；`QUANTS.md:37` "Q4_K CUDA prompt = FP16 path"） | **是最直接的 compute 杠杆（候选 #1）** | 我们的 prompt 计算走 **int8 MMQ**：`engine.cpp:2031-2033`（量化权重批量走 `mmq_->mul`）、`moe_rows` 的专家 GEMM 也是 `mmq_->mul`（`engine.cpp:2716/2684` 区间）；只有 F32/BF16 权重才走 cuBLAS FP32（`:2025`）。上游对同一类 Q4_K 默认"反量化到 FP16 + 张量核 GEMM" | **要**（移植 FP16 专家路径/DQ ring + BF16 dense GEMM；改求和 ⇒ **必须过 ids/max|Δp| 门**） | 现成件=上游有、我们没有；前提=program Prefill + BF16/FP16 布局；我们=poly MMQ 布局 ⇒ **需移植，不是重造**；**先做对照实测（MMQ vs FP16）再决定** |
| 9 | **`mmq_kq` / `strata_kq`（`STRATA_MMQ_KQ=1`）** | **不需要抄** | 上游 `PS\src\prefill\mmq_kq_lib.cu:1` 原文："**strata_kq, the shared library with llama.cpp's MMQ prompt kernels for the K-quants**"；loader `mmq_kq_loader.cpp:67-110`，`moe_mmq.cu:24-31` 说明"CUDA 上这些内核就是 strata_kq 库，只有开了才打开"。**我们的 `PS\src\poly\mmq.cu:1-2` 原文："The kernels are compiled from the llama.cpp checkout the build already takes ggml from"**，类型表 `:16-18` 含 **Q4_K/Q5_K/Q6_K** ⇒ **我们用的就是同一批 llama.cpp MMQ K-quant 内核**（上游只是打包成 DSO、可选编） | 否（除非要做版本/调参对照） | 现成件=同一批 llama.cpp 内核；前提=llama.cpp 版本 + DSO 打包；我们=已直接编入 poly ⇒ **不是重复造轮子，是同一上游的两种打包**；唯一可查的是版本差（未验） |
| 10 | **`wmma_gemm.cu`** | **不适用** | 用途集中在 HC down 投影的**逐位复现**（`gemm.hpp:bf16_hcd_exact`，`STRATA_HCD_EXACT`）和 HIP 的 WMMA 可选路径；Nemotron 无 HC 投影 | 否 | 现成件=GLM/Qwen 专用；前提=HC 层；我们=无该结构 ⇒ 不适用 |
| 11 | **`--prefill auto` 选块规则** | **无增益** | 规则 = "缓冲能放进所借槽的最大块（≤8192）"（`generate.cpp:536-546/2041-2060/6692-6744`）；我们 prompt 6792 < 8192，chunk 已经不可能更大（`--prompt-batch 6912` 单块），**再调只是 0** | 否 | 现成件=上游选块；前提=borrow+ring 预算；我们=单块已到顶 ⇒ 无差可借 |
| 12 | **设备侧路由**（上游 "decided on the device"；`generate.cpp:6480` 打印；R4 token-graph hit path） | **是压气泡 0.400 s 的正确方向，但"半现成"** | 我们的气泡**已钉死 = host CPU 做 top-22 选择**（R12：39 次、0.350 s，气泡内**零 CUDA API、零同步**；`nemotronhmoe.cpp:362-385` 每 token 22×512 次全扫）。上游把路由放设备（`generate.cpp:6480` "decided on the device"）；但上游 `native_router.cu:236` 的 GPU topk **只支持 (512,10)/(256,8)，我们是 (512,22)** ⇒ **不能直接开** | 是（要扩 kernel 支持 (512,22)）或**否**（host 单遍 top-22，~15 行，R12 建议） | 现成件=上游设备路由框架；前提=其模型 top-k 形状；我们=(512,22) ⇒ **半现成，需改**；host 单遍是本靶最便宜的一刀 |
| 13 | **token graph（默认 one graph per token；`--no-token-graph` A/B）** | **对 prefill 气泡无效** | 上游默认"每 token 一张图"（help:139 `--no-token-graph A/B: two graphs per layer … instead of one per token`）——那是 **decode** 框架；我们的气泡 = host 在算路由（R12：气泡内无 launch API）⇒ **CUDA graph 藏不住 CPU 计算** | 否（对气泡） | 现成件=decode 图；前提=decode 路径；我们=气泡在 host ⇒ 不适用 |
| 14 | **`--vram-elastic` / `--expert-cache-per-layer` / profile 写回** | **不压 compute/气泡** | 都是"显存/驻留/命中"类（`generate.cpp:797`、`:5433`、`:633`），新靶子里 compute 与气泡是时间项，不是容量项 | 否 | 属另一轴（驻留），不在本问题内 |

### 5.2 三个判断（给 Lead 的 C 结论）

1. **DMA 已不是瓶颈（A/B，R11），但 compute 那 2.941 s 的"77.5%"与"≈55 TFLOPS 有效"要打问号（D→未验）**：R1-收工 §8.5.1 说"compute 接近打满、别指望大砍"，但按公开规格，RTX 4080 SUPER 的稠密 FP16 张量算力 ≈ 165 TFLOPS、INT8 ≈ 330 TOPS（C/D，厂商规格，本机未实测）⇒ 55 TFLOPS（或 TOPS）只到峰值的 **1/6~1/3**，**"打满"这个结论缺硬件峰值口径**。⇒ **不能据此关闭 compute 优化**（尤其 §5.1-8 的 FP16/BF16 路径对照）。
2. **气泡 0.400 s 的根因在 host，不在 GPU/框架**（R12 已钉死：纯 host top-22 + 表构建，0 CUDA API 0 同步）⇒ **上游 14 项里没有一条 flag 能直接消它**；最便宜两刀：①host 单遍 top-22（~15 行、无数值变化）；②设备侧路由（需扩 (512,22) kernel，编译）。**CUDA graph / pipeline-windows / gpu-stages 都不是治它的药**。
3. **真值得移植的 framework 只有两件**：**FP16/BF16 prompt 计算路径**（压 compute，需编译+数值门）与**设备侧路由**（压气泡，半现成）；其余 `--gpu-*`/`--stage-timing` 是**测具**、`--trim/pipeline/batch` 是**多卡/serve 专用**、`mmq_kq` 是**同批内核的另一种打包（不需要抄）**。

### 5.3 "未藏副本 0.289 s" 与 ring 槽深（结构差，D）

上游 ring 的槽数（IQ3_S pack = **199 槽**、Q2_0 = 384~1024 槽，§1.2）本质是**深流水**；我们的 `STRATA_STAGE_RING2` 是 **2 个 round 槽（每槽 341 专家 = 1,007 MiB，共 2,014 MiB）**（`P3\施工-ring双缓冲设计:9/23-25`；A3 实测 −19% 墙，B/`回执\R9-ring-reserve3000:10`）。⇒ 槽深 2 vs 199 是**结构差**，但 R11 已测"未藏仅 0.289 s"，**上限 ≈ 8% 墙**；加第三槽要多 1,007 MiB，ring ON/reserve 3000 时余量只剩 **1,804 MiB**（`R1-收工:67`）⇒ **装不下满槽**，只能改小槽/换布局（需编译）。**这条不是当务之急**（未藏 0.289 < 气泡 0.400 < compute 2.941）。

---

## §6 重复造轮子核对（常设职责；只给建议，不否决）

| 候选 | 现成件（file:line/KP-*） | 它的前提 | 我们前提的差异 | 建议（Lead 裁） |
|---|---|---|---|---|
| **ring 机制** | 上游 per-blob N 槽 ring（`prefill.cpp:132-134/292/1860-1936`）+ split ring（`split_ring.hpp`）；**KB：`KP-NINFER-0069`（ring 可抄件清单 P0 四条）、`KP-STRATA-0047`（"RING 加大"不适用）** | program Prefill + per-blob 布局；槽数受 pinned-share/字节预算 | 我们已自研 **round-slot 2 槽**（`施工-ring双缓冲设计:17` 说明为什么不能照抄 per-blob：mmq 要 up/down 连续批布局，照抄会丢 pin 直 DMA） | **不是重造**：思想同源、实现因布局不得不改；**不要按"更大 ring"硬加槽**（KB `KP-STRATA-0047`：RING 加大不适用） |
| **K-quant prompt MMQ** | 上游 `strata_kq` DSO（`mmq_kq_lib.cu:1`、`mmq_kq_loader.cpp:67-110`） | llama.cpp MMQ 内核 + 可选编 | 我们**已用同一批 llama.cpp 内核**（`poly/mmq.cu:1-2,16-18` 含 Q4_K） | **不需要抄**；最多对一下 llama.cpp 版本（未验） |
| **FP16/BF16 prompt 路径** | 上游 `gemm.hpp:1-8` + `DQ=2` ring（`prefill.cpp:315/1307/3932-3951`） | program Prefill + BF16 布局 + 数值门 | poly 走 int8 MMQ（`engine.cpp:2031-2033`） | **需移植（非重造）**；先实测对照 |
| **CUDA graph** | 上游 token graph（decode，help:139）；poly 已有 per-layer graph（`engine.cpp:2951-3012`） | decode 窗口 | 我们 prefill 无图；但气泡是 host CPU，图无用 | **不需要移植**（对 prefill 气泡无效） |
| **会话 checkpoint** | 上游 `--prompt-cache-every`（默认 16384，serve-only，`generate.cpp:647`） | serve 会话复用 | poly 自研 `conversations.hpp`（`every=2048`）⇒ 我们已用 `--prompt-cache-every 8192` 修 4-call | **不是重造**（两套不同实现；我们的 cc 是 poly 特有） |
| **测具口径** | 上游 `--stage-timing/--gpu-stages/--gpu-only-full` | program 分段图 | 我们靠 nsys | **不需移植**；R12 已钉气泡 |
| **rlab 库** | `KP-XOFFLOAD-0013`（预取判据）、`KP-STRATA-0047`（RING 加大不适用）、`KP-NINFER-0007`（重叠族天花板）、`KP-UNI-0250`（prefill/decode 分工） | 见 task-5 报告 | — | 作判据用（可直接用） |

---

## §7 最不确定 / 未验

1. **老师那次的 ring 槽数 N 未验**（A2 只证明"开了"）：generate 模式不打印槽数；需 §2.3 的 env 探针一跑。
2. **0.1.40.2 源码本机没有**：代码条件用 0.1.44.1 代理（`prefill.cpp` 两树 sha 相同，但那只是 0.1.44.1 内部一致；0.1.40.2 的默认值可能不同）⇒ §2.1(iii) 是"代理证据"，强度 B。
3. **`wait for rings` 的完整链条只到 session.cpp 的 pre/post 门铃**；再往下"门铃 kernel 名/谁写 flag"未逐行追（本轮到 `session.cpp:686-723` 为止）。
4. **§5.2-1 的硬件峰值质疑是 C/D 级（厂商规格，非本机实测）**：55 TFLOPS/TOPS 的分子口径（是 FLOP 还是 int8 OP？）也尚未与 R11 对齐 ⇒ **"compute 接近打满"不应当作定论**。
5. **FP16/BF16 prompt 路径对我们的加速幅度未验**（只是"上游默认它、我们默认 MMQ"的结构差）——必须先实测同题对照，且过 ids/max|Δp| 门。
6. **poly 的 `gemm_f32` 用量面**：`lin()` 只有 F32/BF16 权重走 cuBLAS（`engine.cpp:2019-2025`），量化权重走 MMQ（`:2031-2033`）；本靶 Q4_K_M 的 dense 权重是否也有 F32/BF16 残余未逐张核（未验）。
7. **`--gpu-stages` 的三段捕获能否平移到 poly**：没有 poly 侧的 `session_capture(split=true)`，判断"要移植"是结构判断，未试编译。
8. 行号对应当前工作树（`PolyStrata` 有未提交改动）；上游机制文件与 `UP` sha 同。

---

## §8 复现（只读命令）

```powershell
# A1 ring 形态
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\prefill\prefill.cpp' -Pattern 'ring_slots','stream_all','m.ring','m.copied','m.used','DQ =' -SimpleMatch
# A2 老师 ring 开没开（streamed ?= resident 差额）
foreach ($f in @('dec-pool2.txt','trace-work.log','trace-long32k.log','trace-tiny.log')) { Select-String -LiteralPath ('E:\strata\_staging\' + $f) -Pattern 'experts streamed','resident' -SimpleMatch }
# A2 "wait for rings" 出处判据
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\core\session.cpp' -Pattern 'poll for the ring','rings_mid_graph','ended without ringing' -SimpleMatch
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\program\generate.cpp' -Pattern 'wait for rings' -SimpleMatch
# B 框架件：poly 的 include 面
Select-String -Path 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\*' -Pattern '#include "strata/' -SimpleMatch
# C 计算路径差（上游默认 FP16 vs 我们 MMQ）
Select-String -LiteralPath 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\engine.cpp' -Pattern 'gemm_f32','Mmq::fits','mmq_->mul' -SimpleMatch
```
