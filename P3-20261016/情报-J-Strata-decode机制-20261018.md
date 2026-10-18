# 情报-J · Strata decode 机制全清单（线 A：老师每轮账拆解 + 有/无 + TOP-5）— 2026-10-18

- **席位**：scout-a（情报）· **任务**：计划 `计划-decode总攻-20261018.md` §2 线 A（零 GPU）。
- **状态**：**达成**（① 24.3 ms/轮 四段账拆开并闭合；② 机制清单 14 条（每条 `file:line` + 宣称收益 + 我们全量计数 + 移植代价/风险）；③ 机制 vs 几何判决；④ TOP-5）。
- **口径/纪律**：零 GPU、零编译、不改码、未 spawn 子代理。老师侧一律 **C/D**（外部引擎源码与日志）；我们侧 **A**（本机代码/日志可复算）。"没有"一律给**全量计数**（见 §5 扫描方法）。
- **我们树读取时点**：`PolyStrata\src\poly` 全量 38 文件 / 850,087 B；`engine.cpp` 208,204 B（2026-10-18 02:02）、`engine.hpp` 53,451 B（02:25）、`main.cpp` 60,345 B（02:26）——**当时有施工在落地**（pcie-frac 与线程放置已进代码，见 §2）；行号/计数以落地时 grep 为准。

---

## §0 三句结论

1. **`wait for rings 12.378 ms/round` 不是在等数据，是在等 GPU 自己**：ring（门铃）由**被捕获的 GPU 图内**的 `doorbell_publish` 在每层"pre 半段"（hc→mixer→router）算完时写入 mapped-pinned `seq`（`src/core/layer.cpp:376,385`；`src/core/verify.cpp:525`），host 只是自旋读这个字（`verify.cpp:792-818`），`ms_wait` = Σ(看到 ring 的时刻 − 进入等待的时刻)（`verify.cpp:843`）⇒ **它就是 GPU 每层 pre 半段的执行时间，48 层合计**。`pool 7.269 ms` 是**每层调一次 CPU 专家池**的挂钟（`verify.cpp:826,844`），它**是 host 在算**（不是等），但**卡住 GPU**：图内的 post 半段要等 pool 发布的 flag（`verify.cpp:558/587/595` 的 `wait_flag_ge`）才继续 ⇒ 每层 GPU 与 CPU 是**握手串行**，不是自由流水。四段账**闭合**（见 §1.3）。
2. **机制清单**：14 条里，我们**已有** 6（CPU 池、GPU/CPU 重叠形式、adaptive/LFU、tail-skip、MTP 壳、pcie-frac 刚落地）、**刚落地待验** 2（pcie-frac 02:02、host-core/aux-cpus 02:25）、**没有** 6（门铃、整窗一张图、窗口级计时、GPU 路由、kv-resident、batch/serve 队列）。**老师真正比我们省的是"每层 host 往返"与"层/专家数"**，不是它的 CPU 池（它的池每漏专家 133 µs，我们只有 27 µs——**我们更快**）。
3. **判定：机制与几何各占一半，但顺序是"几何压上限、机制压每层常数"**。老师 9.24 ms/token 的构成：GPU pre 4.71 + CPU 池 2.76 + MTP 草稿 0.90 + 尾/杂 0.82 + host/commit 0.05。我们 75.0 ms/token：CPU experts 11.1 + router wait 6.6 + 其余 57.3。几何（层 48 vs 88=×1.83、top-k 10 vs 22=×2.2、命中 96.5% vs 52.7% ⇒ 每 token 漏专家 ≈21 vs ≈416=×20、投机 2.63 tok/轮 vs 无）解释了 **CPU 池差与轮成本摊薄**；机制（门铃+整窗一张图+GPU 路由）解释我们 **router wait 6.6 + 57.3 里未拆的 host launch/同步**。⇒ **该抄，但只抄两条"每层常数"的机制 + 两条"几何"的机制**（§4）。

---

## §1 老师 24.3 ms/轮 的账：逐段拆解（C 级：0.1.40.2 源码 + `dec-pool2.txt`）

### §1.1 四段的定义（源码证据）

| 段（日志名） | 老师代码里是什么 | `file:line`（`E:\strata`，v0.1.40.2） | 计时口径 |
|---|---|---|---|
| **wait for rings 12.378** | `Verifier::run` 的逐层门铃等待：`a=now` → 自旋 `while (*seq < want)` → `b=now`；`ms_wait += b-a`。`seq` 是 **GPU 图内**写进 mapped pinned 内存的 ring 序号；ring 由该层 pre 半段末尾的 `doorbell_publish`/`doorbell_ring` 发射 | 等待与累加 `verify.cpp:796-818, :843`；ring 的写 `verify.cpp:525`、`layer.cpp:376,385`；结构注释 `verify.cpp:350`（"pre(l, group): up to the ring"） | **等 GPU**（每层 pre 半段 = hc read + mixer(GDN/QSA) + router + shared），48 层合计 |
| **pool 7.269** | 每层调一次 **CPU 专家池**：`pool(user, h_x_+…, h_ids_+…, n, k, h_ymiss_+…, l)`；`b=now → 调用结束`，`ms_pool += ms_since(b)`。池是 `ExpertPool`（专用线程池、批任务、相位计时）；日志另印它的内部账 `pool multi gate/up 4.923 quantize 0.016 down 1.945 ms/round`、`13.4 GB/s over the rows phases`、`CPU pool call 7.159 ms/round` | `verify.cpp:824-827, :844`；`dec-pool2.txt:71-73`；池实现 `src/kernels/cpu/pool.cpp:176-190, :419-483`；相位口径 `pool.hpp:141-151` | **host 在算 CPU 漏专家**（每层 ~1.14 routed 漏专家）；它**发布 flag 让 GPU 的 post 继续** ⇒ GPU 被它串住 |
| **host 0.101** | 窗口前的 host 准备（token/step/pos/PLE 行 staging + flags 复位），`ms_host += ms_since(t0)`（每次 `run()` 即每轮一次） | `verify.cpp:761-787, :786` | host CPU 准备（每轮一次，极小） |
| **commit 0.037** | `Verifier::commit`（KV/门铃收尾的捕获图 launch + sync） | `verify.cpp:1058-1073`（`:1073 ms_commit += ms_since(t0)`） | 每轮一次，极小 |

### §1.2 每段"在等谁/在算什么"（一句话）

- **wait for rings**：**等 GPU**。它测的是"从 GPU 可以开始跑第 l 层 pre 半段"到"该层路由 ids/weights 被发布"的时间 = **GPU 的 pre 半段时长**（不是等 CPU，也不是等 DMA；0.1.40.2 的 verify 路径里 12.378 ms 全是 GPU 计算）。日志里 12.378/48 ≈ **258 µs/层**（若按 `groups_`=2 分两组则是 96 次等待、129 µs/组）。
- **pool**：**host 的 CPU 专家计算**（不是"等待"）。它之所以重要，是因为**GPU 被它挡住**：图内 post 半段先等 pool 早发布出的"GPU plan"（flagA，`verify.cpp:558`；设计见 `expert_source.cpp:3720-3721` "the GPU's share, decided and published FIRST so the GPU starts while the CPU works"），再等"PCIe 份额已到"（flagB，`:587`），最后等"CPU 份额已写进 mapped rows"（flag，`:595`）。⇒ **每层：GPU pre → host pool → GPU post 握手串行；只有 GPU post 与 CPU pool 这两者重叠**。7.269/48 ≈ **151 µs/层**，对应 1.14 routed 漏专家 ⇒ **133 µs/漏专家**（3 线程池，日志 `participating threads: 3`、`CPU experts 0.93 distinct / 1.14 routed per layer`）。
- **host / commit**：host 侧簿记，合计 0.138 ms/轮 ≈ **0.05 ms/token**——即老师把"每层 host 往返"完全藏进了图 + 门铃里（见 §2-1/2）。

### §1.3 账本闭合（★ 比"残差 4.5"更准的拆法）

日志 `dec-pool2.txt:79`：200 tokens / 1847.9 ms = **76 轮**（`:67` "76 rounds of 6"）⇒ 轮墙 = 1847.9/76 = **24.31 ms/轮**。逐项：

| 项 | ms/轮 | 出处 |
|---|---|---|
| wait for rings | **12.378** | `dec-pool2.txt:71` |
| pool | **7.269** | `:71` |
| host | **0.101** | `:71` |
| commit | **0.037** | `:71` |
| **MTP 草稿（单列，但在这 24.3 里面）** | **2.360** | `:76`（`2.360 ms/round drafting (75 rounds)`） |
| 尾/杂（窗口收尾 `cudaStreamSynchronize` + 图 launch + head/sampling + 循环开销） | **≈2.17**（=24.31−22.15，残差） | 推算（D）：`verify.cpp:846-850` 的窗口尾巴不在四个计时器里 |
| **合计** | **24.31** ✓ | 闭合 |

> **⇒ 队长原来的"残差 4.5 ms（18.6%）"可再拆开：其中 2.36 是 MTP 草稿、≈2.1 是窗口尾巴/采样**（A 级日志 + D 级残差归因）。

### §1.4 每 token 摊薄（×2.63 token/轮）

| 项 | ms/token | 说明 |
|---|---|---|
| GPU pre（rings） | **4.71** | 12.378/2.63 |
| CPU pool | **2.76** | 7.269/2.63 |
| MTP 草稿 | **0.90** | 2.360/2.63 |
| 尾/杂 + host/commit | **0.87** | (2.17+0.14)/2.63 |
| **合计** | **9.24 ms/token = 108.2 tok/s** ✓ | 与 `:79` 自报一致 |

---

## §2 机制清单（14 条；老师侧 C，宣称收益带出处；我们侧 A + 全量计数）

> 我们侧扫描：`PolyStrata\src\poly` **38 文件 / 850,087 B**，计"匹配行数"全量（无截断）。计数为读取时点快照（pcie/线程两条是 02:02/02:25 刚落地）。

| # | 机制 | 老师 `file:line` | 它干什么 | 宣称收益（出处） | 我们有/无（全量计数） | 移植代价 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 | **门铃（mapped-pinned seq + 图内 GPU 写）** | 写：`core/layer.cpp:376,385`、`core/verify.cpp:525`；host 自旋：`verify.cpp:792-818, :843`；单机说明：`session.cpp:634-667` | 每层 pre 半段一完，GPU 把 ids/weights + 递增序号写进 `cudaHostAllocMapped` 内存；host 自旋读一个字即可接管 | host+commit 合计 **0.138 ms/轮 = 0.05 ms/token**（`dec-pool2.txt:71` 反推） | **无**：`doorbell` 0、`h_seq` 0（38 文件全量） | ~250–400 行（mapped 环 + host spin + 去掉逐层 event sync）；需编译 | 中：自旋与内存序（老师注释：驱动必须每轮进一次才能看到写，`session.cpp:643-650`）；观测掩盖 |
| 2 | **整窗一张 CUDA graph（图内含 flag 等待）** | 捕获：`core/verify.cpp:655-685`；launch：`:788`；图内等待：`:558/587/595`；提交图：`:687-745` | 一个 verify 窗口（48 层）**一次 launch**；每层在图内等 pool 的 flag | 与 #1 合计 host≈0.05 ms/token；另：我们注释里 RTX 5090 逐层记录间隙 2–4 µs/层（`engine.cpp` 注释，B） | **部分**：`cudaGraph` 10 行——只有逐层/逐 token 图（n≤window 或 n==1），**无整窗一张图** | 大：400–600 行（须先有 #1；host 路由/cpu 池依赖仍要在图内用 flag 表达）；需编译 | 高：捕获期禁同步（KB `KP-NINFER-0015`）、时序/确定性 |
| 3 | **CPU 专家池（专用线程池 + 批任务 + 相位计时 + host 参与 drain）** | `kernels/cpu/pool.cpp:176-190, :419-483`；口径 `pool.hpp:80-151` | 多线程算漏专家；host 线程也 drain；park/drain/re-park 三段计时 | 5 worker **36.32 GB/s**、host 加入 **33.7 GB/s**（`pool.hpp:80-101`） | **有**：`pool` 124 行 / `Workers` 67 行（`cpu_experts.cpp`）；**且我们每漏专家 27 µs vs 老师 133 µs**（我们更快：24 线程 vs 3 线程） | 不需要抄（只可借"相位计时"口径） | 低 |
| 4 | **pool 先发 GPU plan、GPU 先算驻留/PCIe 专家**（P6/R4.2c） | `core/expert_source.cpp:3720-3721, :196-252` | CPU 池一开始先把"哪些专家给 GPU"发布，GPU 同时开算；CPU 算自己的份额 | 老师每轮 pool 与 GPU post 重叠（账见 §1） | **有等价物**：GPU launch 通过回调交给 CPU 线程做（`engine.cpp:2255-2257` 传 `&launch`） | 不需要（形式不同） | 低 |
| 5 | **MTP/spec verify 窗口（2.63 token/轮）** | `core/mtp.cpp:281-597`（capture_step/draft）；窗口驱动 `verify.cpp:752-850` | 一轮提议 6 个草稿、verify 一次出 2.63 个 token | **2.63 token/轮**（`dec-pool2.txt:67`）；把 24.3 ms 摊成 9.24 ms/token | **有壳**：`draft` 186 行 / `mtp` 9 行 / `spec` 17 行；但 ids 漂、只 +2.2%（线 B） | 已移植（线 B 修 ids） | 数值/显存（线 B 已知） |
| 6 | **tail-skip（ROUTE_TAIL_SKIP=7 默认）** | `expert_source.cpp:3747-3769`；默认开 `b4e3407b` | 窗口内**每个 token 都排在 rank≥7 的漏专家**不拷不算 | decode **1.13–1.16×**（0.1.42 docs C；单卡 5/5 对） | **有但关着**：`tail_skip` 13 行 + flag/env；默认 0、我们跑法 `--tail-skip 0` | 无需编译（改参数；换算：老师 7/10 ⇒ 我们 **15/22**） | 改输出（KL 0.003–0.006，老师数） |
| 7 | **pcie-frac（GPU/CPU 分流漏专家）** | `expert_source.cpp:3770-3853`（m=(nmiss×pcie_num)>>8；pinned 源→staging；plan 先发布） | 一部分漏专家搬上 GPU 算，其余 CPU；挪 PCIe 的活与 CPU 并行 | 老师 decode 里 **0.16 distinct/layer**（share 90/256）（`dec-pool2.txt:75`）；自动 share 在小卡 **1.300×**（0.1.42 docs C） | **刚落地（02:02）**：`pcie` 107 行、`--pcie-frac/--pcie-slots` 已有，`--pcie-auto` 仅解析（"not in this build"）；`frac` 28 行 | 后续：auto 拟合 ~80 行（抄 `generate.cpp:1736-1770`） | 数值（GPU/CPU 末位差）；确定随 auto 漂 |
| 8 | **adaptive tier（每 4 轮换 818 专家）** | `core/expert_cache.cpp:642-731`；开关 `generate.cpp:4067-4069` | VRAM 槽按使用衰减换入/换出（LFU-衰减），预测由 router look-ahead 喂 | **0.175 ms/轮**、每 4 轮 818 个（`dec-pool2.txt:74`） | **有**：`adaptive` 6 / `recent_` 6 / `expert_swaps` 8 行（`follow()` 有收益门+异步到达） | 差在旋钮化（~40 行，默认=现值） | 低 |
| 9 | **per-layer 槽配额** | `expert_cache.cpp:642`；help `:197-203` | 每层自己的槽，防"共享计数器被前几层吃光" | 老师默认（到达序）2.97% → per-layer **8 槽/层 21.4%、64 槽/层 70.4%**（help C） | **无**（`per_layer` 0 命中）——**且我们 task-8 A 级：全局计数贪心已最优，等配额 −0.39 pt ⇒ 不该抄** | 不建议 | — |
| 10 | **host-core / aux-cpus / pool-*（线程放置）** | `kernels/cpu/pool.cpp:74-96, :401`；`platform/aux_cpus.hpp:1-361`；`session.cpp:561-572` | host 线程避开 GPU 中断核（Windows 中断在第一个逻辑核）；aux 线程挪空闲核 | aux `auto` vs off **+11%**（2×3080，0.1.43.1 docs C） | **刚落地（02:25）**：`host_core` 12 / `aux_cpus` 11 行；`affinity::plan_threads(host, aux, threads-1)` + `pin_this_thread`（`engine.cpp:1660-1662`）；flag `main.cpp:940-941` | 已落（待 A/B 验） | 零精度风险（只挪线程） |
| 11 | **kv-resident（KV 常驻 VRAM）** | `generate.cpp` 的 `--kv-resident`；docs 长上下文 | 长上下文才需要 | 老师 `--kv-resident 32768`（dec-pool2 旗标） | **无**：`kv_resident` 0 / `kv-resident` 0 | 不适用（我们 16K 上下文、KV 小） | — |
| 12 | **batch / serve 队列** | help `:100-106`（--batch/--batch-mtp/--batch-groups） | 多请求合批解码 | 多路吞吐，非单条延迟 | `batch` 192 行存在（我们 serve 不同）；无多 slot 合批 | 不适用（batch=1） | — |
| 13 | **fused MoE 内核（PF_FUSED 系）** | `src/prefill/moe_fused.cu`、`moe_fused_iq.cu` | prompt 专家融合；**decode 不覆盖** | prompt +16–22%（5070，C），我们机 +5%（A/B） | `fused` 15 / `moe_fused` 0；`native_mmvq` 41 | 与 decode 无关 | — |
| 14 | **GPU 侧路由（top-k 内核）** | decode：`native_router.cu:228 native_router_top10_multi`；prefill 版 `prefill/kernels.cu:2016-2023, :1491-1528` | 路由 top-k 在 GPU 上算，host 只拿 ids/weights | 无单列收益；但它使"每层 host 往返"变小/可门铃化（老师 host 0.05 ms/token 的前提） | **无**：`native_router` 0 / `router_topk` 0 / `topk` 0；我们 decode 在 **host** 上算（每层 D2H 512 分数 + event sync：`engine.cpp:2139,2156-2157`，`stats_.route_wait_ms`=**6.6 ms/token**） | 中：~150–250 行+新内核（top-22+sigmoid+bias；`router_top10.cu` 已在链接面但只 10/8） | 数值：ids 需逐位判 |

---

## §3 ③ 判定：机制 vs 几何

### §3.1 两侧几何对照（A/ C 混合，各标出处）

| 项 | 老师（Strata 0.1.40.2，Qwen3.8-Flash-Next） | 我们（poly，Nemotron-120B-A12B Q4_K_M） | 倍率 |
|---|---|---|---|
| 层 | **48**（**36 GDN + 12 QSA**，**无 mamba**；`generate.cpp:1874,2136`） | **88**（40 mamba + 8 attention + 40 moe；`pair_D_pb6912.log:2`） | 88/48 = **1.83×** |
| 每条 token 的激活专家 | top-**10**（`prefill/kernels.cu:1511-1525` rank<10；decode 用 `native_router_top10_multi`） | top-**22**（bounds 表 1.16 MiB ⇒ 22；`pair_D_pb6912.log:84`） | **2.2×** |
| 专家总数 | 24,576（48×512；`dec-pool2.txt:19`） | 20,480（40×512；`:86`） | 0.83× |
| 每专家 blob | 2.03 MB（log 自报索引推导） | 3.0966 MB（`pin up 756.0 MB/512`，A） | 1.52× |
| 驻留 | **9,041 / 24,576 = 36.8%**（`:51`；缓存 11,428 槽 21.69 GiB `:34`） | **5,890 / 20,480 = 28.8%**（`:86`；17.0 GiB） | 0.78× |
| 每 token 漏专家（routed） | ≈ **20.8**（1.14/层/轮 ×48 ÷2.63） | ≈ **416**（22×40×(1−52.7%)，A 推算） | **×20** |
| 每漏专家 CPU 成本 | 133 µs（7.269/54.7） | **26.7 µs**（11.1/416） | **0.20×（我们快 5×）** |
| 投机 | **2.63 token/轮**（MTP spec，`:67`） | 无（draft 坏，线 B） | — |
| decode | **108.23 tok/s = 9.24 ms/token**（`:79`） | **13.36 tok/s = 75.0 ms/token**（计划 §0） | **8.1×** |

### §3.2 75 vs 9.24 的分解（算术，D 级）

- **老师 9.24 ms/token**：GPU pre 4.71 + CPU 池 2.76 + MTP 0.90 + 尾/杂 0.87。
- **我们 75.0 ms/token**：CPU experts 11.1 + router wait 6.6 + 其余 57.3（GPU 单 token + 逐层 launch/同步）。
- 差 = 65.8 ms/token，拆：
  - **CPU 池** 差 8.3 ms：几何原因（漏专家 ×20）但被我们每漏专家快 5× 抵消 ⇒ 净 ×4；
  - **host 每层往返** 差 ≈6.6 ms（我们 router wait；老师 0.05）⇒ **纯机制差**；
  - **GPU 每层每 token**：老师 (12.378+尾≈4.5)/2.63/48 = **134 µs**；我们 57.3/88 = **651 µs** ⇒ **4.9×**。其中几何解释部分：MoE 专家数 ×2.2、blob ×1.52；mamba2 层是我们独有（40 层），但老师 GDN 也是 36 层"~13 个 latency-bound 小 launch/层"（`generate.cpp:1874-1876` 注释，C 级；**不能把该注释里的 0.96 ms 当每 token 数**，那是另一测量口径）。
  - **摊薄**：老师 2.63 token/轮 把每轮固定开销（pre 12.4+池 7.3）摊掉 ⇒ 若我们同样每轮固定 75 ms，2.63 token/轮就是 28.5 ms/token（仍 3.1× 老师）⇒ **投机只能救 2.6×，不能救 8×**。

### §3.3 判决

**主要矛盾是几何（层数×专家数×命中率），但"每层 host 往返"与"每层 GPU 常数"是机制差、且可直接抄**：
- 几何决定上限：命中 52.7% → 每 token 416 个漏专家（老师 21 个）⇒ 池 + 串行等待不可避免。**不先把漏专家数压下来（驻留/top-k/tail-skip/pcie），机制优化只剩 6.6+部分 57.3 可拿。**
- 机制决定每层常数：门铃+整窗一张图把老师每层 host 往返压到 ~0.05 ms/token；我们 6.6 ms/token（88 层 × 75 µs）。**这条可直接抄、且有明确机制**。
- **⇒ 不冲突：先抄"最便宜的机制"（GPU 路由 + 门口式发布）⇒ 再抄"几何"（驻留↑/tail-skip/pcie 定参）⇒ 最后才是"整窗一张图"（贵）。**

---

## §4 ③ TOP-5 移植建议（按"杠杆 × 可移植性"排序；每条给 batch=1 的机理）

| 排 | 机制 | 为什么对 batch=1 decode 有用（机理，不是"老师有"） | 移植代价（C/D） | 判定依据（我们侧 A） |
|---|---|---|---|---|
| **1** | **GPU 侧 routing（top-22 内核）+ ids/weights 门铃式发布** | batch=1 时每层都是"路由 GEMM → 512 分数 → host 选 top-k"的小往返：我们的 `engine.cpp:2139/2156` 每层做一次 D2H + `cudaEventSynchronize`，`stats_.route_wait_ms` 实测 **6.6 ms/token（占 75 ms 的 8.8%）**。把 top-k 放卡上（老师 `native_router.cu:228`），host 只读 22 个 ids/weights（88 字节 vs 2 KB+同步），**每层省掉一次 event 往返**；且这是 #2 的前置（host 不再需要完整分数） | ~150–250 行 + 一个 kernel（我们链接面已有 `router_top10.cu`，`native_router_topk_supported` 只支持 (512,10)/(256,8)，要扩 top-22+sigmoid+bias）；**需编译** | A：router wait 6.6；A：逐层 D2H/同步代码 |
| **2** | **整窗/整 token 一张图 + 门铃（host 只自旋一个字）** | 老师把 48 层一次 launch、每层在图内等 pool 的 flag；host 只 spin mapped seq ⇒ host+commit 0.14 ms/轮。我们每层至少一次 launch/同步，88 层 × ~10–75 µs 就是 1–6.6 ms/token 的**纯延迟**；batch=1 无其它 token 可藏，全靠减少"每层串行常数"。**依赖 #1**（路由结果不再需要 host 选） | 大：400–600 行（capture 期禁同步、flag 语义、CPU 池在 loop 中发布）；**需编译** | A：我们 `cudaGraph` 10 行只覆盖逐层/逐 token；`doorbell` 0 |
| **3** | **把每层漏专家数压下来：`--tail-skip 15`（=老师 7/10 的 top-22 换算）+ 驻留 ↑（A 项 +1,010 槽）+ pcie share 定参** | 漏专家 416/token 是 CPU 池 11.1 ms 与串行等待的根；每减 1 个漏专家≈省 27 µs CPU + 一次串行握手。tail-skip 按 route rank 直接砍尾部（老师 decode **1.13–1.16×**，C）；驻留 +1,010 槽按 task-8 收益线 68.30%→73.47%（A/D）；pcie-frac 已落地，按实测 `cpu_us/miss` 定 share（`--pcie-auto` 未落，~80 行） | 参数（无编译）+ A 项（已有方案）+ auto ~80 行 | A：tile 版本已给命中曲线；A：pcie-frac 代码在（02:02） |
| **4** | **`--host-core last` + `--aux-cpus auto` 的 A/B 验证（已落地，零风险）** | Windows 把 GPU 中断派到第一个逻辑核；我们 host 线程每层等 event（spinning），若同位则每层多受中断干扰 ⇒ 只挪线程、不改结果，等于白拿（老师 aux +11%，C） | **0**（已落地，只跑 A/B） | A：`affinity::plan_threads` 已在 `engine.cpp:1660-1662`；`main.cpp:940-941` 有 flag |
| **5** | **MTP ids 修好（拿到 2.63 token/轮 的摊薄）** | 老师每轮固定开销（pre 12.4 + 池 7.3 + 尾 2.2 ≈ 22 ms）由 2.63 token 摊薄；我们 75 ms 若每轮能出 ≥2 token，per-token 接近减半。**这是唯一能一次性拿到 ×2 量级的机制**，但依赖线 B 的 ids 门 | 线 B（builder2） | A：计划 §0；我们 draft 壳在（186 行）但 +2.2%/ids 漂 |

**不抄清单（有理由）**：老师 CPU 池（我们每漏专家快 5×）；Foresight/lookahead（老师实测负收益，0.91–1.00）；kv-resident/batch/serve 队列（batch=1 不适用）；per-layer 配额（我们 task-8 A 级：亏）；fused MoE（decode 不覆盖）；`--mmap-experts`/resident-arena（我们是 mmap+pin 另有路线）。

---

## §5 未验 / 边界 / 扫描方法

- **扫描方法（判"有/无"）**：`Get-ChildItem PolyStrata\src\poly -Recurse -File`（38 文件 / 850,087 B），对每个关键词 `Select-String -AllMatches` 计**匹配行数**（全量，无 `-First N` 截断）：`doorbell 0`、`h_seq 0`、`ms_wait/ms_pool/ms_host/ms_commit 全 0`、`verify window 0`、`cudaGraph 10`、`capture 4`、`pool 124`、`Workers 67`、`cpu_experts 25`、`draft 186`、`mtp 9`、`spec 17`、`adaptive 6`、`expert_swaps 8`、`recent_ 6`、`pcie 107`（02:02 刚落地）、`frac 28`、`kv_resident 0`、`host_core 12`、`aux_cpus 11`（02:25 刚落地）、`pool_workers 1`、`batch 192`、`tail_skip 13`、`foresight 0`、`lookahead 0`、`native_router 0`、`router_topk 0`、`topk 0`、`route_card 13`、`fused 15`、`native_mmvq 41`、`mmvq_il 0`、`pipeline 0`、`serve 86`。
- **【未验】老师 0.1.40.2 的 `groups_` 实际值**（`verify.cpp:794` 的 `G`）：若 G=2，则 12.378 ms 是 96 次等待（129 µs/组）；结论不变，但"258 µs/层"要改。
- **【未验】老师的 2.360 ms/轮 MTP 草稿是否与 pool 重叠**：本件按"加在 24.3 里、不重叠"闭合；若部分重叠，残差再变。
- **【C/D 边界】**：老师源码是 0.1.40.2（`E:\strata` HEAD=v0.1.40.2）；我们读的 `upstream-0.1.44.1` 仅用于交叉，二者差异未逐条核。所有老师性能数都是它机器+它模型，对我们的外推一律 D。
- **【未验】我们 57.3 ms"其余"的构成**（线 C 的活）：本件只指出它 ≥88×每层常数，不拆 kernel。
- **【已查无命中】**：`E:\strata` 全树无 `doorbell_ring_kernel` 字面（注释里出现过，实现名是 `doorbell_ring`，`layer.cpp:385`）——按"已查 X 无命中"记录。

---

## §6 复算命令（只读）

```powershell
# ① 四段账的源码
Get-Content E:\strata\src\core\verify.cpp | Select-Object -Skip 751 -First 100   # Verifier::run: 792-845
Get-Content E:\strata\src\program\generate.cpp | Select-Object -Skip 4049 -First 10   # 打印行
Get-Content E:\strata\src\core\layer.cpp | Select-Object -Skip 370 -First 18   # doorbell_publish/ring
# ② 日志
Select-String -Path 'E:\strata\_staging\dec-pool2.txt' -Pattern 'verify window|pool multi|dispatch|adaptive tier|pcie experts|mtp |decode |speculation'
# ③ 我们侧
Select-String -Path 'PolyStrata\src\poly\engine.cpp' -Pattern 'stats_.route_wait_ms|cudaEventSynchronize|copy2\(rl_'
Select-String -Path 'PolyStrata\src\poly\engine.cpp','PolyStrata\src\poly\cpu_experts.cpp' -Pattern 'affinity::|Workers::'
# ④ 两侧几何
Select-String -Path 'C:\sp3\pair_D_pb6912.log' -Pattern '87 layers|20480|bounds table'
```

**判据（能变红）**：① 若 `verify.cpp:843` 不是 `ms_wait += b-a` ⇒ §1 错；② 若我们用 `Select-String 'doorbell'` 在 src/poly 命中 ⇒ "无门铃"错；③ 若 `affinity::` 不在 engine.cpp ⇒ #4 已落地判错；④ 若老师 `G=2` ⇒ §1.2 的"258 µs/层"改 129。

---

## §7 落盘

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-J-Strata-decode机制-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-decode机制.md`
