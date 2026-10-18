# 情报-O · 本机储备与 Strata decode 架构差（专家图 / 边搬边算 / CPU-GPU 按比例原地计算）— 2026-10-18

- **席位**：scout-a（零 GPU）· **触发（需求记录 2026-10-18）**：先查本机储备，找与 strata 在 decode 优化架构上的差别（**专家图 / 边搬边算 / CPU-GPU 按比例原地计算**）+ lead 派工（**先查本机储备**）。
- **状态**：**达成**（三层全查：① `E:\rlab` 知识库入口三件 + 26 词全量检索；② 既往实验目录 3 套全量 grep；③ Strata 0.1.40.2 / upstream 0.1.44.1 / 我们 poly 三方实现对比 + 2× 判断）。
- **纪律**：零 GPU、零编译、不改码；每条带 `file:line` 或 KP-ID；**全量计数**（未用 `-First N` 截断判定）；"没查到"明写；未 spawn 子代理。

---

## §0 一句话结论

**三样里只有"CPU-GPU 按比例原地计算"有完整本机储备 + Strata 有实现 + 我们已有 80% 骨架**：本机方法论是 `KP-UNI-0221`（放得下的设备算 · 放不下的 CPU **原地**算 · 值得搬的 DMA+门控作业 · **取慢者为节拍** · 份额按层测速对齐）+ `KP-ENGSPEED-0003`（三轮段同跑，**单侧加速上限≈1/3**）；Strata 的实现是 `--pcie-frac`（`m=(nmiss·pcie_num)>>8`）+ `--pcie-mode direct`（`device_alias` 零拷贝，GPU **原地**读宿主 pinned 副本）；我们 02:02 已落 `pcie-frac`+`cpu_share` 自适应（A 级代码），**缺 `pcie_mode`/`device_alias`（poly 全量 0 命中）⇒ 没有"原地"那一半**。**"边搬边算"在本机已被三次独立证据判死（天花板 0，`KP-NINFER-0007`）**，Strata 只在 **prefill** 做成、decode 侧它自己承认窗口结构性为零（`session.hpp:116-130`）。**"专家图"在本机知识库与 Strata 0.1.40.2 都不存在（全量 0 命中）**，最接近的储备是"下一层专家预测"且已被定性为**延迟工具、只改 miss<0.1%**。**2× 判断：唯一可能是第 3 样**（单独 1.6–1.9×，叠加 CPU 内核 VNNI 后 ≈2.1×）；边搬边算 <1.3×、专家图 <1.5×。

---

## §1 第一层 · `E:\rlab` 知识库（入口三件已读 + 26 词全量）

### §1.1 入口三件
- `E:\rlab\90-状态\STATE.md`（2,437 B，2026-10-14）：当前世代替换 `PRJ-KVMEM@G3`；各实体主线 FRK-ENGSPEED / FRK-XOFFLOAD / PRJ-KVMEM / PRJ-NINFER / PRJ-STRATA 等（G1 frozen 居多）。
- `E:\rlab\10-知识\_生成\INDEX.md`（3,591 B，2026-10-16）：全库 **250**（通用）+ **3**（重点）+ **278**（项目）张；长文清单明确"**>5000 t 禁整读，先 grep 定位 offset/limit**"；两条不许踩的线：不 grep `20-生产/<实体>/G<n>/` 冻结副本；单次 ≤1500 token。
- `E:\rlab\30-运行\kb.py search "<词>"`（**必用检索入口**，本次全部用它）。

### §1.2 检索结果（全量；名片模式单次最多列 3 张，`3` 表示命中≥3，`0`/∅ 为真无命中）

| 词 | 命中 | KP-ID（一句话） |
|---|---|---|
| 专家图 | 3（均不含"专家图"概念，只匹配到"专家"字样） | `KP-NINFER-0090`（我方无 CPU 专家核⇒每 miss 过 PCIe，差 2.55× 主项）· `KP-NINFER-0089`（专家取数三档 12× 差）· `KP-ENGSPEED-0018`（CPU 专家核记录面 parity） |
| expert graph | 3（同上，无图概念） | `KP-ENGSPEED-0004`（每层一次同步→整窗一张图）· `KP-UNI-0015`（nsys 不追踪 graph replay）· `KP-KVMEM-0007` |
| **边搬边算** | **0**（已搜"边搬/搬边/边算"，∅ 无命中） | — |
| 搬算重叠 | 3 | `KP-XOFFLOAD-0013`（**预取是延迟工具不是命中率工具**：完美预测只改 miss<0.1%，判据=staged hit 转化+有用字节比<0.3 关）· `KP-NINFER-0007`（**重叠族全部关闭，天花板=0**，三证据）· `KP-STRATA-0044`（机制集合跨引擎不同：图 1.51× vs 重叠族 ninfer 封顶 2.0×） |
| overlap | 1 | `KP-STRATA-0024`（R4 overlap 0 of 0 是**零样本死计数**，不许当证据） |
| 流水 | ≥1 | `KP-STRATA-0030`（防撞车闸门两把尺子） |
| pipeline | 3 | `KP-NINFER-0010`（CUDA 头不许进宿主 TU）· `KP-XOFFLOAD-0007`（我方无 CPU 专家核 q*=1 源码证据）· `KP-XOFFLOAD-0002`（可改范围 19 条） |
| 预取 | 3 | `KP-NINFER-0007` · `KP-XOFFLOAD-0013` · `KP-STRATA-0040`（i7-13790F 无 AVX-512；**AVX2 内核零 prefetch**；自移植上限 ≤0.5%） |
| prefetch | 3 | `KP-STRATA-0040` · `KP-XOFFLOAD-0013` · `KP-XOFFLOAD-0010` |
| CPU GPU 分流 | 3 | `KP-NINFER-0090`（Strata 只 2.2% routed 过 PCIe，其余 CPU）· `KP-NINFER-0006` · `KP-COMFY-0002` |
| 按比例 | 3 | `KP-NINFER-0039`（fixture 比例缩放）· `KP-ENGSPEED-0008`（投机/跳过正交乘数） |
| 原地计算 | 3 | `KP-STRATA-0014`（重排常驻排名表=零代价第一步）· `KP-UNI-0138` · `KP-MOEDEP-0020` |
| in-place | 3 | **`KP-UNI-0221`（CPU 与 GPU 同算：取慢者为节拍 · 份额自动对齐）** |
| 专家缓存 | 3 | `KP-MOEDEP-0048`（6G 档否定结果集）· `KP-UNI-0226`（KV 与专家缓存抢显存）· `KP-MOEDEP-0041`（全常驻档 CPU 池 0 dispatches、decode 134–136） |
| expert cache | 3 | `KP-STRATA-0003`（expert-cache auto 不可复现）· `KP-STRATA-0039`（11000 附近 −75% 封印区）· `KP-UNI-0249`（NVIDIA 官方答案=改模型不是改调度） |
| decode 架构 | 3 | `KP-UNI-0179` · `KP-XOFFLOAD-0012`（NVFP4 是制品编码不是算力）· `KP-UNI-0173` |
| Strata decode | 3 | `KP-STRATA-0032`（`STRATA_DECODE_TIMING=1` 只 serve 打印、自带 ~9% 开销）· `KP-STRATA-0025`（**判死清单**：pcie-frac 上调/双缓冲/fused 拷贝/device-decide…）· `KP-NINFER-0003`（同卡 2.55× 拆解） |
| 专家共现 | 3（同"专家图"，无共现概念） | `KP-NINFER-0090` · `KP-NINFER-0089` · `KP-ENGSPEED-0018` |
| **co-occurrence** | **0**（∅ 无命中） | — |
| staged hit | 3 | `KP-XOFFLOAD-0013` · `KP-UNI-0245`（命中率口径律）· `KP-STRATA-0024` |
| 有用字节比 | 3 | `KP-XOFFLOAD-0013` · … |
| 无人命名 | 3 | `KP-UNI-0172`（grep 漏命名变体⇒误判"没有"）· `KP-XOFFLOAD-0014` · `KP-RLAB-0009`；**注意：库里没有"d 无人命名"的现行卡**（旧结论已收窄，见 §2.2） |
| PLE blocked | 3 | `KP-XOFFLOAD-0007` · `KP-XOFFLOAD-0015` · `KP-UNI-0204` |
| ADAPT | 3 | `KP-MOEDEP-0029` · **`KP-STRATA-0017`（自适应缓存调激进三调全慢=本机判死）** · `KP-MOEDEP-0046` |
| 取慢者 | 2 | **`KP-ENGSPEED-0003`（三轮段同跑，只加速一段上限约 1/3）** · `KP-UNI-0221` |
| 份额自动 | 3 | **`KP-UNI-0221`** · `KP-UNI-0223`（施工顺序与 MVP）· `KP-UNI-0218`（反号与适用域） |

### §1.3 对本题最有用的 6 张储备（正文出处已定位）
1. **`KP-UNI-0221`（= 需求原述的"CPU GPU 按比例原地计算"）** → `E:\视频\_flashnext\从0搭建专武引擎方法-20261014.md:171-253`：三原则（§8.1）+ **取慢者为节拍**（§8.2，原文 *"the next layer waits for the slower of the two"*、*"Speeding up only one side cannot help by more than about a third"*）+ **"原地算"是关键字**（§8.3：CPU 算完就地归约成每 token 一个向量回传，"三篇里唯一真正减字节的机制"）+ 份额自动对齐（§9.1：按层测两侧耗时，给"让两侧同时结束"的份额；§9.2 实测适用域 prompt chunk<1024，512-token chunk prompt −28~−34%）+ 一轮拆账样例（§10：GPU 14.6 / CPU 13.4 / draft 2.0 / 其它 4.3 ms，3.23 token/轮，命中 0.72）。
2. **`KP-ENGSPEED-0003` / `0004`** → `E:\rlab\20-生产\FRK-ENGSPEED\详细\因果链-引擎为什么慢-改哪里-怎么判-20261014.md:134-201`：链③三段同跑（单侧≤1/3；"CPU worker=物理 P 核−2"；6→62.25 vs 24→58.89 tok/s **越多越差**）；链④宿主同步点（每层一次→每 token 48 次往返；整窗一张图/发射落地解耦；**Windows 上自旋等钉住标志会卡在驱动命令缓冲 ⇒ 默认关**；架构级原文 *"decode is limited by the CPU expert pool and its per-layer host↔GPU handoff, not by GPU compute"*）。
3. **`KP-NINFER-0007`（重叠族判死）** → `E:\视频\_flashnext\docs\共识-20260927.md:215-220`：三条独立证据（① 我们实测搬运已在线速：1.089 GB 预测 48.4 ms/实测 48.7，误差 0.6%；8.2 miss/层纯字节 1.01/实测 1.01 分毫不差 ⇒ 总线无空隙；② llama.cpp PR #28414 1-deep lookahead+独立 stream **decode unaffected by construction**、小 prompt +8% 更差，根因"batch-1 decode 时路由 ids 在发起拷贝前才可知 ⇒ 预取只能是预测不是前瞻（朴素下一层前瞻仅 6.7% 覆盖）"；③ musubi-tuner PR #700：pinned+双向双 stream "One stream seems to be faster than two streams"）；**削减清单**在 `:205-211`。
4. **`KP-STRATA-0026` / 挪池三杠杆** → `E:\视频\_flashnext\docs\交接-Strata-挪池起点-优化任务书.md:85-179`：L1 提交语义（hit 路径慢 0.91 ms、"从不重叠"、`hit_poke` 探针）· L2 **窗口结构性为零**（`session.hpp:116-130` 原文 *"A per-layer CPU pool cannot be hidden behind a strictly serial residual chain."*，`posts[l]` 要等 pool 返回）· L3 无前瞻；杠杆 A 免预测降屏障（层×6→层×1，上限=GDN 串行）· B lookahead 预测（**错=白读压饱和 DDR4 可能净亏**）· C 动态淘汰；§9 判据（池 ms 不变而回合变短=真挪了起点；pool 也降=口径串味）。
5. **`KP-STRATA-0025` 判死清单** → 同上 `:190-203`：**pcie-frac 上调 0.80 = −13.3/−13.5%（ABBA×2，最优 0.35）**、整层双缓冲 ≈126 GiB 装不下、fused 拷贝/device-decide/no-hit-poke/减量族/节点融合/tensor-core decode/更大窗口/早退/浮点反推 已判死。
6. **`KP-XOFFLOAD-0013`（专家预测的定性）** → `E:\infer-build\refs\infernix-20261013\docs\maintainer\qwen3_8-flash-next-design.md:1094-1118`（§8.7 Lookahead prefetch）逐字：*"Prefetch is a **latency** tool. Replay shows that perfectly predicted next-layer experts change the miss **count** by < 0.1%. Its value is turning critical-path misses into staged hits."* + 每层预测召回/有用字节比公布、有用比 <0.3 自动关。

---

## §2 第二层 · 既往实验目录（3 套，全量 grep）

### §2.1 命中计数（全量，不计截断）
| 目录 | md 件数/字节 | 专家图 | 共现 | 边搬边算 | 原地 | 重叠 | ADAPT | lookahead | 前瞻 | 预取 | 21.7 | 无人命名 | PLE blocked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `E:\视频\_flashnext\experiments\20261013-infernix-dissect\` | 10 / 1,791,039 | 0 | 0 | 0 | 3 | 26 | 192 | 29 | 7 | 45 | 9 | 0 | 11 |
| `E:\视频\_flashnext\experiments\20261014-speed-campaign\` | 127 / 1,917,813 | 0 | 0 | 0 | 4 | 55 | 24 | 0 | 0 | 36 | 21 | **15** | 13 |
| `E:\视频\_flashnext\docs\` | 85 / 2,399,541 | 0 | 4 | 0 | 31 | 150 | 76 | 26 | 14 | 141 | 18 | 0 | 6 |

### §2.2 三样机制在既往实验里的结论
- **专家图**：**三目录全量 0 命中**（"共现"4 行也不在讲专家图）⇒ **本机从未做过"专家图"**；最接近的是 **lookahead/前瞻预取**：Infernix 设计 §8.7（见 §1.3-6）已定性"延迟工具、改 miss<0.1%、有用字节比<0.3 关"；`KP-STRATA-0026` 杠杆 B 明确"预测错=白读字节压饱和 DDR4，可能净亏"。
- **边搬边算**：**0 命中该词**，但等价概念"重叠/overlap"在 3 套里 231 行（26+55+150）；**结论已在 `共识-20260927.md:215-220` 判死（天花板=0）**；`交接-Strata-挪池起点-优化任务书.md:85-109` 给出 Strata 侧的三层诊断（L1 提交语义 / L2 窗口结构性为零 / L3 无前瞻）；`共识 §7.2`：CUDA 图的价值要改口径为"消除 launch"（Ada 实测 1.028×，值 ~3%）。
- **CPU-GPU 按比例原地计算**：储备最全（`KP-UNI-0221` 全文见 §1.3-1；`KP-ENGSPEED-0003`）；Infernix 实测适用域与收益（512-token chunk prompt −28~−34%、1000-token −20~−26%，8/8 配对，质量散度 0.004/最大 0.025）；`对账-Strata-vs-NInferStageC-同模型同卡-20261005.md:63-108`：Strata 只 **2.2% routed 过 PCIe（26 MB/token）**、其余 miss **交 CPU 原地算**；`--pcie-frac 0.35` 现役/0.55 默认；decode 2.55× 拆解 = 位宽 1.35× + R1 1.90×，"两边都不是 PCIe 受限"。
- **"无人命名的 d ≈21.7 ms/token"**：出处 = `E:\视频\_flashnext\experiments\20261014-speed-campaign\S0-读数-20261014.md:20`：`残余 = 30,810 − 22,130.5 = 8,679.5 ms = 28.2% = 21.7 ms/token`；`规划蓝图-赶超Strata-v6-20261014.md:16,91,132`：**`d` 的天花板算术：`d=21.7 ⇒ 46.1 tok/s`、13.4 ⇒ 74.6、28.5 ⇒ 35.1**，并标注 *"**'无人命名'已收窄** ⇒ 它侧 `generate.cpp:3947-4154` 与旧线（罗盘：`wait for rings` 75% · `PLE blocked` 20.1% · `host 0.205 ms/回合`）先核它们与 `d` 是不是同一物"*。**⇒ 与今天的 24.3 ms 账（`wait for rings 12.378`=GPU pre、`pool 7.269`=CPU 池）互证：当年"无人命名"的那块，就是 Strata 侧的分段账字段。**
- **PLE blocked / ADAPT**：`PLE blocked`（infernix-dissect 11 行 / speed-campaign 13 行 / docs 6 行）指旧线把 GPU 卡在 PLE 页表读上（`KP-XOFFLOAD-0007` 一带）；`ADAPT`（infernix-dissect 192 行）主要是 Infernix 的 adaptive tuning 章（拆解件 :1117 指 §14 L1989–2187）；**本机结论**：`KP-STRATA-0017` 自适应缓存调激进三调全慢=本机判死；`KP-MOEDEP-0048` 6G 否定集（砍自适应=搬运 +82%）。**⇒ ADAPT 别再动。**

---

## §3 第三层 · 与 Strata 的差别（对比表 + 全量计数）

### §3.1 三方全量计数（`src` 全树 grep，计匹配行数，无截断）
| 关键词 | Strata 0.1.40.2（116 文件/2.18 MB） | upstream 0.1.44.1（215/6.17 MB） | 我们 poly（39/0.86 MB） |
|---|---|---|---|
| expert graph / co_occur / co-occur | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** |
| lookahead | 2（仅 prefill ring 暂存） | **37** | **0** |
| predict | 6（注释/采样） | 46 | 2（`nextn_predict_layers` 元数据） |
| foresight | **0** | **32** | **0** |
| prior_ / profile | 0 / 57 | 7 / 234 | 9 / 29 |
| pcie_frac / pcie_num | 16 / 10 | 54 / 27 | 11 / 2 |
| **pcie_mode / device_alias** | **7 / 3** | **9 / 6** | **0 / 0** |
| cpu_share / on_cpu / spare_ | 0 / 0 / 0 | 45 / 12 / 2 | 7 / 5 / 7 |
| doorbell | 62 | 106 | **0** |
| copy stream / stage_live / hit_ready+late | 6 / 7 / 5 | 16 / 10 / 5 | **0 / 0 / 0** |
| overlap | 76 | 145 | 22（含 ring2/cp_ 注释） |
| prefetch | 19 | 181 | 14（prefill plan_ahead） |

### §3.2 对比表（三机制）
| 机制 | 本机储备怎么说（KP-ID/路径 + 结论） | Strata 怎么实现（`file:line`） | 我们现在有没有（poly 全量计数） | 差距 |
|---|---|---|---|---|
| **专家图**（专家依赖/共现图 ⇒ 调度或预取） | **无此储备**：kb 全量 0 命中（"专家图/共现/共激活/co-occurrence"）；最接近 = 下一层预测 `KP-XOFFLOAD-0013`（Infernix §8.7：延迟工具，完美预测改 miss<0.1%，有用字节比<0.3 自动关）+ `KP-STRATA-0026` 杠杆 B（预测错可能净亏） | **0.1.40.2 没有**（`lookahead` 仅 `src\prefill\prefill.cpp:1307` 的 ring 暂存 lookahead；`predict` 命中都是注释；无 expert graph 代码）⇒ 只有 **profile 全局排序**（`generate.cpp` 的 `--expert-profile` 一次性铺满、不驱逐，见 `储备-Strata加速原理…:30`）；**0.1.44.1 才有** lookahead(37)/foresight(32)/predict(46) 行（老师没跑、默认未开） | **无**：`expert graph/co_occur` 0、`lookahead` 0、`foresight` 0；只有 `--expert-profile` 全局计数（`engine.cpp:182-198,1000-1104,1606`；`engine.hpp:771` `use_/prior_`） | **同为零**（都没专家图）；我们与 Strata 0.1.40.2 一样"只有全局 profile"；**专家图是全新方向**，且本地储备已把"预测"定性为低命中收益 |
| **边搬边算**（DMA 与计算重叠） | **判死**：`KP-NINFER-0007` + `共识-20260927.md:215-220` 三证据 ⇒ **天花板 0**（总线已在线速、PR28414 decode by construction 无效、musubi 单流更快）；`KP-STRATA-0044`：机制集合跨引擎不同；`KP-STRATA-0024`：R4 overlap 是死计数；**唯一值得抄的形态在 prefill**（`储备…:26-35` 可预排） | **prefill 完整**：copy stream `src\prefill\prefill.cpp:472`、ring `kRing=16`/`ring_slots`（`:125,83`）、stage-ahead `:1196,1225-1237`、`stage_live` 事件 `:872,882`、固定顺序预排 `:66-82,798-826`；**decode 侧**：`expert_source.cpp:367` `P.fetch(...)`（copy engine beside CPU's work）、`hit_ready/hit_late` 计数 `:547-548`（**死计数**，KB-STRATA-0024）、`expert_source.cpp:3720-3721` plan 先发布；但窗口结构性为零（`session.hpp:116-130`，`posts[l]` 等 pool 返回） | **部分**：prefill 的 `cp_` 流（`engine.cpp:1882-1903`，默认关）+ `ring2_`（`:1903-1917,2519-2521,2645`）+ `stage_ahead`（`engine.hpp:60`）；decode 侧 **`doorbell` 0、`hit_ready/late` 0、`stage_live` 0**；pcie 分流后有"DMA 与 CPU 腿并发"雏形（`engine.cpp:2394` 注释 *"and the CPU's overlapped the copies"*） | **prefill 已有等价物；decode 缺"发射/落地解耦"与跨层重叠**——但储备说这条天花板 0，不该再开；要抄只抄 prefill（我们已有） |
| **CPU-GPU 按比例原地计算** | **储备最全**：`KP-UNI-0221`（`E:\视频\_flashnext\从0搭建专武引擎方法-20261014.md:171-253`）三原则 + 取慢者为节拍（单侧≤1/3）+ **"原地算"=就地归约、只回传每 token 一向量** + 份额按层测速对齐（Infernix 实测 512-token chunk prompt −28~−34%）；`KP-ENGSPEED-0003`（`因果链…:134-201`）三段同跑/单侧上限 1/3/CPU worker=物理 P 核−2 | **有实现（按专家粒度、非按行）**：`src\core\expert_source.cpp:285` 注释 *"resident ones and **the last pcie_num/256 of the missed ones go to the GPU**"*；`:301-302` `m = pcie_ok ? (nmiss*pcie_num)>>8 : 0`；`:348` `P.ptr2[q] = P.pcie_mode != 0 ? device_alias(...)`（**mode≠0 = 零拷贝原地读宿主 pinned 副本**）；`:721` `ArenaExpertSource::device_alias`；`verify.cpp:588` `pcie_mode==2` 用 copy kernel 暂存；选项 `generate.cpp:244-245`（`pcie_frac` 默认 0.2 direct(Q2_0)/0.55 DMA(native)）、`:945-950`（`--pcie-frac/--pcie-mode auto|dma|kernel|direct`）、`:2568` `set_pcie_mode`、`:2630` `pcie_num=frac*256`、`:1192-1202` **PCIe probe 按实测带宽自动调 frac**；CPU 份额在 prefill 侧还有 `--cpu-share`（0.1.44.1 45 行） | **骨架已有 80%**：`pcie_frac` 已落（`engine.cpp:1500,1509,1786,1854,2224-2243`，`m_uses=(n_host*pcie_num)>>8` 与老师同构）+ `cpu_share` 5 档自适应（`engine.cpp:2586-2591,2712-2722,3108`；`engine.hpp:740` `share_[5]={0.25…}`）+ `on_cpu_/spare_` 选择（`engine.hpp:737-738`）；**但 `pcie_mode`/`device_alias` 全量 0 命中 ⇒ 我们没有"原地直接读宿主"模式**（PCIe 份额必须 DMA 进 staging，多一次写显存；CPU 侧从 GGUF 映射读，不是专用 pinned RAM arena） | **差距 = "原地"那一半 + 份额求解**：我们缺 `direct/zero-copy` 模式；我们的 `share_[5]` 是"按调用大小 5 档运行均值"（读码推断），不是老师/Infernix 的"按层测两侧耗时、令两侧同时结束"的显式求解 |

---

## §4 三条结论

1. **最该做、且本机储备最全的是"CPU-GPU 按比例原地计算"（第 3 样）**：方法论（`KP-UNI-0221`/`KP-ENGSPEED-0003`）+ Strata 参考实现（`pcie_frac`+`pcie_mode direct`/`device_alias`）+ 我们已落地骨架（02:02 的 pcie-frac + cpu_share）三者齐；缺的两块是 **① `direct` 零拷贝原地模式（Strata 有，我们 0 命中）② "让两侧同时结束"的份额求解（我们是运行均值）**。
2. **"边搬边算"不要重开**：本机已三次独立判死（`共识-20260927.md:215-220`），Strata 的 decode 侧自己也是窗口结构性为零（`session.hpp:116-130`）+ Windows 上靠自旋等钉住标志的形制默认关（`因果链…:178`）；它做得好的只是 **prefill 的固定顺序预排 + copy stream + ring**，而那套我们已有等价物（`cp_`/`ring2_`/`stage_ahead`）。
3. **"专家图"在本机与 Strata 0.1.40.2 都不存在**（全量 0 命中），0.1.44.1 才出现 lookahead/foresight；本机储备已把"预测下一层专家"定性为**延迟工具、只改 miss<0.1%**（`KP-XOFFLOAD-0013`）。**我们当前 decode 的瓶颈是 CPU 腿算力（100 µs/miss、110 GFLOPS/线程），不是 miss 数** ⇒ 专家图的直接收益可疑；它唯一的价值路径是"把 CPU 腿上的 miss 变成已 staged 的 GPU hit"，而这需要先有第 3 样的 `direct` 模式与份额求解。

---

## §5 ② 三样里哪一样能在 decode 上出 2× 以上？——**只有"CPU-GPU 按比例原地计算"**

依据（算术，D 级，输入全部 A/C）：
- 现 decode **75 ms/token** 中 **CPU 腿 71.9 ms**（= 496.7 miss × 144.7 µs，scout-b 读数）；专家字节 **1.54 GB/token**（496.7 × 3.0966 MB）；PCIe 实测 **23.2 GB/s**（本机 q38 日志）⇒ 链路时间 = **67 ms × s**（s=分给 GPU 的份额）。
- 令两侧同时结束：`71.9(1−s) = 67·s` ⇒ **s = 0.518** ⇒ 节拍 ≈ **34.7 ms ⇒ 2.16×**（上限；假设 GPU 侧能吸收 s 的专家、GPU 自身计算不超 34.7 ms、元数据/同步免费）。
- 现实取 s = 0.30–0.40（留预测/同步余量）⇒ 节拍 ≈ **43–50 ms ⇒ 1.5–1.75×**；**叠加 CPU 内核 VNNI 提速（CPU 腿 71.9→~50 ms，见 `情报-N`）后 s=0.30 ⇒ ~35 ms ⇒ ≈2.1×**。
- 对照：**边搬边算 <1.3×**（天花板 0 已判死，能拿的只是 pcie 分流那段 DMA 与 CPU 腿并发的既有部分 + 减少 launch/handoff）；**专家图 <1.5×**（预测只改 miss<0.1%，只能把 miss 从 CPU 腿挪成 staged hit，且有用字节比<0.3 自动关）。
- **共同约束**：`KP-ENGSPEED-0003` —— 三段同跑、**单侧加速上限≈1/3** ⇒ **任何"只加速一侧"的单旋钮都到不了 2×；2× 只能来自"改变两侧的活分配"**（把 CPU 腿的活按比例挪给 GPU），而不是把某一侧加速。这正好把第 3 样与另两样区分开。

---

## §6 未验 / 边界
- **U1**：kb.py 名片模式单次最多列 3 张（"展开下一层"可再加），表里 `3` = "命中≥3"，`0`/∅ = 真无命中；如需逐卡全文要 `kb.py show <id>`。
- **U2**：0.1.44.1 的 lookahead/foresight **是否默认关**未逐行核（老师 0.1.40.2 没跑；`--foresight`/`STRATA_FS_*` 旗标未验）。
- **U3**：Infernix 的份额实测（−28~−34%）是 **Infernix（另一引擎、另一模型几何）** 的读数 ⇒ C/D，不是本机/我们模型的读数。
- **U4**：Strata `--pcie-mode direct` 在本机 4080S/WDDM 上的实际可用性未验（老师日志用的是 0.55 DMA native）。
- **U5**：我们 `pcie-frac` 的最优 share 未测（代码 02:02 已落，未跑）。
- **U6**：我们 `share_[5]` 是"按调用大小 5 档运行均值"（读码推断 `engine.cpp:2720,3108`），与"两侧同时结束"的求解不是同一物——**未验**。
- **U7**：`专家图` 若指"用于**剪枝/路由**的专家共现图"（外部 Star Elastic/Bolt 方向），本机只有 `KP-UNI-0249`（官方解=改模型不是改调度）一条间接记录；外部文献未本次检索（零网检索）。

---

## §7 复算命令（只读）

```powershell
# ① 知识库入口三件
Get-Content 'E:\rlab\90-状态\STATE.md' -TotalCount 40
Get-Content 'E:\rlab\10-知识\_生成\INDEX.md'
python 'E:\rlab\30-运行\kb.py' search '专家图'
python 'E:\rlab\30-运行\kb.py' search '边搬边算'      # ∅ 无命中
python 'E:\rlab\30-运行\kb.py' search 'in-place'      # KP-UNI-0221
# ② 既往实验全量计数
Get-ChildItem 'E:\视频\_flashnext\experiments\20261013-infernix-dissect','E:\视频\_flashnext\experiments\20261014-speed-campaign','E:\视频\_flashnext\docs' -Recurse -Include *.md |
  Select-String -Pattern '专家图|边搬边算|重叠|原地|ADAPT|lookahead|21\.7'
# ③ 关键储备正文
#   E:\视频\_flashnext\从0搭建专武引擎方法-20261014.md:171-253（CPU/GPU 同算三原则+份额对齐）
#   E:\rlab\20-生产\FRK-ENGSPEED\详细\因果链-引擎为什么慢-改哪里-怎么判-20261014.md:134-201
#   E:\视频\_flashnext\docs\共识-20260927.md:205-220（重叠族判死）
#   E:\视频\_flashnext\docs\交接-Strata-挪池起点-优化任务书.md:85-179（L1/L2/L3+三杠杆）
#   E:\infer-build\refs\infernix-20261013\docs\maintainer\qwen3_8-flash-next-design.md:1094-1118（§8.7）
# ④ 三方实现
Select-String 'E:\strata\src\**\*.cpp' -Pattern 'pcie_mode|device_alias|pcie_num'
Select-String 'E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly\*.cpp' -Pattern 'pcie_frac|cpu_share|share_\['
```

**判据（能变红）**：① kb 全量检索里出现"专家图/共现"的机制卡 ⇒ §0 判错；② `E:\strata` 全树能命中 `expert graph|co_occur` ⇒ "老师无专家图"错；③ poly 全树能命中 `device_alias|pcie_mode` ⇒ "我们没有原地模式"错；④ 若 `pcie_mode direct` 实测在本机不可用（CPU 侧直接读 pinned 跑 PCIe DMA）⇒ 第 3 样的实现路径要改回 DMA。

---

## §8 落盘
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-O-本机储备与Strata-decode架构差-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-储备与架构差.md`
