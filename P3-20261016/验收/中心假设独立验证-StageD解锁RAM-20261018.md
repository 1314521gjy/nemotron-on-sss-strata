# 中心假设独立验证：Stage D 能否解锁 RAM（零 GPU，只读码）— 2026-10-18

```
状态: 达成（穷举 down 读者 + 判决三句 + map_pages/pin 改点 + RAM 预算重算）
席位: 验收（reviewer）· 任务: lead 派（验证"GPU down 内核读 T ⇒ GGUF row-major down 再不会被读 ⇒ RAM 95.1→~68 GB"）
方法: 全树 grep E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\src\poly（40 文件）
      + 逐函数读码（不截断计数）；未编译、未改码、零 GPU、未 spawn 子代理
★ 行号 = 当前源码（R49/borrow/relu2 之后；builder3 在 R53 §0 给的行号已随源码增长而漂移，见 §2）
```

## §0 判决三句

1. **论断：部分成立 / 条件成立，不能按字面成立。** "GPU 侧的 down 内核也读 T" 只覆盖 `moe_card()` 的卡上 down（`engine.cpp:3283-3304`）；**至少还有 10 条路径直接读 GGUF row-major down**：`fill_expert_cache`（:1542）、`grow_cache`（:1899）、swap follower（:1943）、**prompt-borrow refill（:1693，交付 `STRATA_PROMPT_BORROW=1` 正在用）**、`stage()`（:3341-3343 / :3359）、`moe_rows` 的 MMQ down（:3845/:3861，经 staging）、`moe_rows` 的 CPU-share（:3551）、`relu2_down_open` 的 on-load 检查（:583，4 专家 = 6.2 MB）、`pin_expert_arena` 的 down 注册（:1301）、以及 `map_pages` 的 down span（:2312，逐页 touch 的 RAM 大头）。**只改 GPU down 内核，GGUF down 仍会被读，`map_pages` 不能跳。**
2. **若按完整 Stage D 包（计划里的 D2 prefill-MMQ-T + D4 staging 源 + D5 fill/swap + D6a loader 不映射 down）并补 3 处计划外（borrow refill 源、moe_rows CPU-share 的 downT、MTP 块 down / 或交付保持 `--no-draft`）：则成立** —— `map_pages` 删 `engine.cpp:2312` 的 down span、`pin_expert_arena` 只 pin up（:1285-1301）、`relu2_pin_switch` 不再 unregister down（:1342）即可；RAM 从 **95.13 GB** 降到 **≈68.0 GB**（draft 关；MTP 保留 down 则 ≈68.8），pin 预算 **29.53+29.53+8 = 67.06 GiB ≤ 96**（现状 59.06+29.53+8 = **96.59 > 96**，与 R57 A/B 死路一致）。
3. **若不补这些（只做 GPU down 内核）：Stage D 单独不能解决 RAM**；必须同时做 D2/D4/D5/D6a + 上述 3 处边角，且 prefill 不倒退（G4 ≥1700 tok/s）与 GPU-T 位对位（D8②）仍是硬门。

## §1 方法与全量计数（未用 `Select-Object -First N` 截断计数）

全树 grep（`src\poly`，40 个 .cpp/.hpp/.cu；计数 = 全部命中，不截断）：

| 模式 | 全量命中 | 模式 | 全量命中 |
|---|---|---|---|
| `exp_down` | **28**（engine.cpp 26 + engine.hpp 2） | `exp_up` | **31** |
| `exp_gate` | **19** | `d_bytes` | **73**（7 文件） |
| `gu_bytes` | **43** | `map_pages` | **2** |
| `pin_expert_arena` | **6** | `fill_expert_cache` | **15** |
| `moe_card` | **9** | `moe_rows` | **32** |
| `shrink_cache` | **9** | `cover(` | **21** |
| `follow(` | **16** | `trade` | **24** |
| `relu2_down` | **23** | `sidecar` | **52** |
| `pcie_` | **104** | `draft` | **208** |
| `MTP` | **9** | `cudaMemcpy` | **135** |
| `memcpy(` | **116** | `stage(` | **11** |
| `profile` | **30** | `blk.88` / `blk 88` 字面 | **0**（MTP 用 `blk(n_layer_)` 表达，n_layer_=88） |

**关键**：全部 28 个 `exp_down` 命中逐行分类（见 §3 表），`d_bytes` 73 命中已按"读数据/只算尺寸/日志/测试"分类；**GGUF down 的运行时读点全部经由 `Layer::exp_down` 指针**（grep `tensor_data`/`gguf` 无其它 down 读点；`expert_parity.cpp:136/156` 是测试工具，不进引擎）。`cover()/shrink_cache()`（:801/:839）与 `save_profile()`（:2683）**不读 down**（读码确认）。

## §2 builder3 四处"已知读点"复核（行号已漂移，但读点仍存在）

| # | builder3 的 R53 行号 | 现役源码位置 | 读 down？ |
|---|---|---|---|
| 1 | `fill_expert_cache` raw copy `1146-1159` | **`engine.cpp:1429`** 函数；down 拷贝 **`:1542`**（`cudaMemcpy(s + L.slot_down(), L.exp_down + e*d_bytes, …)`） | **是** |
| 2 | `stage()` raw DMA `2664-2669` | **`engine.cpp:3310`** 函数；pinned DMA down **`:3341-3343`**；非 pin host memcpy down **`:3359`** | **是** |
| 3 | `moe_card` down = `table + 2*gu_bytes` `2626` | **`engine.cpp:3283`** 函数；卡上 down kernel **`:3300-3302`**（`native_mmvq_ids(down, table, 2*L.gu_bytes, …)`）——读的是 slot（slot 内容由 fill 从 GGUF 拷入） | **间接**（读 slot，不读 GGUF；slot 来源见 #1/#另一 fill :1899） |
| 4 | `moe_rows` mmq down 行距 `3171` | **`engine.cpp:3378`** 函数；MMQ down **`:3845`** 与子批 **`:3861`**（`mmq_->mul(sbuf + down_at, …, L.d_bytes, …)`）——读 staging（staging 由 `stage()` 从 GGUF 拷入） | **间接**（同上，源头是 `stage()`） |

结论：builder3 的**行号已过时**（R53 之后源码增长约 20%），四个读点本身仍然存在；但"4 处"是**不完整的**——漏了 §3 的 6 类读者。

## §3 穷举：所有会读 GGUF row-major down 的路径（逐条判 Stage D 后）

> 分类：**直接读** = 从 `L.exp_down` 取字节；**间接** = 读 VRAM slot/staging（其内容由直接读点从 GGUF 拷入）；**不读** = 只注册/映射/计数。

| # | 路径 | file:line（现役） | 读什么 | Stage D 后仍读 GGUF down？ | 备注 / 为什么不能不改 |
|---|---|---|---|---|---|
| 1 | `fill_expert_cache`（VRAM 填槽） | engine.cpp:1542（函数 1429） | down（+ gate/up） | **仅当 D5 改源为 sidecar 后才不读** | 直接读；D5 计划内 |
| 2 | `grow_cache`（cache 长大的第二填槽路径） | engine.cpp:1899（函数 1851） | down | 同上 | 与 #1 是两条不同 fill 路径，D5 要两处都改 |
| 3 | `follow_loop` swap follower | engine.cpp:1941-1944 | gate/up/down → pinned→slot | **仅 D5 改源后不读** | 直接读（std::memcpy） |
| 4 | **`refill_prompt_borrow`（借槽 refill）** | engine.cpp:1688-1694（down :1693；函数 1668） | down | **仅 D5 扩到 borrow 才不读** | **交付 R60 正在用 `STRATA_PROMPT_BORROW=1`**（`run_delivery_t16.cmd`）；计划 D5 只写"fill/swap"，**borrow refill 是计划外** |
| 5 | `stage()` pinned DMA（prefill 直搬） | engine.cpp:3341-3343（函数 3310） | down（+ gate/up） | **仅 D4 改源为 sidecar 后不读** | 直接读；prefill 每个 staged expert 都读 |
| 6 | `stage()` 非 pin 路径 host memcpy | engine.cpp:3359 | down | 同上 | 同上 |
| 7 | `moe_rows` MMQ down（prefill） | engine.cpp:3845 / 3861（函数 3378） | staging 里的 down | **D2（T 版 MMQ）后不读** | 间接；若只改 decode 内核，prefill 仍要 row-major staging ⇒ #5/#6 仍读 |
| 8 | `moe_rows` CPU-share 的 `host_` | engine.cpp:3551-3552 | `L.exp_down + e*d_bytes`（CpuExpert.down，downT 未传） | **不改则仍读** | `{gate,up,down,w,row}` 5 字段 → `downT=nullptr` ⇒ CPU 走 ggml row-major vec_dot；`spare=!ahead&&cpu_share!=0&&!all_cached()` 时活跃（delivery 大 batch `ahead=true` 不活跃，但小 prompt/`shared_ahead_` 会） |
| 9 | `moe()` 的 CPU 专家（decode） | engine.cpp:2931-2934 | `L.exp_down…` + `downT=sidecar` | **relu2 on 时 CPU 用 downT（不读 GGUF）；relu2 off 时读** | Stage C 已覆盖（cpu_experts.hpp:51-59 `CpuExpert.downT`） |
| 10 | `relu2_down_open` on-load 检查 | engine.cpp:577-583（4 专家：首/末 MoE 层，专家 0/511） | GGUF down（6.2 MB） | **仍读（除非删检查/改从 manifest）** | 计划外；量小但会让这几页驻留 |
| 11 | `pin_expert_arena` down 注册 | engine.cpp:1301（循环 1282-1307；budget 1285-1288；`used` 1305） | 不读数据，但**注册 down 区** | **不读数据，但要改（pin 只 up）** | 不改则 pin 预算仍按 up+down 59.06 GiB 算，RAM 不降 |
| 12 | `relu2_pin_switch` unregister down | engine.cpp:1339-1350（:1342） | 不读数据 | **要改（down 不再 pin 则不能 unregister）** | 否则报 `refused to unregister`；sidecar 的 pin/预触页要在此或 load 时做（1352-1360 的 touch 逻辑复用） |
| 13 | `map_pages` 的 down span | engine.cpp:2307-2313（**down :2312**）；thread 2334-2337；map_pages 185-195 | 每 4 KiB **touch 一字节**（不语义读，但**把 29.53 GiB 页 fault 进 RAM**） | **是（除非删这个 span）** | **这是 RAM 大头**；删 :2312 后 down 页不再被 touch（但 #1-#10 一读就又 fault，所以必须全改） |
| 14 | pcie 路径 | engine.cpp:2972-3006（pcie_src_down）、3045（DMA） | down（pinned/device alias） | **默认关（`--pcie-frac 0`）时不读；>0 则仍读** | 计划外边界：若开 pcie，需拒绝或用 sidecar；`relu2_pin_switch` 只拒绝 `--pcie-mode direct`（1325-1330） |
| 15 | ring2 check（验证用） | engine.cpp:3797-3815（:3812 memcmp） | GGUF down | 仅 `STRATA_STAGE_RING2_CHECK=1` 时读 | 调试开关，非交付 |
| 16 | borrow verify（验证用） | engine.cpp:1713-1732（:1723 memcmp） | GGUF down | 仅 `STRATA_PROMPT_BORROW_CHECK` 时读 | 调试开关 |
| 17 | mxfp4→q5_0 转换 | engine.cpp:1067-1088（exp_down :1080） | 全部 down | **MXFP4 模型才走；本模型 IQ4_NL ⇒ N/A** | 条件路径；若换 MXFP4 模型则重新变成读点 |
| 18 | MTP/NextN block（`blk.<n_layer_>`，即 blk.88） | engine.cpp:2204（`n_all=n_layer_+(block_layer_?1:0)`）、2209-2210（load_layer）、nemotronhmoe.cpp:146-149（kAttnMoe 有专家）、495-498/560-563（sidecar 只覆盖 `il<n_layer_` 的 40 个主 MoE 层） | down（MTP 自己的 routed experts） | **draft on 时仍读 GGUF down；交付 `--no-draft` ⇒ block_layer_=false（engine.cpp:962）⇒ 不创建/不读** | 计划外：若默认 draft 交付，需把 MTP 层也放进 sidecar（+512×1,548,288 B = **756 MiB/层**）或保留其 down（+0.74 GiB RAM） |
| 19 | `expert_parity.cpp`（测试工具） | expert_parity.cpp:136/156 | GGUF down | 不进运行时 | 测试程序 |
| 20 | `cover()/shrink_cache()/save_profile()` | engine.cpp:801/839/2683 | **不读 down** | — | 读码确认：只动 arena/计数/写 profile 计数 |
| 21 | `load_moe()`/`load_shape()`（几何/映射） | engine.cpp:995-1030（`L.exp_down = home(td).data(td)` :1025） | 只取指针/尺寸（映射不 touch） | 不读 | mapping 本身不耗 RAM（页未 fault） |

## §4 论断判定：哪条路径断的

论断"GPU down 内核读 T ⇒ GGUF row-major down **再也不会被读** ⇒ 可跳过 map_pages"：
- **只改 `moe_card`/GPU 内核（:3283-3304）时**：#1、#2、#3、#4、#5、#6、#7、#8、#10、#11、#12、#13、#18 都还在读（或 touch）GGUF down。**路径断 = #5/#6（prefill `stage()`）与 #1/#2/#4（fill/borrow refill）**，因为它们是**每次 prefill/fill 都直接读**，不经过 GPU 内核。
- **按 Stage D 计划 D1-D9 全做时**：#1/#2/#3（D5）、#4（D5 需扩）、#5/#6（D4）、#7（D2）、#9（Stage C 复用）、#11/#12（D4 改 pin 只 up）、#13（D6a 删 span）都能覆盖；**剩下计划外**：#8（moe_rows CPU-share 的 downT）、#10（on-load 检查 6.2 MB）、#14（pcie 开时）、#18（draft on 时 MTP 0.74 GiB）。补上这些 ⇒ 论断成立。
- **交付配置（`--no-draft`、pcie 0、borrow 1）**：必须处理 #4（borrow 交付在用）、#5/#6、#1/#2、#13、#10；#18 不触发（draft off）。

## §5 若成立：map_pages / pin 改点 + RAM 预算重算

### 5.1 跳过 down 的改点（给行号）

1. **`map_pages` 的 span 表**：`engine.cpp:2307-2313`，删/条件化 **`:2312`** `spans.emplace_back(L.exp_down, (size_t) n_expert_ * L.d_bytes);`（up/gate 的 span 保留；mapper 线程 `:2334-2337`、`map_pages()` `:185-195` 不用改）。
2. **`pin_expert_arena` 只 pin up**：`:1285-1288`（`dn_len` 参与 budget 判断）、`:1301`（`reg(L.exp_down, dn_len, "down")`）、`:1305`（`used += up_len + dn_len + gt_len`）→ 去掉 down 项（pin 预算从 59.06 GiB 降到 29.53 GiB）。
3. **`relu2_pin_switch` 不再 unregister down**：`:1339-1350` 的 down unregister 循环删除；sidecar 的 pin/预触页改到 load 或此处直接做（复用 `:1352-1360` 的 `touch()` 逻辑：先 touch 每 4 KiB 一字节把 29.53 GiB fault in，再 `cudaHostRegister`）。
4. **loader（D6a）**：`L.exp_down = home(td).data(td)`（`:1025`）指针可保留，但不得被任何读点使用；若做 D6b（strip 文件）则另说（盘 −31.7 GB，需一次 70 GB 重写）。
5. **必须同时改的源**：fill（D5：`:1542`、`:1899`、swap `:1943`、borrow refill `:1693`）、staging（D4：`:3341-3343`、`:3359`）、prefill MMQ（D2：`:3845/:3861`）、CPU-share（`:3551`，补 `downT`）、on-load 检查（`:583`，删/改 manifest）；pcie（`:2972-3045`）在 sidecar 模式下拒绝。

### 5.2 RAM 预算重算（逐项）

| 项 | 现状 | Stage D 后 | 依据 |
|---|---|---|---|
| GGUF up（pin，resident） | 29.53 GiB = **31.71 GB** | **31.71 GB** | 40×756 MB=30,240 MiB=29.53 GiB；`onechunk_pin.log:82` "pinned 40 layers (59.06 GiB)" 的一半 |
| GGUF down（touch/pin） | 29.53 GiB = **31.71 GB** | **0**（跳 map_pages + 不 pin） | 同上；现状由 `map_pages`（:2312/185-195）逐页 touch |
| sidecar T（pin，resident） | 29.53 GiB = **31.71 GB** | **31.71 GB** | `relu2_down_open`（:509）`expect = 40×512×1,548,288 B` = 31.71 GB；R57 日志 "sidecar PINNED (29.53 GiB)" |
| MTP/NextN down（仅 draft on） | 756 MiB = 0.79 GB（当前 draft off 不读） | **0.79 GB**（若保留）或 0（sidecar 扩 0.74 GiB） | `n_all=n_layer_+1`（:2204）；MTP 有专家（nemotronhmoe.cpp:146-149）；sidecar 只 40 主层（:487-498/:560-563） |
| 非专家权重/进程/OS 等 | ≈4.6 GB（估） | ≈4.6 GB | 由 lead 的 "~68 GB" 反推（63.42+4.6≈68.0）；**D 级估算**，未逐项实测 |
| **合计（触碰的数据）** | **95.13 GB** | **≈68.0 GB**（draft 关；draft on 保留 MTP ≈68.8） | 95.13 = 63.42（up+down GB）+31.71（sidecar）= lead 的 95.1 |
| **pin 预算** | up+down 59.06 + sidecar 29.53 + 8 余量 = **96.59 GiB > 96 ⇒ 死** | up 29.53 + sidecar 29.53 + 8 = **67.06 GiB ≤ 96 ⇒ 活** | pin 规则 `:1270-1277`（needs budget+8 GiB）；R57 文档 §59 "59.06+29.53+8>96 不可行" |
| 盘 | GGUF 70.1 GB（含 down 31.7） | D6a：70.1 GB（盘不变）；D6b：38.4 GB | Stage D 计划 §D6（doc `计划-StageD-GPU侧T布局-20261018.md`） |

- 注意：`32,760−peak` 的显存约束不在此表（Stage D 不改 VRAM 需求；sidecar 在 RAM/pin）。
- 现状 95.13 GB 与 R57-E1 的失败一致：sidecar 不 prefault（`R57c-fix` 注释 :1352-1356），必须 touch 29.53 GiB；而 down 又占 31.71 GB ⇒ 96 GiB 装不下两份 31.7。

## §6 若不成立（只做 GPU down 内核）：Stage D 单独不能解决 RAM

- 不能跳过 `map_pages` 的 down span：fill（:1542/:1899）、borrow refill（:1693）、swap（:1943）、stage（:3341/3359）、moe_rows（:3845/3861 的 staging）、on-load 检查（:583）仍会读 down；一读就 fault，页还是驻留 ⇒ RAM 不降。
- **Stage D 要单独解决 RAM，必须把 D2（prefill MMQ T）+ D4（staging 源 + pin 只 up）+ D5（fill/swap/**borrow refill**）+ D6a（loader/map_pages/pin 跳 down）作为同一包做**，并补计划外 3 处：`moe_rows` CPU-share `downT`（:3551）、on-load 检查（:583）、MTP（draft on 时 0.74 GiB，或交付保持 `--no-draft`）；pcie 模式下拒绝 sidecar。
- 另外两个硬门不变：**prefill 不倒退**（计划 G4 ≥1700 tok/s）与 **GPU-T 位对位**（计划 D8②，CUDA parity；不过则整个 Stage D 停）。

## §7 读码确认 vs 推断（证据等级）

- **读码确认（A）**：全部 file:line 与读者分类（§3）；28 个 `exp_down` 全量命中；sidecar 只覆盖 40 个主 MoE 层（:487-498/:560-563）；MTP 不覆盖（:2204, nemotronhmoe.cpp:146-149）；map_pages 逐页 touch（:185-195）；R49/borrow 行号漂移。
- **算术/日志（A/B）**：95.13 GB = 63.42+31.71；59.06 GiB pin = up+down（`onechunk_pin.log:82`）；sidecar 31.71 GB（:509）；R57 A/B 死路（R57 文档 / R57-E1 日志）。
- **推断（B/D）**：非专家/进程 ≈4.6 GB（D 级）；"≈68 GB" 依赖该估算；MTP 在默认 draft 下会读 down（读码 + `--no-draft` 交付推断，未实测）；GPU-T gather 效率对 +35~40% 的折损（计划 §5 风险 5，B/C 级）。

## §9 RAM 账对账（lead 68.1 vs scout-b 91）· 2026-10-18 追加

> lead 追加核对项：判 lead（68.1GB 闭合）与 `情报-AD` §2.3 的 scout-b 账（91/96，余 3–5GB）谁对；重点是 **scout-b 的 42.5GB 是不是"额外副本"**，以及 **D6a 后 map_pages 还剩多少预热、稠密多大**。

### 9.1 三方账目逐项（GB，十进制）

| 项 | lead 的账 | scout-b 的账 | 本席核（file:line / 依据） | 判定 |
|---|---|---|---|---|
| GGUF up（pin，驻留） | 31.70 | 含在"非驻留 42.5 + 驻留 20.9 = 63.4"里 | **31.71**（40 层 × 756.0 MB = 30,240 MiB = 29.53 GiB；pin 日志 `C:\sp3\R57-D0-20261018.log` "pinned 40 layers (59.06 GiB)" 的一半；`engine.cpp:1285-1305`） | 双方一致；**Stage D 后仍需全驻留** |
| GGUF down（row-major，D6a 跳过） | 0 | **42.5 全在"非驻留"里**（63.4−20.9） | 现状 **31.71**；post **0**（`engine.cpp:2312` span 删、`:1301` 不 pin、`relu2_down_open` 不读——见 §5） | **42.5 不是额外副本**：它是 63.4 里 up+down 的**非驻留页**（`情报-AD:81` 原文）；Stage D 后 down 那半不在读集，up 仍全驻留 ⇒ **42.5 对 post 没有对应项** |
| sidecar T down（pin，驻留） | 31.71 | 31.7 | **31.71**（`expect=40×512×1,548,288 B=31.71GB`，`engine.cpp:509`；R57 日志 "sidecar PINNED (29.53 GiB)"） | 双方一致；**post 仍需全驻留**（decode CPU down + GPU staging） |
| MTP blk.88 专家（up+down） | 未列（draft off） | 含在"dense 6.7"里 | **1.58 GB**（up 0.79 + down 0.79；GGUF 载荷核算：总专家 65.00 = 主 63.42 + MTP 1.58；MTP 层 = `n_layer_`，`engine.cpp:2204`；sidecar 只覆盖 40 个主 MoE 层 `:487-498/:560-563`） | `--no-draft` ⇒ 该层不创建（`block_layer_=draft_` `:962`）⇒ **post 0**；draft on 则 down 0.79GB 仍读 GGUF（计划外项） |
| 非专家/dense（GGUF 非专家张量） | 4.7 | 6.7 | **5.09 总**：`token_embd` **0.30**（保持映射、每 token 读，`engine.cpp:2190-2200/2778-2779`）+ `output` **0.44** + 其它 dense **4.35**（载荷核算，A 级）；`output/other` 由 `dense_ = dmalloc` 上传 VRAM（`:2224-2226`），GGUF 页只在 **load 读一次**、之后可回收 ⇒ **硬驻留仅 0.30**（token 表；首次 touch 冷页 0.3GB） | scout-b 的 6.7 = 非专家 5.09 + MTP 1.58（70.1−63.42）⇒ 把 MTP 专家当 dense，且把 load 一次性读当驻留；lead 4.7 = output 0.44+其它 4.35，但它们是 **VRAM 上传的一次性读**，非硬 RAM |
| OS/其它进程 | 未列 | 10 | **≈10（D 级估算）**（`情报-AD:82`；本机 96GB，日志 pin 后 free 74–80GiB 与之相容） | 必须计入；lead 的 68.1 是"数据账"，不含 OS |
| housekeeping（host pinned/staging/KV/swap 缓冲） | 未列 | 3–5（"会吃掉余量"） | **≈1–1.5（D 级）**：`--stage-mib` 是 **VRAM** staging（`stage_`），非 RAM；host 侧为 h_r_/h_x_/h_stage_（pin 全层时 host gather 不跑，`h_stage_` 仅非 pin 路径）⇒ 远小于 3–5 | scout-b 高估 |
| **合计（post Stage D）** | **68.1**（数据，无 OS） | **91** | **≈75.2**（=up 31.71 + sidecar 31.71 + token 0.30 + housekeeping 1.5 + OS 10；draft off；MTP/dense 一次性读不计） | **闭合，余 ≈20.8GB**（96−75.2）；把 4.79 非专家页缓存也算上是 80.0 ⇒ 余 ≈16，仍闭合 |

### 9.2 结论（对 lead 三问）

1. **scout-b 的 42.5 不是"额外副本"**：`情报-AD:81` 原文 "63.4GB 里 **非驻留的 42.5GB** 正是 prefill staging/CPU miss 读取/换出后读取的工作集" ⇒ **它已包含在 63.4GB 内**（63.4−20.9 驻留）。所以 scout-b 的 91/96 是**pre-Stage-D 的账**（把 down 读集 42.5 带进 post 为一项），**不是 Stage D 后的账**；按 Stage D 后重算不成立。
2. **lead 的 68.1 方向对、口径需两处修正**：① 4.7GB dense 中 output+other 是 **load 一次读、上传 VRAM**（硬驻留只有 token 表 0.30GB）；② 68.1 **不含 OS ~10GB**。修正后 post ≈ **75.2GB**（含 OS），**余 ≈20.8GB**，仍闭合。
3. **D6a 后 `map_pages` 剩多少**：**63.4GB → 31.71GB**（只留 up 的 span，删 `engine.cpp:2312` 的 down span；spans 见 `:2307-2313`，map_pages `:185-195`）；sidecar 另走 **prefault/touch 31.71GB**（R57c-fix `:1352-1360`，pin 不 prefault 已证）；dense 4.79GB 不在 spans 内、只在 load 读一次；token 表 0.30GB 常映射。**up 区 31.7GB 必须预热**（pin 只注册不 fault；CPU miss 与 prefill staging 都读 up；不预热则冷页读盘）。
- **pin 预算**：post pin up 29.53 + sidecar 29.53 = 59.06 GiB；+8 GiB 余量（`engine.cpp:1270-1277`）= **67.06 GiB ≤ 96GB（89.4GiB）⇒ 活**；现状 59.06+29.53+8 = **96.59 GiB > 96 ⇒ 死**（与 R57 A/B 一致）。
- **pin 的 prefill +53.7% 保留**：Stage D 后 up 从 GGUF pin、down 从 sidecar pin，**都仍是 DMA 直拷**（`stage()` 的 pin 分支 `:3326-3347`）⇒ 不丢这条收益；丢的只是"down 从 GGUF 直拷"改为"down 从 sidecar 直拷"。

### 9.3 冷页风险（存在但可控）

| 冷页来源 | 量 | 何时冷 | 代价 / 依据 |
|---|---|---|---|
| up 区 | 31.7GB | 若 `map_pages` 不预热 up（或 OS 回收） | 必须预热：CPU miss 与 prefill staging 读 up；map_pages 保留 up span 即暖（`engine.cpp:2307-2313/185-195`） |
| **sidecar** | 31.7GB | **若不做 prefault/touch** | **R57-E1 已证死**：decode **1.59 tok/s**、`router_wait 104.9 ms/token`、ledger `router_wait=35462.1 ms`（页错风暴）；R57c-fix 的 touch（`:1352-1360`）必须保留 |
| token 表 | 0.30GB | 首次 touch | 一次性；5.7GB/s ⇒ ~0.05s；可加 span 预热（当前 spans 不含它） |
| MTP down | 0.79GB | draft on 且保留 GGUF down | 一次性/每 draft 读；交付 `--no-draft` ⇒ 0 |
| dense(4.79) | load 一次 | 只在上传 VRAM 时读 | 之后不再读（可回收），非 decode 冷页 |
- scout-b 的 "10% 冷页 ≈ 20–28 ms/token" 是把 **42.5GB 非驻留页全冷**当常态；Stage D 后 up 全驻留+sidecar 预触 ⇒ **不存在每 token 20–28ms 的常态冷读**；残余一次性冷项 ≈1.1GB（token 0.3+MTP 0.79）⇒ ~0.2s（5.7GB/s，D 级）。
- **结论**：**Stage D 后 RAM 闭合**（≈75.2GB，余 ≈20.8GB；含 OS 估算）；冷页风险 **可控**，前提是两项预热/预触保留（up 的 map_pages span + sidecar 的 touch/prefault）。若 OS 实际 14–16GB 或 housekeeping 3–4GB，余量降到 ~11–14GB，仍闭合（但不再"宽裕"）——这个下界要写进立项风险。

## §8 未验 / 落盘

- 未验：没有实际编译/运行 Stage D（本任务零 GPU、只读码）；非专家 RAM 4.6 GB 是估算；MTP 保留 down 的 0.74 GiB 未在日志核对（--no-draft 交付不触发）；D6b 的 70 GB 重写未做。
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\验收\中心假设独立验证-StageD解锁RAM-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\RV-20261018-中心假设独立验证.md`
- 只读依据：`PolyStrata\src\poly\{engine.cpp,engine.hpp,cpu_experts.cpp,cpu_experts.hpp,nemotronhmoe.cpp}`（行号见正文）；`C:\sp3\onechunk_pin.log:82`；`C:\sp3\R57-E1-20261018.log`；`P3-20261016\计划-StageD-GPU侧T布局-20261018.md`；`P3-20261016\R57-ReLU2-StageC-20261018.md`。
