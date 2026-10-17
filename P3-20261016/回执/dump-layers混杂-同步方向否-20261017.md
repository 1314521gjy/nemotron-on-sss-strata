# 回执（补）：`--dump-layers` 是**混杂臂** —— 它根本不进 `moe_rows()`，不能支撑"缺同步"

```
状态: 达成（推翻"缺同步"方向；三条代码链 + 一条运行日志等价性证据；零 GPU、零改码）
席位: scheduler（引擎后端-调度）
时间: 2026-10-17 15:0x 起，约 15 分钟
任务单 version: 队长消息「pb=32 + --dump-layers ⇒ 正确」+ 四个同步问题
```

## §0 一句话

> ★★ **`--dump-layers` 会把 `rows_` 强制设为 1**（`engine.cpp:1592`）⇒ `L.mmq = … && rows_ >= 32` 恒假（`:1623`）⇒ `mmq_` 根本不建 ⇒ `batched()` 恒假（`:1878`）⇒ **`moe_rows()` 一次都没跑**。
> ⇒ **"dump ⇒ 0 NaN" 只证明 `moe()`（每 token 一次）是好的，对 `moe_rows()` 的同步一个字都不能说。** 这条"决定性实测"是**混杂臂**，与昨晚 `snap_` 那次不同型。

## §1 三条代码链（A，逐行）

| # | 位置 | 事实 |
|---|---|---|
| 1 | `engine.cpp:1592` | `rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);` ⇒ **dump ⇒ rows_=1**（与 `--prompt-batch 32` 无关） |
| 2 | `engine.cpp:1623` | `L.mmq = opt_.gpu_experts && rows_ >= kStageRows && …` ⇒ rows_=1 < 32 ⇒ **每层 `L.mmq=false`**；`:1624` `tiles=false` ⇒ `mmq_` **不创建**（`:1627`） |
| 3 | `engine.cpp:1878` | `batched() = mmq_ && L.mmq && n >= kStageRows` ⇒ **恒 false** ⇒ `layer()` 的 `if (batched(L,n)) moe_rows(...)`（`:2730`）**永不成立** |
| 4 | `engine.cpp:1593` | `read_rows_ = rows_ < kStageRows ? rows_ : …` ⇒ **1** ⇒ `read()` 每次只收 1 token（`:2826`/`:2837`）⇒ 700 token 的 prompt = **700 次 `eval(t,1)`** |

## §2 运行日志的强证据：**dump 下 pb 完全不生效**（A）

| 读数 | 证据 |
|---|---|
| `C:\sp3\dlc_pb32.log` 与 `C:\sp3\dlc_pb8.log` | **同 6,173 行、同 617,669 字节**；**掩掉计时字段后 `diff` 为空**（只有载入耗时/速度行不同）；**末尾 ids 完全相同** `1605,3831,5627,3385` |
| ⇒ 结论 | **`--dump-layers` 下 `--prompt-batch 8` 与 `32` 是同一份计算** ⇒ pb 被 `rows_=1` 抹掉 |
| `C:\sp3\dl2_pb32.log`（700 token，pb=32 + dump） | **123,741 行**，其中 `dump pos=…` = **123,732**，恰 = **700 token × 88 层 × 2 个阶段 ≈ 123,200** ⇒ 逐 token 的 `moe()` 路径；`12.3 tok/s`（1-token 调用的速度） |
| dump 标签分布 | `attn_in` 61,864 · `mamba` 28,120 · `ffn` 28,120 · `attn` 5,624 · `out_norm` 4 ⇒ **全部是层首/层尾**（`nemotronhmoe.cpp:332-339`），**`moe_rows()` 内部没有任何 `dbg`** |

⇒ **"dump 零 nan" = `moe()` 路径零 nan**；而 `moe()` 在 n=1 时本来就正确（`in_w/out_w` + `latent_down`，`engine.cpp:1883-1909`）。**这是已知事实的重复，不是新证据。**

## §3 顺带澄清 D1/D2/D3（避免再混口径）

- 计划件 `计划-分环节定位杠杆-20261017.md:198-202` 记的 **D1/D2/D3 是 `pb=10` 的臂**，且 **X/Y 都是通顺文本**（同件 `:189-191`：X=`" The history of computing…"`、Y=`" The abstractions are not lies…"`）⇒ **D2 的"正确"是 pb=10 的 CPU/GPU 分工漂移（X 串），不是 pb=32 上 `moe_rows` 跑通**。
- 因此 **"D2 对 / D3 错"不能当 `moe_rows` 的证据**（D2 根本没进 `moe_rows`）。
- dump 跑出的是 **Y 串**（`dl2_pb32.log` 尾部 `1531,48365,2655,1584`），与"1 token 一次 ⇒ CPU 专家参与更多"一致，**不是"正确修复"的证明**。

## §4 队长点名的四处同步 —— 逐条核（**未发现缺失等待**）

| # | 问题 | 读数 | 判 |
|---|---|---|---|
| 1 | `plan_ahead`/`stage` 之后谁等？ | 早拷贝在 `up_` 上：`stage(L, L.ahead.data(), n_ahead, up_)`（`:2242`）+ `ev_ahead_`（`:2243`）；层内第 0 轮 `cudaStreamWaitEvent(cs(st_), ev_ahead_, 0)`（`:2447-2448`）；下一调用复用缓冲前 `cudaStreamWaitEvent(cs(up_), ev_free_, 0)`（`:2441`） | ✅ 齐 |
| 2 | `quantize(mq_)` 与 `mul` 同流？ | 两者都在 `st_`（`:2378` 与 `:2419`） | ✅ 流内有序 |
| 3 | `moe_sum` 与 CPU 侧 | `card()` 虽在 `behind` 线程，但**用同一条 `st_`**（`:2475`）；`cpu_experts` 是同步调用（`:2479`），`behind.wait()`（`:2482`）在 `moe_sum`（`:2492`）与 `moe_add`（`:2493`）之前 ⇒ 卡侧**入队**顺序 + CPU 侧**返回**顺序都对 | ✅ 齐（★ 这条不是病灶） |
| 4 | `n_cpu==0` 时 `dx_` 谁清 | 不需要清：`n_cpu=0 ⇒ rg=rr`，`bounds` 铺满 `[0,rr)`，第二 mul 用 `d_place` 写回 token-major（`:2428`） | ✅ 见前一件 §4-3 |
| 附 | 图？ | `Gb_pb32_nographs.log` 仍 NaN | ✅ 与 graph 无关 |

★ **附加**：dump 的 `cudaStreamSynchronize(cs(st_))`（`dbg()` `:1826`）**只同步 `st_`**，对 `up_` 上未到的事件**不会**替我们补等待；所以"dump 治好"即便真发生，也**不能**证明 `up_/st_` 缺等待——它只能证明 `st_` 内部的某些乱序。**方向本身就不成立。**

## §5 ⇒ 建议的"干净"同步臂（若还想验同步；**必须不改 `rows_`**）

1. **临时 env 钩子**（builder 做、用完 revert）：`STRATA_SYNC_EACH=1` 时在 `moe_rows()` 的 `card()` 前后各下一次 `cudaStreamSynchronize(st_)` **但不改 `rows_`**（区别于 `opt_.dump`）。若 NaN 消失 ⇒ 才叫"缺同步"。
2. **`compute-sanitizer --tool memcheck`**（已批准，且**不改变几何**）：最小复现 `pb=32` + 短 prompt + `--max-new 1`。★ 本假设的预测很具体：**会报 `invalid __global__ read`，落在 MMQ 的 up/down 乘积**（每专家按 4096 宽读 1024 宽矩阵 ⇒ 读 4×，arena 尾专家越界）。**这是最省的一击**；memcheck 干净则我的"宽度"结论要撤回。
3. 若要"不碰代码"的同步验证：`nsys` 看 `st_` 与 `up_` 的 event 依赖图（但只能看，不能判数值）。

## §6 未验 / 边界

1. **【未验】** 同步假设本身（本件只证"dump 臂不构成证据"，未证"有/无缺同步"；`§5` 给了干净臂）。
2. **【未验】** 上一件的"latent 宽度"结论仍未实测（memcheck/修法后可定）。
3. **【A】** 全部运行读数取自 `C:\sp3\dlc_pb8.log` / `dlc_pb32.log` / `dl2_pb32.log`（同机、同 exe、同时段）。
4. 本件**只读、零 GPU、未改码、未碰 `E:\strata`**。

## §7 落盘

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\dump-layers混杂-同步方向否-20261017.md`
- 前件：`…\回执\moe-vs-moe_rows宽度diff-20261017.md`
- 证据日志：`C:\sp3\dlc_pb8.log` · `dlc_pb32.log` · `dl2_pb32.log` · `Gb_pb32_nographs.log`
- 源码：`engine.cpp:1592, 1593, 1623, 1826, 1878, 2242-2243, 2378, 2419, 2428, 2441, 2447-2448, 2475, 2479-2493`

## 阻塞

无。
