# 回执：`moe()` vs `moe_rows()` 逐行 diff —— 根因 = **`moe_rows()` 从没接 MoE 的 latent 几何**

```
状态: 达成（A 级读码 + GGUF 元数据 + 运行日志 三方互证；根因单点、可一句话说清、解释全部四条实测）
席位: scheduler（引擎后端-调度）
时间: 2026-10-17 14:1x 起，约 35 分钟
任务单 version: 队长消息「逐行 diff moe() vs moe_rows()」+ 点名 5 问
```

## §0 一句话结论

> ★★ **`moe_rows()` 用模型宽度 `n_embd_`（本靶 4096）去乘**1024 宽的**专家矩阵**：
> ① 输入没做 `latent_down` 投影（拿 `xn_` 当专家输入）；② 五个乘积/量化点的宽度全写 `n_embd_`；
> ③ 输出直接写 `fo_`（4096 宽），而 latent MoE 要求写 `moe_out_`（1024 宽）再由 `moe_latent()` 投影。
> ⇒ **与 n 的大小、轮数、class、ahead 全无关，只要进 `moe_rows()` 就错** ⇒ 与队长给的四条实测**全中**。
> **这不是"某个量未初始化"，是"几何量（`L.exp_in/exp_out`）在 `moe_rows()` 里一次都没被读过"。**

## §1 A 级三方互证：本靶的专家确实是 1024 宽，模型是 4096 宽

| 证据 | 读数 | 分级 |
|---|---|---|
| **GGUF 元数据**（直接解析 `…-00001-of-00002.gguf` 头） | `embedding_length = 4096` · `moe_latent_size = 1024` · `expert_feed_forward_length = 2688` · `expert_count = 512` · `expert_used_count = 22` | **A** |
| **加载器**：`nemotronhmoe.cpp:174-177` | `latent = n_latent_ (1024)`；`load_moe(il, true, false, latent, latent)` ⇒ `L.exp_in = L.exp_out = 1024`（`engine.cpp:735-740`） | **A** |
| **形状硬校验**（加载成功即证形状对） | `load_moe()` 的 `exps("ffn_up_exps.weight", in_w, n_ff_exp_)` / `exps("ffn_down_exps.weight", n_ff_exp_, out_w)`（`engine.cpp:759/783`）——用 1024 校验，**GGUF 通过了** | **A** |
| **字节账互证** | up = 1024×2688×0.5625 B = **1.4766 MiB**，down 同 ⇒ 专家 2.953 MiB；日志 `full_pb8.log:3`「8010 of 20480 routed experts (23.1 GiB)」⇒ 23.1 GiB/8010 = **2.953 MiB/专家** ✓ | **A** |

⇒ **专家的输入/输出宽度 = 1024，模型宽度 n_embd_ = 4096，差 4×。**

## §2 两条路径逐点对照（`file:line`，本靶数值）

| 环节 | `moe()`（n<32，**正常**） | `moe_rows()`（n≥32，**NaN**） | 判 |
|---|---|---|---|
| **专家输入宽度** | `in_w = L.exp_in`（`:1901`）、`out_w = L.exp_out`（`:1902`） | ★ **无此二量**；全文用 `n_embd_` | **坏** |
| **专家输入（latent 投影）** | `moe_router()`：`lin(L.latent_down, xn_, moe_in_, n, xq_)`（`:1887`）+ `native_quantize_q8_1(moe_in_, …)`（`:1891`） | ★ **不投影**：`router_scores(L,n)`（`:2229`）只算 router logits；直接 `Mmq::quantize(xn_, d_tok, mq_, gt, n_embd_, n_embd_, rg, st_)`（`:2378`） | ★★ **坏** |
| **xx 行步长** | `xq_row = native_q8_1_bytes(in_w)`（`:1915`）、`xq_ + c.row*xq_row`（`:1979`） | `n_embd_`（`:2378` 的 ld） | **坏** |
| **up/gate 乘** | `gate_up.y = g_ + j*(gated?2ff:ff)`，w 按专家 slot（`:1975-1983`） | `mul(slot_of(r.j), gt, ff2, **n_embd_**, L.slot_bytes, …)`（`:2419`）⇒ **w_cols 4096 vs 实际 1024** ⇒ 每行读 4× | ★★ **坏** |
| **down 乘** | `down.y = d_ + ((i0+j)%kSumRows) * **e_row=out_w**`（`:1983`） | `mul(slot+gu2, dt, **n_embd_**, ff, …)`（`:2428`）⇒ **w_rows 4096 vs 实际 1024** ⇒ 算 4× 行、读 4× 权重 | ★★ **坏** |
| **staged 路径** | —（n<32 不走 stage） | `mul(stage_, …, ff2, **n_embd_**, gu2, …)`（`:2456`）、`mul(stage_+down_at, dt, **n_embd_**, ff, …)`（`:2461`） | ★★ **坏**（同两处） |
| **CPU 专家** | `cpu_experts(…, in_w, …)`（`:1970` 一带，经 `in_w`） | `cpu_experts(*workers_, host_.data(), …, **n_embd_**, n_ff_exp_, h_x_, …)`（`:2479`） | **坏**（本靶 ahead 时 n_cpu=0 不走） |
| **输出落点** | `acc = L.latent_up.p ? **moe_out_** : fo_`（`:1905`）+ 先 `memset(moe_out_, n*e_row)`（`:1908-1909`） | ★ 固定 `moe_sum(dx_, d_w, **fo_**, **n_embd_**, n_used_, n, st_)`（`:2492`）⇒ 1024 宽的结果应进 `moe_out_`，再由 `moe_latent()`（`:2101`，`layer()` 已调）投影 | ★★ **坏** |
| **L.mmq 门** | — | `Mmq::fits(gu_type, gu_w, **n_embd_**)`（`:1623`）⇒ 本应 `L.exp_in`；两者都过整除检查，所以**门没拦住** | 潜伏 |
| `dx_` 尺寸 | — | `fbuf(rr * **n_embd_**)`（`:1577`）⇒ 对 latent 是**多分配**（4096 > 1024），不会越界 | 安全 |

**后果（A 级推理）**：`w_cols = 4096` 而 up 矩阵只有 1024 列 ⇒ 每个专家的一次"点积"跨过自己的 up 矩阵、down 矩阵、直到**下一个专家的 slot**；down 的 `w_rows = 4096` 更把 2688×1024 的矩阵当 2688×4096 用 ⇒ 读 4× 字节。**arena 尾部的专家会读过 arena 边界**（arena 尾部/他缓冲是未定义内容，0xFF 解成 float 即 NaN），与"NaN 而不是有限错"吻合。**所有专家都算错，一层就污染下游。**

## §3 ★ 为什么它同时解释队长给的四条实测（逐条对账）

| 实测 | 本假设的预测 | 中? |
|---|---|---|
| **pb=32 是 `moe_rows()` 的最小入口，已 NaN** | 任何 n≥32 都进这条路径 ⇒ 必错，**与 pb 具体值无关** | ✅ |
| **`sm=2048`（单轮）仍 NaN** | 宽度错在轮的组织之外（`:2419/2428`）⇒ 单轮也错 | ✅ |
| **`sa=0` 仍 NaN** | 与 ahead/staged 记账无关 ⇒ 照样错 | ✅ |
| **class 全程 0 仍 NaN** | 与 `runs_/cb` 无关 ⇒ 照样错 | ✅ |
| **全 CPU 正常**（`--no-gpu-experts`，D1_nogpu.log:6 给对 ids） | `L.mmq=false` ⇒ 走 `moe()`（有 `in_w/out_w`）⇒ 正常 | ✅ |
| **旧 exe 同错**（`978d64bc`） | 上游 HEAD 的 `moe_rows` 同样写 `n_embd_` ⇒ **pre-existing**；上游只用非 latent 模型（Qwen/GLM 的 exp_in==n_embd_）故从未暴露 | ✅ |

## §4 回答你点名的 5 问

**1. 路由/计数（`count_/group_/n_group`）是否一致？—— 一致，不是病灶。**
- `count_` 两边语义相同（每专家被多少 (token, place) 选中）；`moe_rows` 额外把 `ahead` 专家并入 `group_`（`bounds` 可为 0 行），`rows_of(g)=count_[group_[g]]`（`:2315`）自洽。
- 自洽性核过：`rg = Σ count_[group_]`（`:2324`）⇒ `n_cpu=0` 时 `rg = rr`；`rc = bounds[n_cached]`（`:2367`）。
- ★ 唯一副作用（非 NaN）：**`moe_rows` 不更新 `recent_`**（该数组只在 `moe()` 的 `:1936` 更新）⇒ prompt 批期间 swap 策略"看不见"用量 ⇒ 与本靶日志"pb=32 命中 49.5% vs pb=8 76.2%"有关联，但不是正确性问题。

**2. 行 → token 的映射怎么对上？—— 两条路各自自洽；`moe_rows` 是"双坐标系 + 用 ids 桥"**
- `dx_` / `d_` 都是 **token-major**（行 = `t*n_used + place`），这是 `moe_sum_kernel` 的硬假设（`kernels.cu:931-940`：`first = blockIdx.y*n_used`，读 `x[(first+k)*n+i]`、`w[first+k]`、写 `acc[blockIdx.y*n+i]`）。
- `moe_rows` 里：`gu_/mq2_` 是 **专家序（expert-major）**——第一个 mul 的 `ids = iota_`（`:2419`），行号 = `bounds[g]`；第二个 mul 的 `ids = d_place`（`:2428`），`place[row]=i`（`:2338`）把**专家序行号映回 token-major 行号**。staged 一侧同理：`d_rel + (…)` 相对 bounds + `d_place + r0`（`:2461`）。
- ⇒ **映射是通的**（这也是为什么它有"有限但错"的可能，而不是一上来就崩）。**病灶不在这里。**

**3. ★ 你点名的 `dx_` 清零（`n_cpu==0` 时没人 memset）—— 我核了：不是 bug。**
- 覆盖论证（A）：`n_cpu==0` ⇒ 每个路由行都属于 `group_` ⇒ `rg = rr`；`bounds[]` 从 0 单调铺到 `rg`（`:2317-2324`），cached 的 4 段与 staged 的各轮**恰好铺满 `[0, rr)`**；第二个 mul 的 dst 用 `d_place`（`:2428`）把每一行写到 token-major 位置 ⇒ **`dx_` 的 `[0, rr)` 全被写**。
- `n_cpu>0` 时 `memset(dx_, 0, rr*…)`（`:2369`）覆盖 CPU 行 + 未被任何专家铺到的行；同时 `w[i]=0`（`:2333`）。
- ⇒ **`dx_` 初始化这条可以划掉**（与队长此前的 H2 一样，留着只会误导）。

**4. `moe()` 有、`moe_rows()` 漏掉的步骤 —— 三处，都是 latent 几何：**
- (a) **`latent_down` 投影**（`moe_router` `:1887`）—— `moe_rows` 只调 `router_scores`（`:2229`）；
- (b) **`acc = moe_out_`**（`:1905`）—— `moe_rows` 固定写 `fo_`（`:2492`）；
- (c) **`in_w/out_w` 贯穿所有乘积**（`:1901-1915, 1979-1983`）—— `moe_rows` 全用 `n_embd_`。
- （另有 `tail_skip` 只在 `moe()` 实现（`:1949-1958`）；canonical `--tail-skip 0`，不影响本靶。）

**5. ⇒ 一句话：`moe_rows()` 基础路径里，哪个量在 n≥32 时是【未初始化/未更新】的？**
> **不是某个缓冲没初始化**，而是**几何量从未被读取**：`L.exp_in` / `L.exp_out` / `L.latent_down` / `L.latent_up` 在 `moe_rows()` 里**出现 0 次**（`grep exp_in src/poly/engine.cpp` 只命中 1883/1901/2094 等 `moe()` 侧）。
> 因此 `n_embd_`（4096）在 5 个乘积/量化点被当成专家宽度（1024）使用。

## §5 给 builder 的修法清单（**最小改动**，逐处 `file:line`；我这里只出方案不施工）

> 目标：让 `moe_rows()` 与 `moe()` 用同一套几何。**顺序即依赖顺序**。

1. **入口补 latent 投影**（在 `:2229` `router_scores` 之后）：
   `const int in_w = L.exp_in > 0 ? L.exp_in : n_embd_; const int out_w = L.exp_out > 0 ? L.exp_out : n_embd_;`
   `const float* in = xn_; if (L.latent_down.p) { lin(L.latent_down, xn_, moe_in_, n, xq_); in = moe_in_; }`
   （★ 注意 `moe_in_` 也在 `moe_router` 里用 `xq_` 当 lin 的 scratch，`moe_rows` 此时还没用 `xq_`，可复用。）
2. **量化**（`:2378`）：`Mmq::quantize(in, d_tok, mq_, gt, in_w, in_w, rg, st_)`。
3. **cached up**（`:2419`）：`… ff2, **in_w**, L.slot_bytes, …`。
4. **cached down**（`:2428`）：`… **out_w**, ff, …`，且 `dx_` 的行步长改 `out_w`（`d_ = dx_` 的 `ld_dst`）。
5. **staged up/down**（`:2456/`:2461`）：同样 `in_w` / `out_w`。
6. **输出落点**（`:2492`）：`moe_sum(dx_, d_w, L.latent_up.p ? moe_out_ : fo_, out_w, n_used_, n, st_)`；**若为 latent，卡侧也要先 memset `moe_out_`**（照 `moe()` 的 `:1908-1909`；或保留 `dx_` 的 memset 语义后整体拷）——★ 注意 `moe_add()`（`:2493` → `:2094-2095`）已经按 `exp_out` + `moe_out_` 写，**卡侧与 CPU 侧必须落到同一个缓冲**。
7. **CPU 侧**（`:2479` + `:2214/2216` 的 `h_x_` 拷贝）：`n_embd_` → `in_w`；`h_x_`/`h_cpu_` 已按 `max(moe_in_max_,n_embd_)` 开（`:1645-1646`），不用改尺寸。
8. **门**（`:1623`）：`Mmq::fits((int) L.gu_type, gu_w, L.exp_in)`，别再用 `n_embd_`。
9. **`dx_` 尺寸**（`:1577`）：`rr * n_embd_` → `rr * moe_out_max_`（**可留**：现在是多分配，安全；改小省显存）。
10. **寄存器/量化器**：`mq_` 的 `q8_bytes(rr, n_embd_)`（`:1651`）→ `q8_bytes(rr, in_w)`（同样可留，多分配安全）。

**验证口径（给 verifier/队长）**：修完后 **pb=32 的 ids 应与 pb=8 逐位相同**（`1531,6330,1307,23716,1395,1605,1261,6330`）；top-probs 允许末位漂移（两条路径的求和顺序/tile 宽度不同）——**负控**：pb=8 必须逐位不变（`moe()` 侧一行不动）。

## §6 顺带复核队长的否证：**同意，并补一条"为什么混类本身不是错"**

- 队长的否证成立：`tile_class(32)=0`（`engine.cpp:51-55`，`rows > kTile[c]`）⇒ pb=32 全程 class 0 ⇒ 旧代码本就不混类。
- ★ 再补一条（A）：**即使混类，旧代码也是对的**——`cb` 的循环是 `for (int c = 0, at = 0; c < kClasses; ++c)`（`:2353`），**`at` 在 for-init 里声明、只初始化一次、跨类累积** ⇒ `cb[c]` 是**全局行起点**（类 c 的起点 = 低类总行数 + 类内前缀），不是"类内紧凑"。例：A(c0,5 行) + B(c1,40 行) ⇒ `cb[0]=[0,5,5]`、`cb[1]=[5,5,45]`，B 取 `[5,45)` ✓。`most[c]` 只对类 c 专家非零，他类专家 `bounds` 相邻相等 ⇒ 0 行 ✓。
- ⇒ **「修法 1」（按 class 切 runs）对正确性是 no-op**（数学等价，只是 launch 数可能变多）；留着无害，但**别把它当救兵**、别为它单开 GPU 验证。真正要验的是 §5。

## §7 未验 / 边界

1. **【未跑】** 全部为读码 + GGUF 头解析 + 既有日志；**修法与验证口径未实测**。
2. **【未验】** NaN 的具体来源（是"读过 arena 边界读到 0xFF"还是"Q4_K scale 字节被当权重导致 inf"）——**不影响"宽度错"这条结论**，只影响叙述。
3. **【未核】** `Mmq::fits(d_type, n_embd_, n_ff_exp_)`（`:1623` 第二项）对 latent 的 tile 选择是否与 `out_w` 不同（两者 %128 都为 0，预计同档）。
4. **【未核】** `moe_rows` 不更新 `recent_` 对换入策略的量化影响（`pb=32` 命中 49.5% 的成因之一，未定）。
5. 本件**只读、零 GPU、未改码、未碰 `E:\strata`**。

## §8 落盘（绝对路径）

- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\moe-vs-moe_rows宽度diff-20261017.md`
- 关键源码：`…\PolyStrata\src\poly\engine.cpp`（`:1577, 1623, 1883-1915, 1979-1983, 2214-2216, 2229, 2378, 2419, 2428, 2456, 2461, 2479, 2492`）· `engine.cpp:735-740`（`load_moe`）· `nemotronhmoe.cpp:174-177` · `kernels.cu:931-940`（`moe_sum` 的 token-major 假设）
- 运行日志：`C:\sp3\full_pb8.log:3`（专家字节账）· `C:\sp3\Gb_pb32_nographs.log`（pb32 NaN）· `C:\sp3\D1_nogpu.log:6`（全 CPU 正常）

## 阻塞

无。
