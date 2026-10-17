# P2 — Mamba-2 cell 最小探针 · 读数

- 席位：cell-probe ｜ 日期：2026-10-16 ｜ 任务：#5（闸门=已拍基座 B）
- 目录：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P2-20261016\`
- 机器：RTX 4080 SUPER（sm_89）+ CUDA 13.3（`E:\cuda-13.3`）+ VS18 BuildTools（**中文 MUI**，KP-UNI-0203 同款）
- 红线：未碰 `E:\strata`；未改 `PolyStrata\` 取证副本（避免污染 verifier 的 U4 diff）；编译前清 .obj。

---

## 0. 一句话

**一层 Mamba-2 cell（合成权重，Nemotron 全尺寸几何）在 fp32 上对一个 float64 参考实现对齐到 ~1e-6（容差 3e-4）；故意错一个 dt 与丢 softplus 两个负控全部变红（0.08–7.7）；几何不匹配 3 例全部硬抛；新写行数 49 < 搬迁行数 116。** 未验：参考是 llama.cpp 公式的 fp64 转写，非 llama.cpp 编译态 op 直调。

---

## 1. 探针是什么

只做**一层** Mamba-2 mixer（Nemotron 的 mamba 层 = `norm → mixer → 残差`，**无 FFN**；FFN 在独立的 moe 层上——base-judge 更正已纳入）。

范围（被验的算法本体，正是 poly 缺的那块）：

```
conv1d(d_conv=4)+bias+SiLU  →  selective scan(state=state*exp(softplus(dt)*A)+B*(x*softplus(dt)); y=<state,C>)
   →  y += x*D  →  y = silu(z)*y  →  grouped RMSNorm  →  cell 输出
```

不在范围：`in_proj` / `out_proj`（普通 GEMM，poly 已有 `gemv_*`）、层残差、MoE、注意力、全模型。
（故探针直接吃 in_proj 之后的 `z / xBC / dt`，与 llama.cpp `ssm_scan` 的输入边界一致。）

文件：

| 文件 | 角色 |
|---|---|
| `mamba2_cell.hpp` | 接口层（重写，poly 风格）；几何结构 + `mixer/mixer_h` 声明 |
| `mamba2_cell.cu` | **port**：TX=搬迁自 llama.cpp 的算法；NEW=新写接口（每处注「正本没有，故新写」） |
| `mamba2_ref.hpp` | float64 host 参考（对拍 oracle） |
| `mamba2_cell_test.cu` | 主：正控 + fp16 路径 + 两个 dt 负控 + 几何硬抛负控 |
| `build_run.cmd` | 清 .obj → 一次 nvcc → 跑 |
| `count_lines.py` | 数 TX / NEW 行 |

几何（**Nemotron 全尺寸**，非缩水）：`d_state=128, d_inner=8192(128头×64), n_head=128, head_dim=64, n_group=8, d_conv=4, eps=1e-5`，`channels=10240`，`in_proj` 输出宽 18560 一致。

## 2. 算法来源与转写审计（file:line）

正本（全部 MIT）：
- `src/models/mamba-base.cpp:151-302` `build_mamba2_layer()` — 步骤次序与形状
- `ggml/src/ggml-cpu/ops.cpp:9578-9628` `ggml_compute_forward_ssm_conv_f32`
- `ggml/src/ggml-cpu/ops.cpp:9648-9868` `ggml_compute_forward_ssm_scan_f32`（Mamba-2 分支 `A.ne[0]==1`）
- `ggml/src/ggml-cuda/ssm-scan.cu:48-133` `ssm_scan_f32<>`（CUDA 路径）
- `ggml/src/ggml-cpu/vec.h:1445` `ggml_vec_swiglu_f32` = `silu(gate)*up`；`ggml.c:3078` `ggml_swiglu_split`

六处易错点，逐一对回正本（**已核**）：
1. **conv 权重布局** = channel-major：`(tap k, channel c)` 在 `c*d_conv + k`（`ops.cpp:9611,9623` 的 `c[i0+i1*nc]`）→ 探针同。
2. **抽头次序**：`seq[t+k]`，`seq=state(d_conv-1)++inputs`，最旧在前（`ops.cpp:9610-9623`）→ 探针 `reg[0]` 最旧。
3. **conv 作用于全通道**（x|B|C 一起卷积），split **在** conv+SiLU **之后**（`mamba-base.cpp:230-246`）→ 探针 `channels` 全宽、从 post-SiLU 张量取 x/B/C。
4. **softplus** = `x>20 ? x : log1p(exp(x))`（`ops.cpp:9711`/`ssm-scan.cu:376-379`）→ 探针 `dt_softplus`。
5. **swiglu 次序** = `silu(z)*y`（gate=z，`mamba-base.cpp:286`）→ 探针 `gated_norm_kernel`。
6. **D 跳连在 swiglu 之前**、**grouped norm 在 swiglu 之后**，norm 宽度 = `d_inner/n_group`（`mamba-base.cpp:284-292`）→ 探针同。

## 3. 正控读数 — 对齐（判据：容差内）

对拍口径：`worst = max |dev - ref| / (1 + |ref|)`，dev=fp32 卡上、ref=float64 host。容差 **3e-4**（tiny 用 1e-4）。

| 用例 | rows | y 误差 | 末态 s 误差 | fp16-cache 误差 | 判定 |
|---|---|---|---|---|---|
| tiny (d_state16) | 5 | 2.59e-08 | 1.35e-08 | 1.73e-04 | ✅ |
| **full-decode** | **1** | **2.79e-07** | 8.97e-08 | 2.57e-04 | ✅ |
| full | 64 | 2.42e-06 | 1.21e-07 | 4.58e-04 | ✅ |
| full | 257 | 1.06e-06 | 7.34e-08 | 5.90e-04 | ✅ |

- fp32 路径**逐比特可复现**（两次跑 `y`、`s` 完全相同，`det y/y`）。
- **U5 答复**：`d_state=128` 在 kernel 的共享内存界内（`MAX_D_STATE=512`，且是**检查**不是假设）；越界 `d_state` 直接抛（见 §5）。旧 `gdn_step` 的 256 上限与本 cell 无关（本 cell 是新写的 scan，不吃那个界）。

## 4. 混合精度：SSM cache = FP16 + 随机舍入

- 卡上状态 cache 以 `__half` 存，写回时**随机舍入**（`sr_half`，counter 哈希定 seed ⇒ 可复现，两次跑 `det y/y`）。
- 读数：fp16-cache 输出对 fp64 参考的误差 **1.7e-4 – 5.9e-4**（容差 1e-2），**有界、不炸**。路径已"留对"。
- **未验**：SR 的期望无偏性（需统计多 seed 的均值，探针未做）；llama.cpp 自身 cache 是 **fp32**，故 fp16+SR 是本引擎按 NVIDIA 配方的**新写**（正本没有），不是搬迁。

## 5. 负控读数 — 必须能红

| 负控 | 结果 | 判定 |
|---|---|---|
| fault=1 `dt` 读错一个头（`dt[h]→dt[h+1]`，经典移植滑手） | y 误差 **0.085 / 0.45 / 2.49 / 2.42**（远 > 100×3e-4=3e-2） | **红 ✅** |
| fault=2 丢 softplus（dt 原值直接用） | y 误差 **0.337 / 1.68 / 4.70 / 7.70** | **红 ✅** |
| 几何 `n_group=7`（128%7≠0） | `throw std::invalid_argument: n_head not a multiple of n_group` | **硬抛 ✅** |
| 几何 `d_state=1024`（>MAX_D_STATE） | `throw: d_state over MAX_D_STATE` | **硬抛 ✅** |
| 几何 `head_dim=96`（≠d_inner/n_head） | `throw: head_dim != d_inner / n_head` | **硬抛 ✅** |
| 几何正常 | 无异常（`gate ok`） | ✅ |

⇒ 判据**能变红**（KP-UNI-0003）；几何不匹配**硬抛、无静默降级**（Strata 规矩）。

## 6. 新写 < 搬迁（KP-NINFER-0086）

`python count_lines.py mamba2_cell.cu`：

```
transplanted (TX) : 116
new-written (NEW) :  49
ratio TX/NEW      : 2.37  → GREEN
```

NEW 块 = fp16+SR（正本没有）、几何门 `check_geom`/`check_and_launch`、两个 host 入口的 launch 胶水；
每块都注「正本没有，故新写：<原因>」。
（说明：NEW 行数含其内的说明注释；纯代码体量为 SR 两函数 + 门 + 两处 launch。）

## 7. 复现

```
cd E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P2-20261016
build_run.cmd            :: 清 .obj → nvcc(sm_89) 一次 → 跑；EXIT=0 为过
```

## 8. 这条判据**不证明**什么（诚实边界）

1. **参考是 fp64 转写，不是 llama.cpp 编译态 op 直调**。转写点已在 §2 逐条对回 `file:line`，但转写误差仍是理论风险。更强的对拍 = 直调 `ggml_ssm_scan`/`libggml`（本机有 `build-cpu/bin/llama-cli.exe`，但需直搭 ggml graph），**未做**。
2. **只证 cell 数学校准**，不证：poly 端接线正确、MoE/注意力/全模型、性能、`ngroups=8` 的直读（仍是 P0 的 `D` 级反解）。
3. **fp16+SR 的期望无偏性未验**（见 §4）。
4. **未做真实权重**：全合成权重；真实 `ssm_a`/`ssm_d` 的数值范围未参与。
