# P3 施工（切法 A）— 读数与范围更正（2026-10-16）

- 席位：cell-probe（施工）｜任务：#12
- 工作树：`PolyStrata\`（实验区 clone，HEAD `eb5ef11`）
- 纪律：未碰 `E:\strata`；编译前清 .obj（KP-UNI-0203）；**本机真实构建标准是 C++20**（`engine.cpp` 用了 `#include <bit>` / `std::bit_width` / `std::popcount`）——check 脚本据此。

---

## 1. 已落盘并**编译通过**（切法 A 两块）

| 文件 | 改动 | 内容 |
|---|---|---|
| `include/strata/poly/kernels.hpp` | +23 | `mamba2_conv_silu` / `mamba2_scan` / `mamba2_gated_norm` 声明 |
| `src/poly/kernels.cu` | +146 | 三个内核（**从 P2-20261016 已验证实现搬迁**，KP-NINFER-0086） |
| `src/poly/nemotronhmoe.hpp` | 新 85 | `NemotronHMoe` 架构类声明 |
| `src/poly/nemotronhmoe.cpp` | 新 314 | 架构类实现（几何 + 层型表 + Mamba-2 mixer + 注意力 + 挂点） |
| `src/poly/engine.cpp` | +2 | `nemotron_h_moe` → `NemotronHMoe`（挂点 :821 处）|
| `CMakeLists.txt` | +2/-2 | `nemotronhmoe.cpp` 加进 `strata-poly` 与 `poly_draft_race_test` 两个目标 |

**编译读数（A，本机 nvcc 13.3 / sm_89 / C++20 / 清 .obj 单文件编译）**：

```
kernels.cu          -> BUILD_OK
nemotronhmoe.cpp    -> ARCH_OK
engine.cpp          -> ENGINE_OK     （唯一改动=+1 include +1 分派行）
```

脚本 `build_check_kernels.cmd` / `build_check_arch.cmd`（都在清 .obj 后单次编译）。

## 2. ★ 一条更正 team-lead 指令的实测发现：**层型表不能只靠 `head_count_kv`**

team-lead 说"层型表直接用 GGUF `attention.head_count_kv` 89 项数组"。**不够**：

- llama.cpp 判层型是 `is_recr(i) = n_head_kv(i)==0 && n_ff(i)==0`（`nemotron-h.cpp:16-18`）。**该数组 0 同时覆盖 mamba 层与纯 MoE 层**，只有 2 = 注意力层。⇒ 数组只能分"注意力 / 非注意力"，**分不出 mamba 与 MoE**。
- 真正的区分维度是**每层 n_ff**，而 GGUF 不把它作为标量键暴露，只在**张量在场**上体现。

⇒ 我把层型表建在**张量在场**上（poly 既有机制，`qwen35.cpp:86` 同法）：`ssm_in.weight` ⇒ mamba；`ffn_gate_inp.weight` ⇒ MoE；`attn_q.weight` ⇒ 注意力；三者皆无 ⇒ 硬抛。
`head_count_kv` 作**交叉校验**：有 `attn_q` 的层其 kv 必须 > 0（`nemotronhmoe.cpp` load_geometry 内 `need`）。

## 3. 架构类实现要点（对齐 llama.cpp `nemotron-h.cpp`，`A(码)`）

- **每层一个模块 + 残差**（`:191-199, :186-211`）：`layer_head` = norm → (mamba | attn) → 残差加回 `r_`；`layer_tail` 空（Nemotron 无 FFN）。
- **Mamba-2 mixer**（`:74-90`，`mamba-base.cpp:190-200`）：`ssm_in` 的**融合投影**一行给 `z | xBC | dt`（d_in_proj = 2·d_inner + 2·n_group·d_state + n_head = 18560）→ `mamba2_conv_silu`（吃 strided 输入）→ `mamba2_scan`（dt 带 `ssm_dt.bias`、strided）→ `mamba2_gated_norm`（z strided）→ `ssm_out`。
- **注意力**（`:244-258`）：照 `lfm2moe.cpp` 的 attn；`attn_q_norm`/`attn_k_norm`/`attn_output.bias` **按张量在场可选加载**（GGUF 实测 attn_* 约 5 后缀，可能无 qk-norm）。
- **几何硬抛**：`ssm.conv_kernel∈2..8`、`d_inner % n_head`、`n_head % n_group`、`d_state∈1..512`、`d_state>512` 等 → `std::runtime_error`；内核层另有 `std::invalid_argument`。

## 4. MoE 层——**仍未接线，且加载即熔断**（不是静默降级）

`load_layer` 遇到 MoE 层**直接抛**一条指名错误（"the expert pipeline is gated and n_embd-wide"）。理由：引擎的 slot 布局是 `gate|up|down`、专家输入/输出宽 = `n_embd_`，而 Nemotron 非 gated、吃 latent 1024；若硬塞会让引擎**静默算错**。宁熔断不静默。

## 5. ★ 范围更正（关键）：单改 `load_moe` **不足以**让 Nemotron 跑起来

（team-lead 已据此把 MoE 通路泛化拆成"下一步"，本件保留证据）

1. **专家 gated**：`engine.cpp` moe() `native_mmvq(..., n_embd_, 2*n_ff_exp_, ...)` → `swiglu_clamp_rows`；slot `down = slot + 2*L.gu_bytes`；CPU 侧 `cpu_experts.cpp:206-218` 同算 gate+up。Nemotron **非 gated、只有 up+down、relu2**（`nemotron-h.cpp:284` 传 `nullptr` gate + `LLM_FFN_RELU_SQR`）。
2. **专家宽 = `n_embd_`（4096）**：moe() 输入 `xn_`、`moe_router` 拷 `n_embd_` 行、`cpu_experts(..., n_embd, n_ff_exp_, ...)`。Nemotron 吃 **latent 1024**（`:111-112` up_exps `{moe_n_embd, n_ff_exp, n_expert}`）。

⇒ 需动**跨文件**：`moe()` / `moe_rows()`（MMQ 批）/ `stage()` / `moe_card()` / slot 布局与 `fill_expert_cache` / `cpu_experts()` 的点积+激活，估 150–250 行，多轮构建。

## 6. 未验 / 下一步

- **未验**：`NemotronHMoe` 未接真实 GGUF（权重在下载），**层张量名未与 GGUF 目录逐条对账**（llama.cpp 源名 vs GGUF 名，侦察 §8.3 已证二者可能不同）；注意力是否真无 qk-norm/输出 bias 未核（已按可选处理）。
- **负控**：几何硬抛已实现（可红）；**MoE 侧的负控（故意错 latent 宽/漏 relu2 必须红）待 MoE 通路落地后补**。
- **下一步**：MoE 通路泛化（非 gated relu2 + latent 输入/输出 + `ffn_latent_up/down`）+ MTP（blk.88 折叠块的 `nextn.*`）。

---

## 7. MoE 通路泛化 — 第 1 轮（本机实测，A）

**本轮落盘（PolyStrata 树内）**
- `include/strata/poly/kernels.hpp` +2、`src/poly/kernels.cu` +13：**`relu2(x)=max(x,0)^2`** 内核（llama.cpp `LLM_FFN_RELU_SQR`；Nemotron 专家非 gated，激活是 relu2 不是 `silu(gate)*up`）。
- `src/poly/nemotron_moe_test.cu` 新增 ~170：**可跑对拍测试**（host fp64 oracle vs 卡上 `gemv_f32 → relu2 → gemv_f32`，latent 1024 / FF 2688）。
- `CMakeLists.txt`：新目标 `poly_nemotron_moe_test`。
- 脚本 `build_run_moe_test.cmd`（清 obj → C++20 nvcc → 跑）。

**对拍读数（A，RTX 4080S / sm_89）**

```
relu2 block: align 1.38e-05 (tol 1e-04)  |  neg gated-oracle 1.38  |  neg latent-width throw yes
=>  GREEN  (neg1 red ok, neg2 red ok)   EXIT=0
```

- **正控**：卡上 relu2 块 vs host fp64 oracle，误差 **1.38e-05**（容差 1e-4）。
- **负控 1（可红）**：同一卡上输出 vs **gated（`silu(g)*up`）** oracle → **1.38**（>> 100×tol）⇒ 判据**能分辨"非 gated relu2"与"gated swiglu"**，正是本轮要防的错。
- **负控 2（硬抛）**：latent 宽故意报 4096（实际 1024）→ `throw std::invalid_argument: ffn_up_exps is not [n_ff, latent]=[2688, 4096]`。

**本轮未做（下一轮）**：把 `relu2`/latent 宽**接进引擎的专家通路**——`moe()` / `moe_rows()` / `stage()` / `moe_card()` / `fill_expert_cache()` 的 slot 布局（现为 `gate|up|down`，需支持 `up|down`）/ `cpu_experts()` 的点积+激活 / `layer` 的专家输入宽（latent vs `n_embd_`）/ `moe_router` 拷 latent。**架构类的 kMoe 分支仍熔断**（`load_layer` 抛），因为通路未接完，接了才会静默算错。

---

## 8. MoE 通路泛化 — 第 2 轮：**少 token 路径已接**（`A`）

**改动（PolyStrata 树内，7 文件 +345/−58；另 3 个新文件）**
- `engine.hpp`：`Layer` 加 `gated / relu2 / exp_in / exp_out / latent_down / latent_up` + `slot_up()/slot_down()`；`load_moe` 增签名；引擎加 `moe_in_ / moe_out_ / moe_up_ / moe_in_max_ / moe_out_max_`。
- `engine.cpp`：`load_moe`（**非 gated 分支**：只 up+down，`exp_gate=nullptr`；shared 无 gate；宽按 latent）、`fill_expert_cache`/`swaps`/`spans`/`mmq` 判定的 slot 布局（`up|down` 兼容 `gate|up|down`）、`moe_router`（算 `latent_down @ xn_` 并量化）、`moe()`（专家宽 `in_w/out_w`、非 gated 走 `relu2`、累加进 `moe_out_`）、`moe_add`/新增 **`moe_latent`**（`latent_up @ moe_out_` 加进 `fo_`）、`shared_scratch`/`alloc`（按最宽专家宽分配）。
- `cpu_experts.{hpp,cpp}`：**`gate==nullptr` 即非 gated**（跳过 gate 点积、激活改 relu2、输出宽可选）。
- `nemotronhmoe.cpp`：kMoe **不再熔断**——`ffn_latent_down/up` 加载 + `load_moe(il, true, false, latent, latent)` + `route()`（sigmoid+probs bias+norm+scale）+ `moe_shared()`（4096 上的 up→relu2→down）+ 残差（`layer_head` 加 mixer 残差 / `layer_tail` 加 `fo_`）。

**编译读数（A，nvcc 13.3 / sm_89 / **C++20** / 清 .obj）**：`ARCH_OK · ENGINE_OK · CPUEXPERTS_OK · KERNELS_OK`（脚本 `build_check_arch.cmd`）。

**非 gated 层守卫**：`L.mmq=false`（`batched()`⇒false）与 `moe_card()` 头部 `if(!L.gated) return false` ⇒ 少 token 走 `moe()`，prompt 批**暂**也走 `moe()`（正确但慢），`moe_rows()` MMQ 批路径**下一轮**。

**本轮未做 / 未验**
- `==TX==/==NEW==` 行标记**未补**（KP-NINFER-0086 计数），下一轮补。
- `moe_rows()`（~300 行 MMQ 批路径）未泛化。
- 通路级**运行时负控**（真跑引擎、故意错 latent 宽）需 GGUF 权重，**权重仍在下载**；`relu2` 块的负控（§7）已可红。
- 未接真实 GGUF，层张量名未逐条对账。

---

## 9. 权重前的小活：TX/NEW 标记 + latent 宽门升生产（2026-10-16 续）

**1. `==TX==/==NEW==` 行标记已补**（KP-NINFER-0086 计数）。`src/poly/kernels.cu` 的 P3 新增块内：
- `// ==TX==` … `// ==/TX==`：移植自 llama.cpp 的 Mamba-2 selective scan 算术（softplus + scan/conv/gated_norm 三内核）。
- `// ==NEW==` … `// ==/NEW==`：本引擎侧接口（`relu2` 内核 + 四个 host 入口 + 几何门）。

读数（`P3-20261016\count_lines.py`，脚本从 P2 复用）：

```
kernels.cu: 2698 lines total
  transplanted (TX)  : 92
  new-written (NEW)  : 58
  ratio TX/NEW       : 1.59  → GREEN（KP-NINFER-0086 要 TX > NEW）
```

**2. latent 宽门升到生产路径**（verifier caveat ①）：`NemotronHMoe::load_geometry` 新增硬门
`need(n_moe_ == 0 || n_latent_ > 0, "a MoE layer whose moe_latent_size is 0")`；
且**每张专家矩阵与两条 latent 投影都在加载时按该宽校验**（`load_moe()` 的 `exps()` 形状断言 + `upload()`）——
错宽在读到任何专家前就抛，不是测试本地才有的门。几何日志也带出 latent 宽。

**编译读数（A）**：`ARCH_OK · ENGINE_OK · CPUEXPERTS_OK · KERNELS_OK`（C++20 / 清 .obj）。

---

## 10. verifier 抓的缺陷已修 + U43 张量名对账（2026-10-16 续）

### 10.1 `moe_out_` 未清零（verifier PARTIAL 的高置信缺陷）— **已修**
`add_scaled`/`axpy(add=true)` 是 read-modify-write；gated 路靠 `moe_shared` 先写 `fo_` 覆盖，latent 路却换到全新缓冲 `moe_out_` 而没清零 ⇒ routed 和 = 未初始化内存 + Σ专家。
**修**：`Engine::moe()` 里，latent 层进入累加前 `cudaMemsetAsync(moe_out_, 0, n*out_w*sizeof(float), st_)`——与 `fo_` 被 `moe_shared` 覆盖对称。修后清 .obj 重编：`ARCH_OK · ENGINE_OK · CPUEXPERTS_OK · KERNELS_OK`。

### 10.2 U43 — 真 GGUF 张量名逐条对账：**PASS**（`gguf_u43.py`，读两片本地头部，未取权重）
- 两片 sha256 已过（team-lead）：39,730,340,416 + 30,372,951,904 B。
- `general.architecture = nemotron_h_moe` ✓；几何全对：block_count 89、embedding 4096、experts 512/22、expert_ff 2688、shexp 5376、**moe_latent_size 1024**、nextn 1、ssm 4/8192/128/128/8、head_count 32、`head_count_kv` 89 项非零在 [7,16,25,36,47,58,69,78,88] ✓。
- **后端要的每个名字都在**：`mamba missing: none`（8 个 ssm 后缀，含无后缀的 `ssm_a`/`ssm_d`）· `attn missing: none` · `moe missing: none`（gate_inp/up_exps/down_exps/up_shexp/down_shexp/**latent_up/latent_down**/exp_probs_b.bias）· `present but not loaded: []`。
- **attn_q_norm / attn_k_norm / attn_output.bias 实测缺失** ⇒ 我为它们做的**按张量在场可选加载**是必需的（否则会硬抛）。
- 唯一未加载的是 **MTP 块 blk.88** 的 `nextn.eh_proj/enorm/hnorm/shared_head_norm` + `post_attention_norm`（MTP 本轮不做）。
- `attn_norm.weight` 每层都有（含 MTP）。

### 10.3 e2e 构建的前置阻塞
`third_party/ggml` **未 vendored**（只有 LICENSE/VERSION.txt/ggml-common.h）；CMake 需要 `-DSTRATA_GGML_DIR=<llama.cpp checkout>`，否则取 pin `3cf03257f219…`。本机唯一现成 checkout 是 `E:\llama-cpp\prism-b10687`（**不同 commit，有 ABI 风险**）。cmake 在 `…VS\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`。**需确认匹配的 ggml 源**才能 configure。
