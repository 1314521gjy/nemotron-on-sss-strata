# P1 — PolyStrata 判定（只读取证）

- 日期：**2026-10-16**
- 席位：p1-scout（第三号专武 P1 情报席）
- 判定对象：`VecSzn/PolyStrata`
- 取证副本：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\`（clone 自 `https://github.com/VecSzn/PolyStrata`，`--depth 1`，HEAD `eb5ef11`）
- 红线遵守：未碰 `E:\strata`；clone 只落实验区；**未编译**；未跑 GPU
- 等级：`A`=本机实测（本轮亲自读到的码/档）｜`C`=外部读数｜`D`=推论/估计
  - `A(码)` = 直接读源码 ｜ `A(档)` = 直接读仓内文档（内容为仓作者自述，读数本身是本机的）

---

## 0. 三句话

1. **新层型挂点**：`src/poly/engine.cpp:816` 的 `Engine::open()`（按 `general.architecture` 造类，第 821 行是挂点）+ 各架构类 `layer_head()` 里的**逐层 `if/else` 分派**（如 `src/poly/lfm2moe.cpp:255-256`）。
2. **同源程度**：`src/poly/` 是**独立实现**，不是 fork 改名；与 strata 引擎只共用**极薄**的头文件面（`include/strata/core` 40 头里只 1 个、`include/strata/kernels` 30 头里只 1 个）。PolyStrata 同时**完整保留**上游引擎 ⇒ B ⊇ A。
3. **负控命中**：LFM2.5 的 30/40 层三抽头短卷积先例**在码在测**（`src/poly/lfm2moe.cpp:202`、`src/poly/kernels.cu:2490`、测试 `src/poly/shortconv_test.cu`，CMake 目标 `poly_shortconv_test`）⇒ **不是"没有"，是命中**。

**另有一条推翻项目书定性的读数（§4）**：`arch_defaults` **不是**几何泛化层，是 GPU 架构（gfx1151/gfx1200）的**环境开关默认表**。项目书 §2.2 对它的定性**错**（`A(码)`）。

---

## 1. 判据一：新层型挂在哪一个函数 / 分派点

### 1.1 工厂分派（架构级）— `A(码)`

`src/poly/engine.cpp:814-822`：

```cpp
// The engine for a model file: its general.architecture says which.
std::unique_ptr<Engine> Engine::open(const Options& o) {
    const Shards file(o.model);
    const MetaValue* arch = file.get("general.architecture");
    const std::string name = arch ? arch->s : std::string();
    const auto make = [&]() -> std::unique_ptr<Engine> {
        if (name == "glm5-next" || name == "glm5next") return std::make_unique<Glm5Next>();   // Unsloth: no hyphen
        if (name == "qwen35moe") return std::make_unique<Qwen35>();
        if (name == "gemma4") return std::make_unique<Gemma4>();
        if (name == "deepseek4") return std::make_unique<DeepSeek4>();
        if (name == "lfm2moe") return std::make_unique<Lfm2Moe>();
        throw std::runtime_error(o.model + " is a '" + name + "' model, which this engine does not run");
    };
```

- **分派点**：`Engine::open()` @ `src/poly/engine.cpp:814`；**新架构加一行**在第 **821** 行。
- 机制：**字符串比较链**（不是白名单 enum，不是注册表），未知架构**硬抛**、不静默降级。
- 文档同述：`docs/BACKENDS.md:173-189`（"How a model is recognized"，表里 5 行 ↔ 码里 5 行，**完全一致**）。

### 1.2 逐层分派（层型级）— `A(码)`

`src/poly/lfm2moe.cpp:250-256`：

```cpp
void Lfm2Moe::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    ...
    if (B.attn) attn(B, pos_, n);
    else conv(B, n);          // ← 短卷积层
```

`src/poly/qwen35.cpp:316`（同构）：`if (B.gdn) gdn(B, n); else <attention>;`

- **层型分派点**：架构类的 `layer_head()`。Mamba-2 cell 在此处是**第三条分支**。
- 「B.attn / B.gdn」是逐层布尔，来源见下。

### 1.3 层型从哪来（数据驱动，非取模）— `A(码)`

| 架构 | 层型判据 | file:line |
|---|---|---|
| LFM2.5 | GGUF 数组 `attention.head_count_kv`，**0 = 卷积层** | `src/poly/lfm2moe.cpp:50-60`（读）→ `:88` `B.attn = kv_heads_[il] > 0` |
| Qwen3.6/Ornith | **张量在场**：`attn_qkv.weight` 在 ⇒ GDN | `src/poly/qwen35.cpp:86` |
| GLM 5.3 Flash | **张量别名**（`kda_*` ↔ `ssm_*`）+ 在场 | `src/poly/glm5next.cpp:37-46` |

- 三种都是**逐层数据驱动**，**全仓 `src/poly/` 内无任何取模式交错**（`grep "interval\|% n_layer"` 在 `src/poly/*.cpp|hpp` 只命中测试里的随机数，`A(码)`）。
- ⇒ LFM2.5 用 `kv_heads_` 数组带**逐层显式表**（`src/poly/lfm2moe.hpp:63`），这是"非等周期交错"可直接复用的机制。

### 1.4 其余挂点（加一个架构要动的全部地方）— `A(码)` + `A(档)`

| 处 | file:line |
|---|---|
| 1. 类声明 + `Block` 结构 | 新 `src/poly/<arch>.hpp`（样板：`lfm2moe.hpp:13-77`） |
| 2. 几何/权重/分派实现 | 新 `src/poly/<arch>.cpp` |
| 3. 新 kernel | `src/poly/kernels.cu` + 声明 `include/strata/poly/kernels.hpp` |
| 4. 工厂登记 | `src/poly/engine.cpp:821` |
| 5. 构建登记 | `CMakeLists.txt:1818-1820`（`strata-poly` 目标源列表）与 `:1844`（CPU 专家目标） |

**文档配方**：`docs/BACKENDS.md:1935-1944`「Adding an architecture」5 步：
1. 读 llama.cpp 的同架构图（`src/models/<arch>.cpp`）= 参考实现
2. 从 `Engine` 派生一个类；**原文点名 `qwen35.cpp` 是"同时带 recurrent 层与 attention 的最短样例"**
3. 新 kernel 放 `src/poly/kernels.cu`；要求"窗口猜 token 时一个 token 的 kernel 结果与邻居无关"
4. 在 `Engine::open()` 与 `CMakeLists.txt` 里命名
5. **任何调优之前先与 llama.cpp 对拍**（同文件、同提示、贪心；短/长提示、专家全在 CPU、带/不带猜 token 各一遍）

> **判据一结论**：能明确指认。架构级挂点 `engine.cpp:821`，层型级挂点 `layer_head()` 的 `if/else`。**等级 A**。

---

## 2. 判据二：`poly` 与 `strata` 的同源程度

### 2.1 结论：**独立实现**（非 fork 改名）— `A(码)` + `A(档)`

| 证据 | 读数 | 等级 |
|---|---|---|
| 命名空间 | `strata::poly`（全部 `src/poly/`）vs `strata::core`（`src/core/`） | `A(码)` |
| `class Engine` 唯一性 | **全仓只有 `src/poly/engine.hpp:147` 定义 `class Engine`**；上游 strata 引擎**没有这个类**（它的入口是 `src/program/generate.cpp` + `src/core/layer.cpp` 的函数式路径） | `A(码)` |
| 自有 kernel | `src/poly/kernels.cu` **137 KB** + `include/strata/poly/kernels.hpp`（自有命名空间 `strata::poly`） | `A(码)` |
| 自有运行时 | `src/poly/runtime.hpp:118` 定义 `class Shards`（GGUF 读入），**不来自上游** | `A(码)` |
| 自有入口 | `src/poly/main.cpp`（50 KB） | `A(码)` |
| **共享头面（关键量化）** | `grep '#include "strata/' src/poly/` 全部命中：`strata/poly/kernels.hpp` ×15（自有）、**`strata/kernels/native_mmvq.hpp` ×8（借上游）**、`strata/spec/draft_policy.hpp` ×2、`strata/artifact/gguf_reader.hpp` ×2、`strata/spec/suffix_drafter.hpp` ×1、**`strata/core/conversation_memory.hpp` ×1（借上游）** | `A(码)` |
| 下游方向 | `src/core`、`src/qwen35`、`src/kernels` 里 **0 处** include `poly/`（单向依赖，poly 是叶子） | `A(码)` |
| 链接面 | `CMakeLists.txt:1818-1820`：`strata-poly` = `src/poly/*` + **仅 `src/core/conversation_memory.cpp` 一个上游 .cpp** | `A(码)` |
| 作者自述 | `README.md:20-25`：*"It runs eight models, on **two engines** … [Qwen3.8-Flash-Next] **Its engine is the Strata project's, unchanged.** The other seven run on **an engine written for PolyStrata**"* | `A(档)` |

- **共享面占比**：上游 `include/strata/core/` 有 **40 个头文件**，poly 只碰 **1 个**；`include/strata/kernels/` 有 **约 30 个**，poly 只碰 **1 个**（`native_mmvq.hpp`）。
- ⇒ **不是 fork 改名**，是「**同仓两个引擎**」：上游引擎**原样保留**（`src/core` 23751 行、`src/kernels` 53324 行、`src/prefill` 13455 行、`src/program` 14565 行、`src/qwen35` 3379 行、`metal/`、`sycl/` 全在，`A(码)`），poly 引擎并排新增（`src/poly` 13714 行）。
- ⇒ **B ⊇ A（能力面）**：选 B 不会丢掉 A 的引擎，只是多一个引擎可挂。**这一条对基座裁决有直接意义**，项目书未记。

### 2.2 顺带修正：PolyStrata 的 `src/core` **不是** 1.43.1 的那份 `layout.hpp` — `A(码)`

- 项目书 §2.3 记基座 A 的 `include/strata/core/layout.hpp`（97 行）里有 `qsa_interval = 4` / `is_qsa_layer() = (layer % 4 == 3)`。
- **PolyStrata 里的同名文件是完全另一个东西**：`include/strata/core/layout.hpp`（72 行）是 `WeightTable` / `LayerView` / `check_layer`，**没有 `qsa_interval`**（`grep qsa_interval include/strata/core/layout.hpp` = 0 命中，`A(码)`）。
- 交错逻辑在 PolyStrata 里搬到了 `include/strata/core/geometry.hpp` + `family_layout.hpp`（见 §4.2）。
- ⇒ 若走 B，"按 A 现行路径重核几何"这条（项目书缺口表 #3）**必须重做一遍**，不能照搬 1.43.1 的路径清单。**这一条是给 P0/P4 的输入**。

---

## 3. 判据三：能否容纳 Nemotron 的 Mamba-2 + MoE + 注意力交错

分项判定（**"能容纳"= 机制在，或改动面小且已指认；不含"已实现"**）：

### 3.1 MoE（含专家驻留/流式）— **能容纳，且是成熟件** `A(码)`

- 这是 poly 引擎的**立身之本**：`Engine` 基类注释 `src/poly/engine.hpp:3-5`：*"Everything but the routed experts goes to VRAM; as many routed experts as fit go there too, and the rest are computed on the CPU from the mapped file while the GPU does its share."*
- 挂点齐备：`route()` / `route_card()` / `moe_shared()` / `router_scores()` / `load_moe()`（`dl` 侧 LFM2.5 实现见 `src/poly/lfm2moe.cpp:278-311`：sigmoid 门 + bias 选 top-k + 权重归一 → **与 Nemotron 的 MoE/Sigmoid 家族同族**）。
- CPU 专家路径独立成件：`src/poly/cpu_experts.cpp`（11 KB）。

### 3.2 注意力 — **能容纳** `A(码)`

- 现成：fp16 KV cache（`store_f16`）、flash 式 prefill（`attn_flash64_kernel`）、分块 decode（`attn_score_reg_kernel` / `attn_value_t_kernel`）、`head_norm_rope`。
- 但 poly 的注意力是**逐架构类内实现**（`Lfm2Moe::attn()` / `Qwen35::attn()`），**不是共享基类** ⇒ Nemotron 的注意力（若与两者都不同）要**在 Nemotron 类里写一遍**。

### 3.3 非等周期交错 — **能容纳**（poly 侧）`A(码)`

- §1.3 已证：层型来自逐层数据（数组 / 张量在场），**无取模**。
- 反证（strata 引擎侧）：`include/strata/core/geometry.hpp:76` `is_qsa_layer = layer % qsa_interval == qsa_interval-1`；`family_layout.hpp:92` 若 `n_layers % qsa_interval != 0` **直接 `bad(...)` 硬拒**。详见 §4.2。

### 3.4 Mamba-2 cell — **原料大部分在场，但 cell 本体要新写** `A(码)` + `D`

**在场（关键）**：poly 已有**三个 recurrent cell 先例**，其中 GDN 用的**就是 Mamba 的参数族**：

| 件 | file:line | 对本任务的意义 |
|---|---|---|
| `gdn_step(... const float* dt, const float* a, float* s ...)` | `src/poly/kernels.cu:2504` | **`dt` / `a` 两个 Mamba 参数已在 kernel 签名里**；`s` = `v_heads*s_dim*s_dim` 逐头方阵状态 |
| `conv4_silu`（4 抽头因果卷积 + silu + 状态） | `src/poly/kernels.cu:2478` / kernel `:1754` | **因果短卷积 + 携带状态**已实现（Mamba-2 是 d_conv=4）|
| `shortconv`（3 抽头门控短卷积，状态 = 2 向量/层） | `src/poly/kernels.cu:2490` / test `src/poly/shortconv_test.cu` | 见 §5 负控 |
| `kda_conv` / `kda_step` | `src/poly/kernels.cu:2139` / `:2152` | 第三个 recurrent 家族（GLM/KDA）|
| 逐头 S×S 状态的分配与存取 | `src/poly/qwen35.cpp:113-114`、`:121-124`、`:163-168` | **Mamba-2 状态（n_head × d_state × d_state）的形状与生命周期已跑通** |
| 状态的任务级存/取 + 猜 token 的 snapshot | `recurrent_floats()` / `save_layer()` / `snap_bits_` 机制 | Mamba 的 SSM cache 存档可直接套 |
| fp16 cache + 随机舍入的落点 | `store_f16` / `kv_alloc`（`lfm2moe.cpp:126-141`） | 项目书 §3.6「SSM cache FP16」的宿主已有 |

**不在场（要新写）**：
- **Mamba-2 的选择性扫描（chunked scan）本体**：`gdn_step` 是门控 delta 规则（带 q/k L2 归一与 beta/gate），**不是** Mamba-2 的 `A_log/dt/D` 选择性扫描。要新写一个 kernel（约一层的量级）。
- `D` 跳连项、Mamba-2 的门控 RMSNorm（`ssm_norm`）在 poly 侧的落点——`gdn` 里有 `o_norm`，形状相符，但语义要对齐。
- **规模风险**：Nemotron 的 `d_state`（Mamba-2 常为 128/256）比 GDN 的 `s_dim`（64）大，`gdn_step` 注释写 *"a head of 32 to 256 values, in 32s"*（`src/poly/kernels.cu:2507`），⇒ 若 d_state > 256 需扩 kernel 约束（**未验**，取决于 P0 抽出的真实几何）。

### 3.5 MTP（草稿块）— **能容纳** `A(码)`

- 基类有 `check_draft_block()` / `block_attend()` / `block_rows()` / `draft_step()` / `guess_vocab()`（`src/poly/engine.hpp:285-293,349-350`）；`qwen35` 是一个完整实现样例（`qwen35.hpp:28,34`）。
- LFM2.5 恰好**没有**草稿块（`docs/LFM.md:185`）⇒ 说明该能力**可选**，不阻塞。

### 3.6 视觉（C-RADIO + VL connector）— **不在 poly 引擎内** `A(档)`

- `docs/BACKENDS.md:125-129`：图片走**第二个子进程** `strata-vision`（"llama.cpp's image code with the model's encoder file"），引擎侧只通过 `sees()` / `whole()` 挂钩，把图的行当 token 嵌入。
- ⇒ 与项目书缺口 #5 判断一致：**建议后置，先只做文本**。
- 与项目书 §2.2 的差异：项目书说"Strata 是 `strata-vision.exe` + mmproj 路线，不同族"——**在 poly 里同样是这个路线**，所以 B 不比 A 多也不比 A 少。

### 3.7 关键未知（**未验**，不许当结论）

1. **Nemotron 的 GGUF 元数据里有没有 poly 能读的逐层信号**。poly 三例分别用「`attention.head_count_kv` 数组」、「`attn_qkv.weight` 在场」、「张量别名」。Nemotron-H 族在 llama.cpp 里用的是**另一套张量命名**（`nemotron-h.cpp`）。**poly 的类能不能直接按张量在场区分 Mamba 层 / Attention 层，取决于 P0 抽出的实际张量名单**——**未验**。
2. **Nemotron 的交错是否等周期**——属 P0，未抽（项目书 §7 已列）。
3. **`d_state` 的规模**是否超出 `gdn_step` 的 256 上限——见 §3.4。
4. **poly 的专家缓存能力与上游 `--expert-cache*` / `--expert-profile` / `--pcie-frac` / `--peer-device` / CPU 瀑布是否等价**。项目书 §0 把"Strata 那套专家驻留"当作**核心资产**。poly 引擎**有自己的一套**（VRAM 缓存按使用频率填充 + CPU 计算 + `--expert-swaps` 等，见 `engine.hpp:3-5`、`src/poly/engine.cpp:1068 swap_`），但**功能是否对齐未逐项比对**——**未验**。⚠️ 这是走 B 的**最大未验风险**。

---

## 4. 顺带核对项

### 4.1 `arch_defaults` 能否覆盖非等周期交错 —— **问题不成立；且项目书对它的定性是错的**

**`arch_defaults` 是 GPU 架构开关表，不是几何泛化层。** `A(码)`

- `include/strata/core/arch_defaults.hpp:1-11` 原文：*"the per-architecture defaults of switches that are exact everywhere. Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora …"*
- 签名：`arch_default_env(const char* gcn_arch)` —— 入参是 **`hipDeviceProp_t::gcnArchName`**（"gfx1151"），不是模型几何。
- 表内容全是**环境变量开关**：`STRATA_GDN_HEAD` / `STRATA_Q8_PACKED` / `STRATA_MMVF_ROWS` / `STRATA_HFUSE`… （`src/core/arch_defaults.cpp:24-56`）。
- 测试用例全部是 GPU 名：`arch_defaults_test.cpp:28-35` 断言 `gfx1100` / `gfx1150` / `gfx1152` / `gfx11510`（非前缀匹配）/ `""` / `nullptr` **返回空**。
- ⇒ **它只影响"哪些 kernel 变体默认开"，与"层型/层序"零关系。**
- ⚠️ **项目书 §2.2 第 53 行把 `arch_defaults` 记作「这就是把编译期几何泛化出来的那一层」——该定性错误**（`A(码)`）。项目书 §2.4 的"路线 B 省掉泛化几何这一大块"**因此不成立**：B 的几何泛化能力在**架构类 + 逐层数据驱动**里，不在 `arch_defaults`。

### 4.2 非等周期交错的真正归属（A vs B 的硬对比）`A(码)`

| | 基座 A（strata 引擎 `src/core`，PolyStrata 里原样保留） | 基座 B（poly 引擎 `src/poly`） |
|---|---|---|
| 层型判据 | `is_qsa_layer(g,l) = (l % qsa_interval == qsa_interval-1)`<br>`include/strata/core/geometry.hpp:76` | 逐层数据：数组 `kv_heads_[l]`（0=非注意力）或张量在场<br>`src/poly/lfm2moe.cpp:88`、`qwen35.cpp:86` |
| 周期来源 | GGUF `full_attention_interval`（`family_layout.hpp:61`），默认 4（`geometry.hpp:26`） | 无周期概念 |
| 非等周期的后果 | `family_layout.hpp:92`：`if (g.n_layers % g.qsa_interval) return bad("the layer count is not a multiple of full_attention_interval")` ⇒ **直接拒载** | **天然支持**，无需改机制 |
| 一致性守卫 | `gguf_reader.hpp:730-744`：若 GGUF 带 `attention.compress_ratios` 数组，逐层校验它与取模**必须一致**，不一致报错 | 无此约束 |
| 层型枚举 | `enum class LayerKind { Gdn, Attention, Mtp }`（`family_layout.hpp:139`）——**只有三种，没有 Mamba 位** | 逐架构 `Block` 结构体自由定义（`lfm2moe.hpp:41-56`、`qwen35.hpp:44-60`） |

> **对项目书缺口 #2 的更新**：项目书说"若 Nemotron 交错非等周期 ⇒ 必须改成显式层型表"——**方向对，但 A 侧比预想更硬**：不只是"改"，而是**拒载**（`n_layers % interval` 那条会先炸），且 `LayerKind` 是三值枚举。B 侧**零成本**。
> ⇒ 若 P0 抽出的交错**确为非等周期**，B 在这一点上的优势是**判定性的**，不是程度问题。

### 4.3 `docs/BACKENDS.md` 的加后端步骤是否适用于 Nemotron —— **适用** `A(档)` + `D`

- §1.4 已列 5 步，**通用**，且第 2 步**主动点名** `qwen35.cpp`（recurrent + attention 样例）。
- 第 5 步（与 llama.cpp 对拍）与本项目书 P2/P5 的判据**同构**；llama.cpp 侧有现成参考：项目书 §4 记 PR #30520 已 merged，`src/models/` 有 `mamba2.cpp` / `nemotron-h.cpp` / `nemotron-h-moe.cpp` / `nemotron.cpp`。⇒ **第 1 步（参考图）与第 5 步（对拍基准）都有货**。
- **附加约束（须记账）**：
  - `docs/BACKENDS.md:1908-1933`：模型要能跑，还要 `tools/strata_tokenizer.py` 造 tokenizer 目录 + 写 `strata-<name>.json` + 在 `setup.py` 的 `FAMILIES` 里挂号。
  - **平台面**：`BACKENDS.md:1946-1952` 限制节明说 *"NVIDIA cards only… GLM 5.3 Flash was not run on Windows"*，且 README 记 **poly 的 7 个模型只有较小的 5 个支持 Windows**。⇒ **Nemotron 走 poly 时，Windows 可用性未验**（本机是 Windows + RTX 4080S）。
  - **与 PR #1518 的差别**：poly 的做法是**直接改 5 处**（新 .hpp/.cpp + `engine.cpp` + `CMakeLists.txt` + `kernels.cu`），**不是** `add_subdirectory(<model>_ext)` 外挂子目录。⇒ 若走 B，"外挂子目录"这一条项目书 §6.2 的对策**与 poly 的现行惯例不符**，要另议。

### 4.4 关键词重核（修正项目书 §2.2 的一处偏差）`A(码)`

| 词 | 项目书记 | 本轮精确重核（`src/ include/ docs/ serve/ tools/`，排除 `.git`/`bench`） |
|---|---|---|
| `mamba` / `Mamba` / `Mamba2` | 0 | **0 / 0 / 0** ✅ 与项目书一致 |
| `nemotron` | 0 | **0** ✅ |
| `state_space` | 0 | **0** ✅ |
| **`SSM`** | 项目书未记 | **386 命中** ⚠️ |

- `SSM` 命中不是噪声：`src/core/layer.cpp:229` 真实 **加载** Mamba 参数族张量 ——
  `ssm_out.weight` · `ssm_alpha.weight` · `ssm_beta.weight` · **`ssm_conv1d.weight`** · **`ssm_dt.bias`** · `ssm_a` · `ssm_norm.weight`；
  `src/core/layout.cpp:69,100` 把它们列进布局表；`src/kernels/elementwise_parity.cpp:10,76` 明写 *"`ssm_a` IS NEGATIVE … the artifact's `ssm_a = -exp(A_log)`"*。
- ⇒ **必须分开说**：字符串 "mamba" **确实 0**；但 **Mamba/SSM 的参数族、状态结构、weight 加载路径在 strata core 里已经落地**——只是以 **GDN（门控 delta 规则）** 的名义。
- **对判据的影响**：这不改变"**cell 算法要新写**"（`gdn_step` ≠ Mamba-2 selective scan），但**削弱**了项目书 §4「Strata 世界从未碰过 Mamba」的表述强度：**碰过参数族与状态，没碰过那个扫描**。

---

## 5. 负控：LFM2.5 的 30 层短卷积（非注意力层先例）—— ✅ **命中，且在码在测**

**未报"没有"——是命中。** 三级证据齐：

| 级 | 证据 | file:line |
|---|---|---|
| 档（作者自述，带层数）| **"Most layers have no attention. 30 of its 40 layers mix the tokens with a short convolution over the last three: each projects a token to three vectors B, C and x, convolves B times x, and scales the result by C. **Its state is two vectors a layer.**"** | `docs/LFM.md:180-182` |
| 码（结构）| `struct Block { bool attn = false; ///< attention over the cache; else the short convolution` + `Dev in_proj, conv;` + `float* past = nullptr; ///< the two B * x before the next token, 2 * n_embd` | `src/poly/lfm2moe.hpp:41-48` |
| 码（分派）| `if (B.attn) attn(...); else conv(B, n);` | `src/poly/lfm2moe.cpp:255-256` |
| 码（实现）| `void Lfm2Moe::conv(Block& B, int n)`：`lin(in_proj)` → `shortconv(proj_, B.past, ...)` → `lin(wo)` | `src/poly/lfm2moe.cpp:202-207` |
| 码（状态尺寸）| `size_t Lfm2Moe::recurrent_floats() const { return size_t(n_conv_) * 2 * n_embd_; }` ← **"每层只记两个向量"在代码层面精确成立** | `src/poly/lfm2moe.cpp:121` |
| 码（状态分配）| `alloc_caches()`：非注意力层 `B.past = st; st += 2 * n_embd_;` | `src/poly/lfm2moe.cpp:135-138` |
| **kernel** | `void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream, ...)`；kernel `shortconv_kernel` / `shortconv_rows_kernel` / `shortconv_state_kernel` | `src/poly/kernels.cu:2490` / `:1805` / `:1830` / `:1846` |
| **测试（已接 CMake）** | `add_executable(poly_shortconv_test src/poly/shortconv_test.cu src/poly/kernels.cu)`；测试内容 = **host 参考实现 vs device** 对拍输出、**状态、跨调用的状态携带**、以及 snapshots | `CMakeLists.txt:1873`；`src/poly/shortconv_test.cu:22-40,98-102` |

**行为规格（`shortconv_test.cu:21,30-33`）**：
```cpp
// out = C * (s0 * w0 + s1 * w1 + bx * w2), bx = B * x; the state moves on by a token.
const float bx = in[c] * in[2 * kChannels + c];
const float y = st[0] * w[3*c] + st[1] * w[3*c+1] + bx * w[3*c+2];
st[0] = st[1];  st[1] = bx;
out[row*kChannels + c] = in[kChannels + c] * y;
```
= **门控线性递归，2 抽头状态**——**是 Mamba-2 cell 的退化形态**（没有 `dt` 选择性、没有 A 矩阵、没有 d_state 维）。**这就是"非注意力层占多数"的现成先例**，且带**容差断言**（`e_out < 1e-5f && e_state == 0.f`）与**跨调用状态一致性**检查。

**另有两个补充先例**（同判据、非注意力层）：
- `Qwen35::gdn` —— **带 `dt` / `a` 的真 recurrent**：`src/poly/qwen35.cpp:207-216`（`gdn_step(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, ...)`），`Block::gdn` 分派 `qwen35.hpp:44`。
- `Glm5Next` 的 KDA：`src/poly/kernels.cu:2139,2152`。

> **负控结论（等级 A）**：命中，且**不是"源码里有名字"，而是"有 kernel + 有 host 对拍测试 + 有 CMake 目标 + 有跨调用状态判据"**。

---

## 6. 对 A/B 基座的倾向（**建议，不代裁**）

### 6.1 两条路在本轮取证下的事实差

| 维度 | A：1.43.1 上做 | B：以 PolyStrata 为上游 |
|---|---|---|
| 新层型挂点 | 需自己找（项目书 §2.3 只到 GDN 原料层） | **已指认**：`engine.cpp:821` + `layer_head()`（判据一达成） |
| 非等周期交错 | **硬伤**：`is_qsa_layer` 取模 + `n_layers % interval` **拒载** + `LayerKind` 三值枚举（§4.2） | **零成本**：逐层数据驱动（§4.2） |
| recurrent cell 先例 | GDN 1 个 | **3 个**（GDN / KDA / shortconv）+ 其中 GDN 已带 `dt`·`a`·S×S 逐头状态 |
| 非注意力层占多数的先例 | 无（QSA/GDN 是 1:3，注意力占 1/4，但 GDN 是注意力式 recurrent） | **LFM2.5 30/40**，且带测试（§5） |
| 专家流式体系 | 上游原件（项目书中心资产） | poly **自带一套**；**是否等价未验**（§3.7-4）⚠️ |
| 加后端的文档 | 无（项目书 §4 记 lcz 那篇是**失败报告**：常量散落 14 文件、粗估 6000–10000 行） | **有 5 步配方**且点名 recurrent 样例（§1.4） |
| 上游引擎是否丢失 | — | **不丢**：B 完整保留上游引擎（§2.1） |
| Windows 可用性 | 现役二进制就在 Windows | poly 的 7 模型中仅 5 个支持 Windows，**Nemotron 档位未验** ⚠️ |
| 第三方仓风险 | 上游官方 | 126★ / 2026-10-06 建仓 / MIT |
| 升版冲突 | 上游单模型意图 | 同上（两引擎并存，冲突面更小？**未验**） |

### 6.2 我的倾向（`D`）

1. **技术上，B 在"容纳 Nemotron 新层型"这件事上明显占优**，且优势集中在两处**判定性**的点：非等周期交错（B 零成本 vs A 拒载）、加后端配方与先例密度（5 步 vs 失败报告）。
2. **但项目书 §2.4 给 B 记的那条收益（"省掉泛化几何这一大块"）不成立**——`arch_defaults` 不是几何层（§4.1）。B 的真实收益是"**逐层数据驱动的层型机制**"，不是 `arch_defaults`。**这一点必须改写，否则会照错的理由做决策。**
3. **走 B 前必须补的三条未验**，任何一条为负都可能翻转结论：
   - ① **Nemotron 的 GGUF 张量名单能否让 poly 类直接区分 Mamba/Attention 层**（§3.7-1）；
   - ② **poly 的专家缓存/CPU 瀑布与上游 `--expert-cache*` 那套是否等价**（§3.7-4）——因为项目书 §0 把上游那套定为"**资源，不是对手**"，若 poly 那套不等价，B 的核心卖点就换人了；
   - ③ **poly 在 Windows 上能否跑 Nemotron 这个体量**（§4.3）。
4. 若 ①②③ 全部为正，我倾向 **B**；若 ② 为负，则"以 A 为基座 + 照抄 poly 的层型机制与 5 步配方（PR #1518 外挂子目录）"是更稳的折中。
5. **P0 的交错结果是本轮的**第一性输入**：若 P0 证明 Nemotron 交错**等周期**（例如 Mamba:Attn = 3:1 且周期恒定），则 A 的硬伤消失，A/B 差距大幅缩小，**不必为交错而换基座**。

> 以上为**建议**。基座裁决属决策人（项目书 §9-1）。

---

## 7. 未验清单（明写，不许当结论用）

1. Nemotron GGUF 的逐层张量名单与 poly 可读信号的对齐性（§3.7-1）— **未验**
2. Nemotron 交错是否等周期（§3.7-2）— 属 P0 — **未验**
3. Nemotron 的 `d_state` 规模 vs `gdn_step` 的 256 上限（§3.4）— **未验**
4. poly 专家缓存能力 vs 上游 `--expert-cache*` / `--expert-profile` / `--pcie-frac` / `--peer-device` / CPU 瀑布的逐项等价性（§3.7-4）— **未验**（走 B 的最大风险）
5. poly 各模型在 Windows 上的实际可用面，尤其中大体量（§4.3）— **未验**
6. **PolyStrata 的 `src/core`（上游引擎）与 `E:\strata-v01431`（v0.1.43.1）是否逐文件同一** — **未验**。本轮只证明了"同名 `layout.hpp` 内容不同"（§2.2），**没有**做树对树 diff（本轮红线：只读 PolyStrata）。**建议派给 verifier 或 P0 席**：`git -C E:\strata-v01431` 与本副本的 `src/core`/`include/strata/core` 逐文件 sha256 比对。这是判断"B 的上游引擎是否原样"的唯一硬证据。
7. poly 的 `deepseek4` / `gemma4` 是否含**第四条 recurrent 先例** — 本轮只读了三例（GDN/KDA/shortconv），**未逐读 `deepseek4.cpp`（29 KB）与 `gemma4.cpp`（30 KB）全文**。
8. `docs/` 下另有 `LFM.md` / `GLM.md` / `DEEPSEEK.md`，只读了 `LFM.md` 与 `BACKENDS.md` 的关键节 — 未通读。
9. `multi-GPU`（`Options::layer_split`）路径对 Mamba 类状态的兼容性 — **未读**。

---

## 8. 本轮读数索引（全部为本机亲读）

| # | 读数 | file:line | 等级 |
|---|---|---|---|
| 1 | 架构工厂 + 新架构挂点 | `src/poly/engine.cpp:814-822`（挂点 `:821`） | A(码) |
| 2 | 逐层型分派点（短卷积先例） | `src/poly/lfm2moe.cpp:255-256` | A(码) |
| 3 | 逐层型分派点（GDN 先例） | `src/poly/qwen35.cpp:316` | A(码) |
| 4 | 层型判据 = GGUF 数组（0=非注意力） | `src/poly/lfm2moe.cpp:50-60,88`；`lfm2moe.hpp:63` | A(码) |
| 5 | 层型判据 = 张量在场 | `src/poly/qwen35.cpp:86` | A(码) |
| 6 | `class Engine` 唯一定义处（同源判据） | `src/poly/engine.hpp:147` | A(码) |
| 7 | 两引擎并存 + 平台限制 | `README.md:20-25`、`docs/BACKENDS.md:1946-1952` | A(档) |
| 8 | 加后端 5 步配方 | `docs/BACKENDS.md:1935-1944` | A(档) |
| 9 | 加同架构模型 6 步（tokenizer/run config/FAMILIES） | `docs/BACKENDS.md:1908-1933` | A(档) |
| 10 | `Engine` 的 hook 契约表 | `docs/BACKENDS.md:144-156` | A(档) |
| 11 | **arch_defaults = GPU 架构开关表**（推翻项目书定性） | `include/strata/core/arch_defaults.hpp:1-11`；`src/core/arch_defaults.cpp:13-56`；`arch_defaults_test.cpp:28-35` | A(码) |
| 12 | strata 引擎：单周期取模 | `include/strata/core/geometry.hpp:26,65-66,73-77` | A(码) |
| 13 | strata 引擎：非等周期**拒载** | `include/strata/core/family_layout.hpp:92`；一致性校验 `include/strata/artifact/gguf_reader.hpp:730-744` | A(码) |
| 14 | `LayerKind` 三值枚举 | `include/strata/core/family_layout.hpp:139-143` | A(码) |
| 15 | 家族白名单只有 2 族 | `include/strata/artifact/model_family.hpp:19-25,57-61` | A(码) |
| 16 | 负控：LFM2.5 30/40 层 + 状态=2 向量 | `docs/LFM.md:180-182` | A(档) |
| 17 | 负控：短卷积结构/分派/实现/状态量 | `lfm2moe.hpp:41-48`；`lfm2moe.cpp:121,135-138,202-207` | A(码) |
| 18 | 负控：`shortconv` kernel 与测试接线 | `src/poly/kernels.cu:2490`；`src/poly/shortconv_test.cu:21-40,98-102`；`CMakeLists.txt:1873` | A(码) |
| 19 | GDN 带 `dt`/`a`/S×S 状态（Mamba 参数族） | `src/poly/kernels.cu:2504-2507`；`src/poly/qwen35.cpp:113-114,207-216` | A(码) |
| 20 | 上游 core 真实加载 `ssm_conv1d`/`ssm_dt`/`ssm_a` | `src/core/layer.cpp:229,290,300-303`；`src/core/layout.cpp:69,100`；`src/kernels/elementwise_parity.cpp:10,76` | A(码) |
| 21 | `strata-poly` 目标的完整源列表（只借 1 个上游 .cpp） | `CMakeLists.txt:1818-1820`, `:1844` | A(码) |
| 22 | poly → 上游的 include 面量化 | `grep '#include "strata/' src/poly/`（6 类，共 29 处） | A(码) |
| 23 | 同名 `layout.hpp` 内容不同（B 的 core 已漂移） | `include/strata/core/layout.hpp`（72 行，无 `qsa_interval`） | A(码) |
| 24 | 关键词重核：`mamba`/`Mamba2`/`nemotron`/`state_space` = 0；**`SSM` = 386** | 全仓 grep | A(码) |

---

*本件为 P1 只读取证产出，未编译、未跑 GPU、未碰 `E:\strata`。所有判据均带 file:line；未验项集中在 §7。*
