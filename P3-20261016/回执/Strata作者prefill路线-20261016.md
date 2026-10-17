# 回执：Strata 作者 prefill 优化路线（git 调研 + 外派稿复核）

```
状态: 部分（外派稿的复核**达成**；"全历史路线表"**未达成** —— 浅克隆，需授权 fetch --unshallow）
席位: scheduler（引擎后端-调度）
时间: 2026-10-16 23:0x 起，约 45 分钟
任务单 version: 队长消息（无盘上任务单）
```

## 交付物

| 绝对路径 | 字节数 | 期望内容在场？ |
|---|---|---|
| `E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\Strata作者优化路线-按prefill-20261016.md` | **25,439** | ✅ **外派正文（§1–§4）一字未改** + **本席复核横幅** + **附录 A「我们开了没」（15 行）** + **附录 B「复核明细」（14 行）** |

## 判据读数

| 判据 | 命令 | 实际读数 | 这条怎么变红 |
|---|---|---|---|
| **是否走成 MCP** | 检查本席工具清单 | ★ **无 MCP 工具**（只有 `Read/Grep/Glob/Write/Edit/Bash/SendMessage/Task*`，**一个 `mcp__*` 都没有**）⇒ **外派不是我发出的**；外派稿由队侧产出后落在同一路径 | 若我具有 `mcp__*` 工具 ⇒ 本条错 |
| **抽检编造（12 条）** | `sed -n` / `git log` 逐条回读原文比对 | **12/12 逐字吻合**：`poly/engine.hpp:122`（`std::vector<int32_t> ahead`）· `:491`（`st_` "the stream everything but the staging ahead runs on"）· `:617-619`（`stage_`/`h_stage_[2]`/`ev_stage_[2]`）· `:620-622`（`up_`/`ev_ahead_`/`ev_free_`）· `poly/engine.cpp:2225-2234`（"…in under half the time the copy ahead takes"）· `:2241-2243`（"the buffer is free once the layer before has read it" + wait/stage/record）· `:2426` · `:2442` · `strata` 侧 `include/strata/core/expert_source.hpp:241-245` · `include/strata/prefill/prefill.hpp:1-8` · `docs/PREFILL_HEAD_ALIAS.md:5-6` · `docs/DETAILS.md:332-336`；另 **`docs/` 被引 9 件全部存在** | **任一条回读对不上 ⇒ 判编造** |
| **编造判定** | 同上 + 全机检索 | 🔴 **发现 1 处**：外派 §4.5「PolyStrata 的 Qwen 引擎来自 Strata（**github.com/Niko121/Strata**）」—— **`Niko121/Strata` 在本机任何记录里都不存在**（我们记录的上游只有 `github.com/VecSzn/PolyStrata`）⇒ **判疑似幻觉 URL，不许引用** | 若能证明该仓存在 ⇒ 更正 |
| **结构性问题：混引** | 按引用路径分流 | §1.1/1.6/1.7/1.9/1.10/1.11/1.12/1.13 = **`strata` 引擎**（`include/strata/**`、`src/program/generate.cpp`、`src/prefill/**`）；§1.3/1.4/1.5/1.14 = **`poly`**（`src/poly/**`）⇒ **两套实现被串成一条因果链** | 若两台引擎的实现确实兼容 ⇒ 可合并叙述（**本席判：不兼容**，`strata` 借缓存槽做 ring、`poly` 用独立 `stage_`） |
| 未支持的推论 | 读 `engine.hpp:618` 注释 | §1.5「双缓冲 ⇒ 层间流水」：注释原文只有 `h_stage_[2] = {}; ///< pinned; filled in turn` ⇒ **推不出层间流水** ⇒ 标 **D（不采信）** | —— |
| 数字缺证据级 | 读 `docs/*` | §1.11(+125%) / §1.12(−5.3%/−4.1%) / §1.13(−82.7%) 均**作者自报 ⇒ C 级**（守 `KP-STRATA-0043` 只许排序、不许配比） | —— |
| **全历史可得性** | `git log --oneline \| wc -l` = **1**；`git log --oneline origin/main \| wc -l` = **25**（**全部 2026-10-15**） | **本 clone 是浅克隆** ⇒ **全历史不可得**；要全历史须一次 `git fetch --unshallow`（**写操作，未做，待授权**） | 若 `git log` 能列出更早提交 ⇒ 本条错 |

## 红臂（我怎么试图让它变红）

- **反向核 URL**：`Niko121` 在本机**所有**己方件里零命中 ⇒ 判幻觉。
- **反向核"有无 prefill 专门文档"**：外派说"有大量专门文档" ⇒ 我**逐件 `test -f`** 了它引的 9 件（全在），**但也确认仓内没有"prefill 路线总账"**（那是本机 `储备-Strata加速原理与优势来源-20260929.md` 做的事）⇒ 两者**不矛盾**，外派那句表述**基本成立**。

## 关键决定

**不重写外派稿**（保持"只增不改"的可追溯性）⇒ 只在**头部加复核横幅**、**文末加两节**。**结论归属本席**：外派只产原始扫描，**编造判定与"混引"警示由本席下**。

## 关键结论（一句话）

**作者"专门的 prefill 文档"存在（`docs/PREFILL_HEAD_ALIAS.md` / `KV_PREFETCH.md` / `BATCHED_DMA.md` / `PROMPT_CACHE_TAIL.md` / `DETAILS.md:332-342` / `HOW_IT_WORKS.md`），但都是"离散的 opt-in 开关说明页"；贯穿性的 prefill 因果链只在代码注释里（`full batch starts the copy of L.ahead … on a stream of its own, so the upload runs under the layer's attention`）；而"按时间的历史路线"在仓内根本没有。**

## 未验 / 我不知道的

1. **全历史**（浅克隆；`fetch --unshallow` 未授权）⇒ §2 的路线表只覆盖**可见 25 条**（其中**无一条**是 prefill/重叠/搬运改动）。
2. 外派引的 **`strata` 引擎**那些行号（`generate.cpp`、`expert_source.hpp` 等）**我只抽核了 3 条**，**其余未逐条核**。
3. 本机储备件里的**作者读数未复核**（原始出自 `E:\strata\…`，**红线未碰**）。

## 阻塞

**需队长裁决**：是否授权在本实验区 clone 里做一次 **`git fetch --unshallow`**（写操作）⇒ 授权则本席可自行补全"全历史路线表"（**不需要 MCP**）。

---

## 追加一节（2026-10-17，**只增不改**）：`fetch --unshallow` **已执行成功** + **一条推翻上一版结论的更正**

- **授权**：队长批（限定版：只在我们自己的 clone、只这一条命令、不 checkout/pull/reset/rebase/merge、工作树不动）。
- **执行**：`git -c http.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl fetch --unshallow origin`
  ⚠️ **环境坑**：**先失败两次**（`fatal: … schannel: failed to receive handshake, SSL/TLS connection failed`），而**同一代理 curl 正常**（`https://github.com/` 200 / 4.3 s）⇒ **加 `-c http.sslBackend=openssl` 后成功**。**记一笔**（后来人会踩）。
- **结果**：**全历史 = 2,448 条**，范围 **2026-09-24 → 2026-10-15**。
- **★ 更正（推翻上一版的结论）**：上一版说"**作者没有按时间的 prefill 路线、可见 25 条里无一条是 prefill 改动**" —— **后半句对那 25 条仍成立，但整体结论【作废】**：
  **作者【有】在提交信息里谈 prefill 路线，而且逐机制、带实测**；最硬的一条是
  **`9bf62c2e`（2026-10-11）「poly: copy ahead from 1,024 tokens, read a short prompt call by call」**
  ⇒ **这句话逐字就是我们从代码里读出的那道门**（阈值 1024 + 短 prompt 逐调用）⇒ **门判读有了作者自己的书面确认**。
- **补全物**：`Strata作者优化路线-按prefill-20261016.md` **附录 C**（按时间路线表 ~45 条 + 五类杠杆分布 + 对我们最直接的三条）。
- **仍未做**：**`origin/main` 与 HEAD 的逐文件 diff**；**2,448 条正文未逐条读**（只按标题关键词筛）⇒ 可能漏掉不喊关键词的 prefill 改动。

### 追加（红线条件的**自证**）：**工作树未被 `fetch` 动过**

队长授权的限定条件里有一条「**工作树不许被动**（我们有未提交的施工改动）」⇒ 本席收尾时**实测核对**：

| 检查 | 命令 | 读数 | 判定 |
|---|---|---|---|
| 工作树状态 | `git status --short` | **只有 ` M`（已改未暂存）与 `??`（未跟踪）两类** —— `M`：`CMakeLists.txt` · `include/strata/poly/kernels.hpp` · `src/poly/{cpu_experts.cpp,cpu_experts.hpp,engine.cpp,engine.hpp,kernels.cu,main.cpp,qwen35.cpp}`；`??`：`src/poly/{nemotronhmoe.cpp,nemotronhmoe.hpp,nemotron_moe_test.cu}` | ✅ **全部是我们的施工改动**（与队长说的"未提交的施工改动"一致）· **无 `D`（删除）· 无 staged 变更** |
| HEAD 未动 | `git log --oneline -1 HEAD` / `git rev-parse --abbrev-ref HEAD` | **`eb5ef118` / `main`（= fetch 之前）** | ✅ 未 checkout、未切分支 |
| 历史确已补全 | `git log --oneline \| wc -l` | **2448**（fetch 前可见 **1**） | ✅ `--unshallow` 生效 |

**⇒ 结论**：**`git fetch --unshallow` 只更新了对象与远端跟踪引用，工作树与 HEAD 均未被动**（`fetch` 语义上不碰工作树；上表为实测自证）。**限定条件全部满足。**
