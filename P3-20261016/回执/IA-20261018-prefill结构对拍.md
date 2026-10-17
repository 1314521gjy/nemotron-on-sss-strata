# 回执 · 情报E 上游 prefill 计算结构对拍（task-14）— 2026-10-18

## 1. 任务
`task-14`：零 GPU 只读，对拍上游 prefill 的段结构 / CUDA graph 用不用 / `src\prefill\` 11 个文件用途与"我们是否已用" / KB 对拍 / 在 compute 2.941 s + kernel top-10 前提下给"能压我们 compute 的上游结构"排序（收益/工作量 + A/D 标注）。

## 2. 状态
**达成**（4 项必做全给 + 结论排序 + 3 个"不可搬"坑钉死）。零 GPU、零编译、零进程、未改码；只写本回执与详细件。

## 3. 结论（一句话）
**上游 prefill 不把层拆成并行段、也不用 CUDA graph**（图只在 decode；`--gpu-stages` 是"一个 token 的三段测量、打印后退出"）；**最值得搬的一条是"GPU 侧 routing"**（对得上我们 40 个层边界气泡 0.400 s：我们每 MoE 层 D2H 整张 13.9 MB 分数 + 24 线程上做 sigmoid/bias/top-22），其后是 mamba2_scan 分块并行（0.845 s = compute 28.7%）、MoE epilogue 融合（quantize_mmq+relu2 = 0.427 s）；`wmma_gemm`（AMD）、NVFP4（Blackwell）、K-quant fused（仅 HIP）**不可搬**。

## 4. 证据（关键读数 + 口径 + 级别）

| # | 结论 | 证据 | 级 |
|---|---|---|---|
| E1 | **上游一层 prefill 段序列** = `embed+steps → hc read → [GDN: gdn/gdn conv+gates/gdn recurrence/gdn out | QSA: qsa proj/indexer/select/attn+kv stage] → router+shared → (fused: GPU grouping→fused experts | MMQ: host grouping→gather→wait copy→dequant→gemm gate/up→gemm down→combine) → 层尾（gr_write/norm）` | `upstream-0.1.44.1\src\prefill\prefill.cpp:2004-2010`（PfPhase）+ 各段 mark 行（详细件 §1.2 表） | A |
| E2 | 段**串行**，并行只来自两流（`m.cs` 计算 + `m.copy` 拷贝，`prefill.cpp:1093/1119`）、CPU 池/peer 重叠、下一层专家预取、async PLE、延后 shared 专家 | 同上 + `:3019-3022, :3055-3056, :4014-4059` | A |
| E3 | **`--gpu-stages` 不是 prefill**：`session_capture(split=true)` 捕 preA/preB/preP[0..4]，打印头是 "one token over 48 layers"，**prints and exits** | `C:\sp3\help.txt:162-164`；`generate.cpp:5970, :6149-6181`；`core/session.cpp:212-282` | A |
| E4 | **上游 prefill 无 CUDA graph**：`src/prefill/` 全目录 `cudaStreamBeginCapture` = 0 命中；捕获点全在 `core/session.cpp:241`、`core/mtp.cpp:1250-1322`、`core/verify.cpp:1891-2802`（decode/token 侧） | 全树 grep | A |
| E5 | 上游图的作用 = 省 host 逐 kernel enqueue 间隙（原文："those events … includes every gap where the GPU sat idle waiting for the host to enqueue the next kernel … proportional to the KERNEL COUNT"） | `generate.cpp:6142-6147` | A |
| E6 | 图量化数字：**1.51×**（Strata decode，KB `KP-STRATA-0044`，源 `docs\储备-Strata加速原理与优势来源-20260929.md:461-467`）；**2–4 µs/layer @ RTX 5090**（poly `engine.cpp:3141-3142` 注释）；**~2500 节点@~5 µs**（上游 `docs\INTEL.md:491`）；**上游 prefill 图 A/B = 无数字（不存在）** | 同列 | B/C/A(缺失) |
| E7 | **我们 poly prefill 也无图**：`replay = opt_.graphs && n <= opt_.window && (windowed_ || n == 1) && !opt_.dump`（`engine.cpp:3007`）；`--no-graphs` help 是"launch a **token's** kernels one by one" | `PolyStrata\src\poly\engine.cpp:3007, :2953-2972, :3143-3166`；`main.cpp:767,848`；**engine.cpp sha256 `347C8C13…`（2026-10-17 23:01）** | A |
| E8 | `src\prefill\` 11 文件：**poly 只编了 `ggml_cuda_host.cu`**（`CMakeLists.txt:1803-1804`）；另外 10 个（moe_fused/moe_fused_iq/moe_mmq/mmq_kq_lib/mmq_kq_loader/wmma_gemm/gemm/mmq_nvfp4_w4a8/w4a4/mmq_resident_sort/gdn_rec_parity）**全未链接** | 详细件 §3.2 逐文件表；`CMakeLists.txt:612-631,1427-1486,1775-1831` | A |
| E9 | 不可搬三坑：`wmma_gemm.cu`=AMD RDNA3 HIP intrinsics；`mmq_nvfp4_*`=Blackwell sm_120 FP4；`STRATA_PF_FUSED_KQ`（K-quant 融合）**只在 `#if defined(__HIPCC__)`** | `wmma_gemm.cu:1-26`；`mmq_nvfp4_w4a8.cu:1-24`；`moe_fused_iq.cu:1067-1089` | A |
| E10 | **"还没用但重要"**：① GPU routing（上游 `src\prefill\kernels.cu:2016-2023 route()` + `:1491-1528 route_kernel`；**我们链接面里已有 `src/kernels/cuda/router_top10.cu` 但 poly 零调用**，且 `native_router_topk_supported` 只支持 (512,10)/(256,8)，`src/kernels/cuda/native_router.cu:236`）；② `gemm.cu` 的 dense 大 tile GEMM（cublas/bf16）；③ `moe_mmq.cu` 的组 gather（16 专家/launch） | 同列 | A（可用性）/D（收益） |
| E11 | 我们 host routing 现状（气泡的结构性候选）：每 MoE 层 `cudaMemcpyAsync(h_rl_, rl_, T×512×4)` + `cudaEventSynchronize` + 24 线程 `route()`（sigmoid+bias+top-22+归一）+ host bounds 表 | `PolyStrata\src\poly\engine.cpp:2436, :2392, :2505-2509, :2669`；`nemotronhmoe.cpp:360-412` | A |
| E12 | 上游"fused"的量化数字（**标机器**）：Q2_0 prompt 4K 1294→1570 / 32K 2170→2653 tok/s（**RTX 5070 12GB**，`DETAILS.md:22-23`）；我们老师机 PF_FUSED 仅 **+5%**（`回执\长prompt-3.8对拍-20261017.md:24`）；`mmq_resident_sort` 数字是 **4×P100**（`mmq_resident_sort.hpp:24-25`）⇒ **一律不许当我们的读数** | 同列 | A/B（非本机同模型） |
| E13 | KB：`CUDA graph`→KP-UNI-0015/0251/0040 + **KP-STRATA-0044（图 1.51×，跨引擎结论不可搬）**；`图捕获`→KP-NINFER-0015/KP-UNI-0029/KP-STRATA-0044；`PF_FUSED`→KP-STRATA-0008/0018；`prefill auto`→KP-MOEDEP-0039/KP-STRATA-0039/**KP-STRATA-0043（禁说"某机制占 X%"）**；`kernel 融合`→KP-UNI-0243/**KP-STRATA-0025（上游判死清单含"节点融合无空间"）**/KP-MOEDEP-0027；**`moe_fused`/`wmma`/`微批` = 无命中** | 详细件 §4 | A |

**能变红的判据**：① `src/prefill/` grep 到 `cudaStreamBeginCapture` ⇒ E4 错；② `--gpu-stages` 打印头不是 "one token" ⇒ E3 错；③ `native_router_topk_supported` 允许 (512,22) ⇒ #1 工作量从"中"降"低"；④ `CMakeLists` 里 `strata-poly` 链了 `strata_prefill`/`strata_mmq` ⇒ E8 错。

## 5. 未验（按对决策的影响排序）
- **U1（最关键）**：0.400 s 气泡的**层标签**。队长说"一 mamba 层一个"，但 **mamba 层数 = MoE 层数 = 40**，计数分不开 ⇒ 需 nsys 层标记或 `moe_rows` host 段打点。气泡在 MoE 层 ⇒ #1 直击；在 mamba 层 ⇒ #1 不成立，需另查。
- **U2**：`mul_mat_q type12` 的 **MoE vs dense 占比**（决定 gemm.cu 大 tile GEMM 的收益量级）。
- **U3**：我们 host `route()` 的**实测耗时**（本件按 3.5M sigmoid/层估，未打点；属改码，未做）。
- **U4**：上游所有"卡注释值"（4080S 1.41×、5090 阈值、5070 +16~22%、R9700、4×P100）**都不是我们模型/我们的读数**，一律 D 级。
- **U5**：`src/prefill/kernels.cu` 2,225 行只读了段/route/combine 相关部分，未逐行通读。
- **U6**：`GraphRegistry`（`graph.cpp`）"无调用者"是 grep 结论（全树只在 graph.cpp 出现），若模板/宏间接引用未覆盖需复核。

## 6. 落盘（绝对路径）
- 详细件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-E-prefill计算结构对拍-20261018.md`
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-prefill结构对拍.md`
