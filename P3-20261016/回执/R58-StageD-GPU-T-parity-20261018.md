# R58 · Stage D 第一道门：GPU-T vs GPU-row 位精确 parity — **PASS**

- **席位**：builder3 · **1 跑（GPU，~0.3 s）** · 跑前断言：无引擎进程、GPU 868 MiB（<1.5 GB）、RAM 84.3 GiB · 跑完 GPU 867 MiB（已让出）· 未改 tree、未 spawn 子代理。
- **判决：PASS** —— T 布局（neuron-major）IQ4_NL down 内核与引擎现用的 row-major 小批 MMVQ 内核**逐位相同**（3 用例 × 2 真实专家 × 1024 行，全 0 差），负控变红（1296/2048 行），`inverse(T)==GGUF` 0 diff ⇒ **Stage D 的"GPU 侧可位精确"前提成立，可进第二轮（MMQ/prefill 那一路）**。

## 1. 跑法与证据（绝对路径）

- 证据：`C:\sp3\evidence\staged-gpu-parity-20261018.txt`（含全部 sha 与逐行读数，exit=0）
- 工具 exe：`C:\sp3\staged-gpu-parity.exe` sha256 `F0C80AD7AFB54962679341865DB8E0561AB190EB836C05BB6C976981B6BB6D61`
- host：`C:\sp3\staged-gpu-parity.cpp` sha256 `BABA7F54A498FAEF47866173C6A4B744234CAC004B53596138469E937A96E5FF`
- T 内核：`C:\sp3\staged_gpu_t.cu` sha256 `706FB84685BF75B798625BDC64D9B8A7B35A221FEDDEA471E88E7646AF848EA1`
- 参考库（引擎同款）：`PolyStrata\build-p3\strata_kernels.lib` sha256 `CBE401E2812B835BF265D6BBB9C48C894E98E15AEFA2FE81D494244EF39CB1A5`
- 设备：RTX 4080 SUPER（sm_89），驱动 13.4；q8_1 行字节 3024（=84×36）自证 ✓

## 2. 读数（不可放宽的判据逐条）

| 专家 | inverse(T)==GGUF | dense（全非零） | 真实稀疏 7.94% | 50% 零 | 负控（错序 warp 树） |
|---|---|---|---|---|---|
| layer 1 expert 0 | **0 diff** | **1024 行 0 差** | **1024 行 0 差** | **1024 行 0 差** | **630/1024 行变红** |
| layer 87 expert 511 | **0 diff** | **1024 行 0 差** | **1024 行 0 差** | **1024 行 0 差** | **666/1024 行变红** |
| 合计 | 2/2 | 2048 行 0 差 | 2048 行 0 差 | 2048 行 0 差 | **1296/2048 变红（比较器不盲）** |

## 3. 关键结构发现（本轮读码，决定实现正确性）

GPU row 内核（`native_small_mmvq_kernel<IQ4NLBlock,4,false>`；n_in=2688 ⇒ 84 块 ≥ 64 阈值 ⇒ ROWS=1、每输出行一个 block、block=(32,4)=128 线程）：
1. **一个 32 值块的 int 点积拆给两个线程**：`kbx=tid/2`、`kqs=2*(tid%2)`；偶数线程点值集 {0..7,16..23}，奇数线程点 {8..15,24..31}（int32 求和，顺序无关）。
2. **两半分别** `tmp += d * float(sumi_half)`（`d = __half2float(w.d) * __low2float(x.ds)`），**不是** CPU 的"整块 32 值和再乘 d" ⇒ 位精确的 T 内核必须照这个拆法。
3. 块序：每线程 `kbx` 升序、步长 `2*WARPS*WARP/4 = 64`；跨 warp：`partial[0..2]` 按序相加 → `warp_sum`（`__shfl_xor` 偏移 16,8,4,2,1）→ `tx==0` 写 `y[row]`。
4. T 内核（`staged_gpu_t.cu`）逐条照抄 1–3，只把权重 nibble 换成从 T 块 gather（神经元 n 的 512 B、字节 row/2、偶行低/奇行高 nibble；scale = `scales[kbx*2048 + row*2]`）⇒ 逐位相同。

## 4. 下一步（等 lead 批第二轮）

- **第二轮 = MMQ（prefill）down 那一路**的 GPU-T vs GPU-row parity（同样的方法：真实专家 + 真实激活布局；MMQ 的 tile/累加结构不同，需另读 `moe_fused_iq.cu`/`mmq` 相关实现）。
- 两路都 PASS 后，Stage D 的 D1–D9 改动才可动主路径（槽/staging/fill 数据源 + 模型文件策略 D6a/D6b）。
- **本回执不改默认值**：树仍默认关（`STRATA_RELU2_DOWN=0` 原路径）；本门只证明"GPU 侧位精确可行"。

## 4b. D2a（prefill 的 GPU T→row 反变换）— **PASS（三门全绿）**

> 背景：MMQ（prefill）不能原生吃 T（一个行主序 16 B 块在 T 里散在 32 个神经元 ⇒ 32 次跨 cache-line gather，~8× 流量）。
> 方案改为：MMQ **一行不改**，prefill staging 时用 GPU 把 T 块**纯字节置换回行主序**再喂它 ⇒ 位精确由"输入字节相同"保证。

- 工具：`C:\sp3\staged-d2a-parity.exe` sha256 `4062504FC688DC9438F5607A5C4DBB9BBCA7B31EA9172DCFFB7D61FF1C73E911`（源 `C:\sp3\staged-d2a-parity.cu` sha `68EB76D3…`；nvcc sm_89）
- 证据：`C:\sp3\evidence\staged-d2a-byte-parity-20261018.txt`（run 2026-10-18 13:21:53，exit=0）
- 读数：**4/4 真实专家**（layer1-e0/e511、layer87-e0/e511，跨两分片）× **1,548,288 B 逐字节 0 差**；**负控**（翻转输出第 1000 字节）⇒ `diff=1`（恰好 1）⇒ 比较器不盲。
- 跑前断言：无引擎进程、GPU 818 MiB；跑后 GPU 817 MiB（**已让出**）。
- **Stage D 三门全绿**：① decode mmvq 路位精确（§2）② prefill 反变换字节精确（本节）③ RAM 闭合（reviewer §9：75.2 GB / pin 67.06 GiB≤96，本文件 §0c 并入）⇒ **可进 D1（decode mmvq T 核产品化）**。

## 5. 落盘

- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R58-StageD-GPU-T-parity-20261018.md`
- 证据：`C:\sp3\evidence\staged-gpu-parity-20261018.txt`
- 源码/exe：`C:\sp3\staged_gpu_t.cu` / `C:\sp3\staged-gpu-parity.cpp` / `C:\sp3\staged-gpu-parity.exe`
- 立项与前置：`…\回执\R57b-StageD立项-20261018.md`、`…\计划-StageD-GPU侧T布局-20261018.md`、`…\回执\R57f-ReLU2收口-20261018.md`
