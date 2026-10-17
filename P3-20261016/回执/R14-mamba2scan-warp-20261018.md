# 回执 · R14：mamba2_scan d_state 切 warp（A#1）— 2026-10-18

## 任务
lead 批 A#1：`d_state` 切给 warp（32 lane 分摊 128 states、按 st 序归约、block 256/8 warp、grid ~1,024）；旗标 `STRATA_M2_SCAN_WARP` 默认 0、**关掉逐字节走旧路**；先 micro parity 再端到端；门=**逐位**（ids 45/45 + pos0 Δp=0.000e+00）；1 编译 + 5 跑（micro ×1 + 端到端 ×3 + top-probs ×1），峰值 ≤31,700；判据含**墙中位 < 3.456 s（≥−3%）**，收益 <3% ⇒ 判"没做对"，不许硬解释。

## 状态
**未达成（性能门严重不过，按"任何一条不过 ⇒ 停手、原样报"执行）。**
- ✅ **micro parity 逐位通过**：`M2_SCAN_MICRO: rows=64 y_equal=1 s_equal=1 first_diff_y=524288/524288 first_diff_s=1048576/1048576`（参考 `mamba2_scan_kernel<false>` vs 新 `mamba2_scan_warp_kernel`，bitwise）。
- ❌ **端到端 prefill 644.8 tok/s（wall 10.53 s）**，对比基线 1906.7 tok/s（wall 3.5622 s）= **−66.2%**；墙门（<3.456 s）不过。
- **停手**：micro 那跑本身就是端到端，已足以判"没做对"；**未再跑端到端 ×3 / top-probs ×1**（预算保留）。

## 第一行结论
**prefill 中位 644.8 tok/s（−66.2%）；mamba2_scan 未单测（端到端 +6.97 s，D 级推断 0.872→~7.8 s）；micro parity=1（逐位全等）；pos0 Δp=未测（停手）。**

## 证据（A）
### 跑次（1 跑，delivery 配置 + WARP=1 + MICRO=1）
| 项 | 读数 |
|---|---|
| exe sha | `ea3c97e71507c94ce7f0f27e0dfeab190e0a5004264cf4a26f03497480dd4447` |
| prefill / wall | **644.8 tok/s / 10.53 s**（基线 1906.7 / 3.5622） |
| decode | 11.09 tok/s |
| 峰值 VRAM | **31,033 MiB ≤31,700**（余 667） |
| 驻留 / DMA / ledger | 5,721 / 29,132 / `fill 5,721·16,894.83 MB; prefill 29,132·42.01 GB; swap 2,014·5.81 GB` —— **三列与 R13 相同** |
| ids | **45/45 与基线逐位相同**（逐位实现下符合预期） |
| micro parity | **y 全等 524,288/524,288；s 全等 1,048,576/1,048,576** |
- 说明：ids 是拿 R14 日志与 `C:\sp3\pair_D_pb6912.log` 基线逐位比；**pos0 Δp 没测**（top-probs 跑被停手省掉）。

### 为什么"逐位过、性能崩"（D，结构分析）
1. **保序归约把并行度换成了串行链**：为逐位复现旧 kernel 的 `sumf += state*sC`（升序），每个 warp 每 token 要跑 **128 步 `__shfl_sync` + 128 步 fma 的串行链**（旧 kernel 等价链在 8,192 个线程里各自独立、无 shuffle）；shuffle 延迟（~10–20 cycle）叠加后，每 token 链长从 ~128 fma 变成 ~128×(shuffle+fma) ⇒ **单链慢 2–4×**。
2. **每 token 两次 `__syncthreads()`**：block=1,024 线程（32 warp），每 token 首尾各一次全 block barrier ⇒ 6,791 token × 40 层 × 2 次，barrier 成本随 warp 数上升。
3. **sB/sC 加载变 2×**：head_dim 64 / 每 block 32 warp ⇒ 2 blocks/head，每个 block 都各自加载 128 个 B + 128 个 C（旧 kernel 1 block/head 只加载一次）。
4. 净效果：**用"逐位"换掉了"A#1 的全部收益"**。这与判据台 `正确性判据-mamba2scan改动-20261017.md` §2.1 的既有判断一致：**d_state 维切线程属于"位级必变、允许变但有界"那一类**；我们这次用保序归约硬做成"位级不变"，代价就是慢。

## 结论与请求（不自行改门）
- **A#1（保序逐位版）判"没做对"**：micro 逐位 ✅ 但端到端 −66% ⇒ 按 lead 判据不通过；**我没有把门改成 2e-3，也没有继续跑**。
- 若要继续这条线，**必须 lead 改判据**：按既有判据台 §2 的三层门（**Q1 ids 逐位硬 · Q2 top-probs 末位±1 · Q3 逐层 rel ≤1e-5，>1% ⇒ FAIL**），实现上放弃保序归约、改用 **warp 内 tree/分组归约 + 分块 scan**（预计 3–5×，位级会变但在 §2 容差内）。
- 备选：**放弃 A**；或先做"state 入寄存器"（逐位、无 shuffle，但单做不增 occupancy ⇒ 预计 <5%，不值得单独一跑）。

## 记账
- **改前/改后（只有两段，旗标默认 0）**：新 kernel `mamba2_scan_warp_kernel` 插在 `src/poly/kernels.cu:2612-2695`（旧 kernel 之后、conv kernel 之前）；launch 侧改动在 `:2782-2874`（旧 wrapper `mamba2_scan()` 内：flag 自证行、micro parity 块、warp 启动 + `d_state>128` 时大声回退）。**旧路径（SMEM/global 两个分支）源码一字未动**；flag 未设时只多两次 `getenv`。
- **构建**：`C:\sp3\build_cpstream3.cmd`（cmake --build build-p3 --target strata-poly -j 24）；**第一次失败**（`kernels.cu(2819): error: namespace "std" has no member "vector"`）→ 改成 `new/delete` 后 **EXIT=0，64.3 s**；日志 `C:\sp3\build_m2warp2.log`（失败版 `build_m2warp.log` 也留档）。改 `.cu` 不涉 hpp，未删 obj。
- **日志**：`C:\sp3\R14-MICRO-20261018.log`；脚本 `C:\sp3\run_R14_micro.cmd`；exe：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\build-p3\strata-poly.exe`（= `C:\sp3\sp_ring2.exe`，`ea3c97e7…`）。
- 峰值采样：0.5 s 轮询 `nvidia-smi memory.used` 取最大；空闲 = 32,760 − 峰值。

## 未验 / 边界
1. 新 kernel 的**单 kernel 时间没测**（nsys 未跑）；"+6.97 s 全归 mamba2_scan"是 D 级推断（0.872→~7.8 s），实际可能含 barrier/带宽等。
2. micro parity 只比了 **64 token 前缀**、且参考是 `mamba2_scan_kernel<false>`（非 SMEM 版）；与生产用的 `<true>`（SMEM）是否也逐位，**未验**（按判据台 §0，① 是"可证伪的强预期"，但本轮没测）。
3. flag 关掉时"逐字节走旧路"是**代码保证**（旧分支源码未动、flag 未设不进入新代码），但**没有跑一跑验证**（停手省了）。
4. `d_state>128` 的回退分支没被触发过（本模型 128）。
5. 零编译承诺未违（本轮按批编译 1 次）；**GPU 只跑了 1 次**（micro），峰值 31,033 ≤31,700。

## 落盘（绝对路径）
- 本回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\R14-mamba2scan-warp-20261018.md`
- 日志/脚本：`C:\sp3\R14-MICRO-20261018.log` · `C:\sp3\run_R14_micro.cmd` · `C:\sp3\build_m2warp2.log`（成功）· `C:\sp3\build_m2warp.log`（首次失败留档）
- exe：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\PolyStrata\build-p3\strata-poly.exe`（sha `ea3c97e7…`）= `C:\sp3\sp_ring2.exe`
- 判据台（只读引用）：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\正确性判据-mamba2scan改动-20261017.md` §0/§2.1/§2.2
