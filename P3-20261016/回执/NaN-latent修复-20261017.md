# 回执 · **NaN 根因修复：`moe_rows()` 接上 latent 几何** — 2026-10-17

- **任务**：修 `pb ≥ 32` 的 NaN。根因由 scheduler 定死（`回执\moe-vs-moe_rows宽度diff-20261017.md`，A 级三方互证），队长批准施工。
- **状态**：**达成（修复验证通过；性能异常已如实上报）**。
- **结论一句**：**`moe_rows()` 从没用过专家的 latent 几何（`L.exp_in/exp_out=1024`），全程按模型宽 `n_embd_=4096` 乘 ⇒ up 每行读 4× 跨过相邻 slot、down 按 4096 行算、读过 arena 边界 ⇒ 0xFF ⇒ NaN。** 接上 `in_w/out_w` + `latent_down` 投影 + `moe_sum→moe_out_` 后：**pb=32 不再 NaN 且与 pb=31 逐位同**；开关关仍 NaN（负控）。

## 1. 改动（一次一变量 = "latent 几何"）

`src/poly/engine.cpp` · `Engine::moe_rows()`：

1. **入口**（`router_scores` 后）新增：`in_w = L.exp_in>0?L.exp_in:n_embd_` · `out_w = L.exp_out>0?L.exp_out:n_embd_` · `in = xn_`；**latent 则 `lin(L.latent_down, xn_, moe_in_, n, xq_)` + `in = moe_in_`**；latent 时 **`cudaMemsetAsync(moe_out_, 0, n*out_w)`**（`moe_sum` 是累加）。
2. **量化** `Mmq::quantize(in, d_tok, …, in_w, in_w, rg)`（原 `xn_, n_embd_, n_embd_`）。
3. **cached up** `mul(slot, gt, ff2, **in_w**, …)`；**cached down** `mul(slot+gu2, dt, **out_w**, ff, …, dx_, **out_w**)`。
4. **staged up/down** 同 (§3 的两处 `in_w`/`out_w`，含 `dx_` 的 `ld_dst`)。
5. **CPU** `cpu_experts(…, **in_w**, …)`；**h_x_ 拷贝** `n*in_w`。
6. **输出** `moe_sum(dx_, d_w, latent ? moe_out_ : fo_, **out_w**, …)`。
7. **门** `Mmq::fits(gu_type, gu_w, **L.exp_in**)` / `fits(d_type, **L.exp_out**, n_ff_exp_)`（本靶两者都 %128==0 ⇒ 门行为不变）。
- **开关**：**`STRATA_MOE_ROWS_LATENT=0` 回到旧行为**（`in_w=out_w=n_embd_`、不投影、`moe_sum→fo_`）。默认开。
- **另**：上一轮 `runs_` 的 **class 切段**（`STRATA_CARD_SPLIT_CLASS`）**保留**；scheduler 复核：**对正确性是 no-op（`cb` 的 `at` 跨类累积 ⇒ 旧代码即使混类也对）**，**它不是 NaN 的成因**，仅作潜伏 bug 的加固。
- **未碰** §7 四处。

## 2. 验收（A 级实测）

| 门 | 臂 | 结果 | 判 |
|---|---|---|---|
| **V3 负控** | `STRATA_MOE_ROWS_LATENT=0` + pb=32 | `0,0,…` **NaN** | **✓ 路径确被走到、判据不空转** |
| **V1 正控** | `=1` + pb=32 | **Y** `1531,48365,2655,1584,9632,1044,1809,2127`（有限） | **✓** |
| **V1 强形式** | pb=32（批，`moe_rows`） vs **pb=31（非批，`moe()`）** | **逐位相同 = Y** | **✓ 批/非批等价** |
| **安全** | pb=8 / pb=31（新旧） | `X` / `Y` **逐位不变** | **✓** |
| **V2** | pb=1024 + 700-token | **无 NaN/Inf**、文本 **" The abstractions are useful because they let"**（通顺） | **✓** |
| **大批** | pb=64 / 128 | 无 NaN（Y） | **✓** |

**★ 关于"pb=32 == pb=8"**：**不成立，且不是失败** —— pb=8=`X`、pb≥10=`Y` 是**已认定的 CPU/GPU 分工漂移**（`KP-UNI-0244`：CPU 走 Q8_K、GPU 走 Q8_1 有位差；两串都通顺英文）。**正确的判据是"批路径 == 非批路径在同 pb 组内"**（pb=32 vs pb=31 ⇒ 逐位同）✓。

## 3. ★ 正确路径性能（pb=1024 canonical，≥3 跑 + 预热丢弃，max-new 45）

| 臂 | prefill ×3 | **中位** | MAD | 极差 | decode 中位 |
|---|---|---|---|---|---|
| **SMEM 关** | 121.1 / 123.5 / 119.7 | **121.1** | 1.2% | 3.8 | **18.5** |
| **SMEM 开**（`STRATA_M2_SCAN_SMEM=1`） | 146.8 / 145.6 / 145.5 | **145.6** | 0.07% | 1.3 | **19.1** |

- 加载 **7388 of 20480 in VRAM（21.3 GiB）**，无静默退化。
- **SMEM 在正确路径上仅 +20%**（不是旧 +109%）。

### ★★ 异常（已报队长，未下结论）
- **旧坏路径**（pre-latent，sha 2719c3df）pb=1024：**prefill ~348 · decode ~55**；**新正确路径**：**~121 · ~19** ⇒ **修复后反而慢 ~2.9×**。
- **decode（n=1）走 `moe()`、不受 latent 修复影响，却也掉 2.9×** ⇒ **掉速很可能不来自修复本身**：旧"快"可能是坏路径的假象（4096 跨读/NaN 状态/预取整 arena），或驻留状态变化。**正在用旧 exe 同参复核 3 跑。**
- **⇒ 在正确路径上，"12.8×"不成立。当前正确路径最好 = 145.6 tok/s（pb=1024 + SMEM）。**

## 4. 两条"今天不改、记录在案"

1. **`moe_card()` 仍用 `n_embd_`**（`:2142/2144/2149/2150/2151`）—— 今天不可达（`if (!L.gated) return false;`，本模型非门控）；**将来 latent+gated 会错**。
2. **`h_x_` 步长两条路不一致**（`moe()` 按 `out_w` 取、填 `n*in_w`；`moe_rows()` 全按 `in_w`）—— **仅当 `in_w == out_w`（本靶 1024/1024）时正确**。

## 5. 落盘（绝对路径）
- 本件：`…\P3-20261016\回执\NaN-latent修复-20261017.md`
- 代码：`PolyStrata\src\poly\engine.cpp` · `src\poly\kernels.cu`
- exe（修后）：`PolyStrata\build-p3\strata-poly.exe` = **`5a19838cec8a674e10e7ad708d680417f1cf4a4f0213ec04496d50d946aa4229`** · `C:\sp3\sp_latent.exe`
- 旧 exe（对照）：`C:\sp3\sp_new.exe` = `2719c3df…`（pre-latent）· `sp_old.exe` = `978d64bc…`（原始）
- 日志：`C:\sp3\{lat_pb32_on, lat_pb32_off, lat_pb8, lat_pb31, lat_pb64, lat_pb128, lat_pb1024, pf_off_1..3, pf_on_1..3, pf_warm_off, oldchk_1..3}.log`
- 根因件：`…\回执\moe-vs-moe_rows宽度diff-20261017.md`（scheduler）
- 解码器：`C:\sp3\gguf_vocab.py`
