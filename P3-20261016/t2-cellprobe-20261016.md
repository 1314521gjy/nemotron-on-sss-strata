# t2 cell-probe：routed ~10× 根因定位（一次构建 · 三跑矩阵，2026-10-16）

- **exe**：`build-p3\strata-poly.exe` sha256 `e744b3b4c8f8eca5d919d490e992f633ab7f68470e708599496b1ff2468319cd`，mtime **12:38:30**（> 源 12:33 ✓）。
- **跑法**：canonical t2（`--tokens 1784,8961,1307,5498,1395 --max-new 1 --prompt-batch 8 --vram-reserve-mib 2500 --cpu-share 0.55 --expert-swaps 64 --tail-skip 0 --expert-profile … --dump-layers --top-probs 5`）+ env `STRATA_DUMP_LATENT=1 STRATA_PROBE_HOST=1 STRATA_DUMP_ROUTE=1`。
- **口径**：同 5-token prompt，比 **pos=0 / 前 3 通道**；参考 = `evalcb-pos0.log`（llama-eval-callback，同源 pin）。
- 三跑 = 同一次构建，仅 env 门控不同：`old`（①旧②旧）/ `cgold`（①旧②新）/ `fix`（①新②新）。

## 1. 判决表（pos0 前3，引擎实测 A 级）

| 节点 | 参考（evalcb） | `old` | `cgold` | **`fix`** | fix vs 参考 |
|---|---|---|---|---|---|
| `moe_in`（node_68, 1024） | -0.1444, 0.2673, 0.2056 | -0.146300, 0.259492, 0.202436 | 同 old | -0.146300, 0.259492, 0.202436 | ~1% ✓ |
| **`moe_routed`（ffn_moe_out-1, 1024）** | **0.0062, 0.0053, 0.0049** | **0.058500, 0.053046, 0.047196**（≈9.4–10×） | **0.058500, 0.053046, 0.047196**（**仍10×**） | **0.006276, 0.005278, 0.004891** | **+1.2% / -0.4% / -0.2% ✓** |
| `moe_up`（node_126, 4096） | 0.0024, 0.0012, 0.0058 | 0.024295, 0.017769, 0.059427（≈10×） | 同 old | 0.002469, 0.001244, 0.005752 | ~3% ✓ |
| `shared`（ffn_shexp-1, 4096） | -0.0081, 0.0010, 0.0005 | -0.003882, 0.000379, 0.000300（≈2×小） | **-0.008080, 0.000975, 0.000501** | **-0.008080, 0.000975, 0.000501** | ~0.4% ✓ |
| `ffn`=l_out-1（4096） | -0.0056, 0.0022, 0.0063 | 0.020413, 0.018149, 0.059726 | 0.016215, 0.018744, 0.059927 | **-0.005611, 0.002218, 0.006252** | **~0.5% ✓** |
| top-1 token | — | 1044 | 5498 | **6993** | 输出族变了 |

## 2. 判据（能变红）

- **主判据** `blk=1 moe_routed` pos0 前3 → `0.0062/0.0053/0.0049 ±2%`：`old` **10×（红）** → `fix` **±2%（绿）**。✔ 能变红。
- **负控①**（只改②不动①）：`cgold` 的 `moe_routed` = `old` **逐位相同**（0.058500/0.053046/0.047196）⇒ **② 不是 routed 的驱动**。✔
- **负控②**（改回 `gate` 键）：`old`/`cgold` 均 10× 重现。✔
- **② 的独立效应**：`shared` 由 `-0.003882,0.000379,0.000300`（2×小）→ `-0.008080,0.000975,0.000501`（对参考 ~0.4%）——**② 只动 shared，不动 routed**（② 在 `moe_shared` 写 `fo_`，routed 在 `moe()` 写 `moe_out_`，互不相干）。✔

## 3. §4-2 旁证（`STRATA_PROBE_HOST`，layer 1，5 token 各一 chunk）

- `n_host/n_cached`（每 token）：token0 **17/5**，token1 **12/10**，token2 **14/8**，token3 **15/7**，token4 **15/7**。
- token0 的 17 个 host 专家 id：`230,313,374,265,503,216,223,340,491,437,501,469,493,504,498,252,245` —— **17 个互不相同**、却全被 `cpu_experts` 塌成 **1 组**（`STRATA_CG_GATE=1` 时）。
- ROUTE dump：`ids 230 313 135 374 37 265` 与参考 `ffn_moe_topk-1`（token0 前3 = 230,313,135）**逐字对上**。
- ⇒ 发作条件（`n_host≥2` 且专家不唯一）**满足**；token0 的 routed 和全程只用 `experts[230].up/down` 却乘 `Σ w_j`。

## 4. 结论（本席下）

1. **~10× 的根因 = ① CPU 专家分组键选错**（`cpu_experts.cpp:193/196` 用 `gate`，非门控恒 `nullptr` ⇒ `n_run=1` ⇒ 全体 host use 用 `experts[by[0]]` 的 up/down）。修成 `gate?gate:up` 后 routed 落回 **±2%**。① 是**必要**（去掉即 10× 重现）且（对 routed）**充分**（② 不触及 `moe_out_`）。
2. **② 是真实但独立的缺陷**（`nemotronhmoe.cpp:423` 复用被 `moe_router` 改写成 1024 宽的 `xq_`）：只影响 **shared**，修后 shared 由 2×小 → 对参考 ~0.4%。**这解释了 verifier 此前「shared 同号只小 2×」的张力**——那 2× 正是此 bug。
3. 修后 **layer-1 latent MoE 全链**（moe_in / moe_routed / moe_up / shared / l_out）**逐段吻合参考**（≤3%），`l_out-1` 前3 从「10×且符号反」→ `-0.005611,0.002218,0.006252` ≈ 参考 `-0.0056,0.0022,0.0063`。
4. **修法要点（写进结论，解释"时对时错"）**：
   - ② **不能**就地改 `xq_` 的宽——其后 `engine.cpp:2634` 的 `moe()` 还要吃 **1024 宽**的 `xq_`；就地改会反过来弄坏 routed。故传 `nullptr` 让 `lin` 自己量化进 `q8_`（尺寸够：`engine.cpp:1629` 按 `round_up(q_max)≥n_embd_` 分配）。
   - `engine.cpp:3107` 那条 **draft 路径**本就先按 4096 量化再调 `moe_shared` ⇒ **那条路本就是对**：同一份 `moe_shared` 在两条路上对/错不同，正是"有时对有时错"的来源。

## 5. 残余 / 未验

- **exe sha 口径**：本次全部用**现役新 exe** `e744b3b4…`；`old` 在它上面**逐位重现**了 `dump_lat.log`（旧 exe `05f8af9b`）的 `moe_routed` ⇒ 顺带证明 `--cpu-share/--expert-swaps` 不扰动 pos0 读数。
- **`--expert-profile` 写失败**：三跑均报 `cannot write …\profile_p3.txt.tmp`，进程 EXIT=1；**不影响读数**（所有 dump 已落盘），但 profile 文件未生成 —— 归属待查（疑 .tmp 写法/路径权限）。
- **未验**：只比 pos0；pos1–4 的 dump 已落盘未逐元素核；88 层全链的收敛（token 从 1044→6993 只是现象）；`--expert-swaps 64` 改变了专家 VRAM 布局（与 `dump_lat` 不同）但 pos0 结果仍逐位同。
- 修复目前是 **env 门控**（`STRATA_CG_GATE` / `STRATA_SH_XQ`）；**t3 固化为正式修复**（去门控）后，这两个 env 与其余临时钩子一并 revert（清单见 `临时钩子revert清单-20261016.md`，我会补上这两项）。
