# R79 · shared expert 独立流重叠 — 判否（decode +1.7% 中位，最差配对 −7.1%）— 2026-10-18

- 席位 builder4；env `STRATA_SHARED_OVERLAP=0|1`（默认 0）；配置：p6792、max-new 128、交付形态、**冻结表**（`p_shipped.txt`→`p_frozen.txt`，`STRATA_PROFILE_AUTO_LEARN=0`，四臂 `profile: loaded 16900` / `saved=0`）；exe `C:\sp3\sp_sharedx.exe` sha `A0C136A987EF…`。
- 日志：`C:\sp3\R79-{O0,O1}-20261018.log`、`C:\sp3\R79b-{O0,O1}-20261018.log`；汇总 `C:\sp3\R79-all-runs.json`；驱动 `C:\sp3\run_shared_overlap.ps1` + `run_shared_overlap_extra.ps1`。

## 三轮迭代（每次都是真 bug，不是调参）

1. **第一版**（仅 `q8_` 疑点修正后）：ids O0 vs O1 first_diff=0、O1 自身不自一致 ⇒ **真 race**；hit 75.4→84.5 是污染旁证。
2. **第二版**（补齐双向事件环：`moe_shared` 开头 `record(ev_st_, st_)`+`wait(sh_, ev_st_)`，`moe_latent` `wait(st_, ev_sh_)`）：O1 **崩溃** `cudaStreamEndCapture: capturing stream has unjoined work` ⇒ capture 期间 `sh_` 上的工作不属于 capture 流。
3. **第三版（最终）**：① `moe_shared` 加 `cudaStreamIsCapturing(st_)` gate（capture 期间走旧路，图与原版逐字节相同）；② 仅 gate 不够 —— decode 的 shared 走 `replayed(L.shared,…)` 图（图内跑 st_ ⇒ 永无 overlap），故在 `engine.cpp` 的 `up()` 加旁路：`shared_overlap_ && m<=8 && !all_cached()` 时直调（host 在 sh_ 上发），否则旧 replay。
   - capture 点核查：全树 `cudaStreamBeginCapture` **只有 `record()`（`engine.cpp:4455`）一处** ⇒ capture 期间必走旧路 ⇒ **录出的图与原版逐字节相同**（ids 门前提成立）。

## 最终 6 跑（3 配对）

| 臂 | prefill | decode | hit% | miss/token | 峰值 MiB |
|---|---|---|---|---|---|
| O0 #1 | 1743.0 | 42.03 | 75.4 | 219.4 | 31,271 |
| O1 #2 | 1753.8 | 39.03 | 75.1 | 221.9 | 31,247 |
| O1 #3 | 1751.6 | 40.71 | 75.1 | 221.9 | 31,248 |
| O0 #4 | 1728.8 | 40.04 | 75.4 | 219.4 | 31,268 |
| O0 #5 | 1732.6 | 37.96 | 75.4 | 219.4 | 31,276 |
| O1 #6 | 1772.7 | 40.15 | 75.1 | 221.9 | 31,249 |

- **配对（按跑序）：−7.1% / +1.7% / +5.8% ⇒ 中位 +1.7%，最差 −7.1%**。
- 预注册规则（3 配对中位 ≥ +1.5% **且** 最差 ≥ −3%）⇒ **最差 −7.1% 不满足 ⇒ 判否**（不进交付；`STRATA_SHARED_OVERLAP` 留树默认 0）。
- ids：**批内 O0 vs O1 逐位同**（R79 四跑 first_diff 全 −1；R79b 两跑 first_diff −1）；两批各验自一致 ✓。hit/miss 差 0.3pp/2.5 来自 swap 的墙钟依赖（ids 已同 ⇒ 非算术被踩），按 lead 裁决不作门。
- 峰值全部 31,247–31,276 ≤31,700 ✓。

## "shared expert 真实串行代价 ≈ 0.4 ms/token"

从本轮 overlap 上界反推：decode ≈25ms/token（40 tok/s），配对中位 +1.7% ≈ **+0.42 ms/token** ⇒ **shared 的串行代价 ≈0.4 ms/token**。
**为什么之前差分（2.08/7.34 ms/token）高估**：那是 `STRATA_SKIP_SHARED` 的**功能 skip**——不写 fo_ ⇒ 残差变 ⇒ 下游路由/命中全变（hit 与 miss 都漂），时间差里混进了 cache/路由变化；overlap 则**保留全部数值**（ids 逐位同），只改时序，所以 0.42ms 才是可信的串行代价。

## 代码处置

`STRATA_SHARED_OVERLAP` **默认 0**；累计 **~+75 行**（engine.hpp 成员/声明 ~16 · engine.cpp lin 分流+wait+alloc+析构+up 旁路 ~51 · nemotronhmoe.cpp 分支+capture gate ~16，含注释；净代码 ~40）。`=0` 逐位等价旧行为（无流/事件/scratch、无 wait、日志不变）。
`lin()` 现在按调用流的 `s` 参数选用**各自** scratch（`q8_sh_/wf_sh_`），不再与 `q8_/wf_` 共享。
