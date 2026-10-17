# 回执 · 长 prompt 杠杆 + 3.8-Flash 本机对拍（★ 已更正基准） — 2026-10-17 下午

- **任务**：任务令（2026-10-17）停止细枝末节，回到与 3.8-Flash 的机制/数据对拍，找超级杠杆。
- **状态**：**达成**（判决数、消融归因、长度×sm 规则、单 chunk 试验、tokens-file 均出数）。
- **结论一句**：**本机（4080 SUPER）3.8-Flash @ 6792 token、用其 `dec-pool2` 配置 = 2805.87 tok/s；我们同 prompt 最好 = 383.6 ⇒ 真实差距 7.3×。** 差距来自 **单 chunk（借槽）+ kv-resident + config**（`STRATA_PF_FUSED` 只 +5%）；**这三项机制我们的 `strata-poly` 全都没有**。

## 0. ★ 更正（自我留痕）
- 先前我用**社区 IQ3_S 配置**跑 3.8 @ 4096 = **465.9 tok/s**，并据此判"只差 1.36×" —— **基准选错了**（社区配置：`--max-context 65536`、无 `--kv-resident`、无 `STRATA_PF_FUSED`、非单 chunk）。
- **正确的本机基准 = 2805.87（6792 token，`dec-pool2` 那套旗标）**。**1.36× 作废，真实 7.3×。**

## 1. ★ 判决数（A 级，本机实测）

| 谁 / 配置 | prompt | prefill | 备注 |
|---|---|---|---|
| **3.8-Flash · `dec-pool2` 配置（含 PF_FUSED）** | 6792 | **2805.87 tok/s** | 单 chunk 6912 · borrows 1903 槽(3.62 GiB) · resident 10109 · streamed 14467 |
| **3.8-Flash · 同配置去掉 PF_FUSED** | 6792 | **2670.68** | borrows 2057 槽(3.91 GiB) · resident 9097 · streamed 14627 |
| 3.8-Flash · **社区 IQ3_S 配置** | 4096 | 465.9 | **错误基准**（见 §0） |
| **我们（Nemotron）· pb=1024 sm=2048** | 6792 | **383.6** | resident 6892 |
| 我们 · 单 chunk（pb=6912） | 6792 | 290.0 | resident 3640（scratch 抢 arena） |

- **⇒ 3.8 用的旗标**：`STRATA_PF_FUSED=1` · `--vram-reserve-mib 1500` · `--kv int8 --kv-resident 32768` · `--max-context 262144 --rope-scaling yarn` · `--expert-profile` · `--mtp --spec 4 --spec-min-p 0.5 --suffix-draft 3` · `--tokens-file`（0.1.43.1；日志原件 0.1.40.2，版本差已标）。

## 2. ★ 消融归因：PF_FUSED 只 +5%
- **full 2805.87 vs 去 fused 2670.68 = 1.05×** ⇒ **fused 不是主因**。
- ⇒ **6× 级的主因 = 单 chunk（借槽） + kv-resident + config**。
- **⇒ 对我们 poly 的 ROI 排序**：① 借槽 + 缩 scratch（让单 chunk 兑现）；② kv-resident；③ PF_FUSED（最后，+5%）。

## 3. ★ poly 与 program 线的能力差（A 级，源码+exe）
| 机制 | program 线（strata.exe） | 我们（strata-poly） |
|---|---|---|
| `STRATA_PF_FUSED` | 有 | **无**（exe 0 命中） |
| `--kv-resident` | 有 | **无** |
| `prompt path borrows` | 有 | **无** |
| `--tokens-file` | 有 | **本轮已补上**（见 §5） |

## 4. `--stage-mib` 最优随 prompt 长度翻转（A 级；ids 逐位同、无 NaN）
| prompt | sm=48 | sm=2048 | 倍数 |
|---|---|---|---|
| 700 | **146.8** | 136.7 | 0.93× |
| 4096 | 160.9 | **342.4** | 2.13× |
| 6000 | 213.4 | **373.5** | 1.75× |
| 6792 | — | **383.6** | — |
- **规则**：短 prompt 用 sm≈48；长 prompt（≥~2–4K）用 sm≈2048。三档 ids 逐位同（`10041,1278,4206,1636,9956,1317,6347,1261`）。

## 5. `--tokens-file`（已加，等价性过）
- 原因：`--tokens` 撞 Windows 32,767 字符（8192 ids=42.9 KB → `Argument list too long`）。
- 实现（`src/poly/main.cpp`）：读文件（逗号/空白/换行均可）→ **填同一个 `ids`**；`--tokens` 与 `--tokens-file` **互斥报错**；非数字/越界响亮报错。
- **等价性 ✓**：p4096 走 `--tokens` vs `--tokens-file` ⇒ **ids-md5 `3093a677…` 与 top-probs-md5 `3b972446…` 逐位同**。

## 6. 单 chunk 试验（负结果，`STRATA_MAX_PROMPT_BATCH`）
- `--prompt-batch` 上限（`1..4096`）已做成 **env `STRATA_MAX_PROMPT_BATCH`**（默认 4096=旧行为，可关）。
- 同 6792 prompt：**pb=1024+sm2048 = 383.6（resident 6892）** vs **单 chunk pb=6912 = 290.0（resident 3640）** ⇒ **单 chunk 慢 24%**。
- 根因：**scratch ∝ rows_ 单独 alloc（`gu_`/`hx_`/`dx_` @ `engine.cpp:1584-1586`）⇒ 吃走 ~10.8 GiB ⇒ arena 腰斩**。**3.8 用 `borrows N cache slots`（借），我们无此机制。**

## 7. 其它（均有数）
- **驻留拉满**（700 tok, sm48）：reserve 800 → 7964 专家、prefill 158.9（+8%，单跑含噪，有压穿风险）。
- **页缓存预填**：dd 冷读 4.5 GB/s；预填后 prefill 134.6（≤145.6）⇒ 兜不住 65 GiB 模型。
- **H2D 拆账**（700 tok 正确路径）：prefill 窗口 5.16 s = H2D 3.83 s（74%, 34.87 GB @ 9.1 GB/s）+ kernel 0.47 s（9%）+ 缺口 0.87 s（17%）。
- **pinned benchmark**：pinned 23.5–23.9 GB/s、pageable 11.4–11.8；引擎 9.1 ⇒ 瓶颈在 `stage()` 循环。

## 8. 当前 exe 与交付
- **exe sha `1fe9784f…`**（latent 修复 + class 切段 + SMEM ① + `--tokens-file` + `STRATA_MAX_PROMPT_BATCH`；所有开关默认回旧）。
- **成绩（"算得对"）**：700 tok = 146.8 · 4096 tok = 342.4 · 6792 tok = 383.6。
- **方案件（只方案）**：`方案-prompt缓冲借缓存槽-20261017.md`（借槽语义/冲突保证/§7 影响/scratch 10.8 vs 3.8 的 3.62 GiB 根因）。

## 9. 未验 / 待
- **借槽 + 缩 scratch**（未做，待批）· **kv-resident 移植**（未做）· **PF_FUSED 内核**（未做，+5%）。
- **8192/16384 的长度×sm**（`--tokens-file` 已具备，未扫）。
- **3.8 的 host 1896 ms（有 fused）vs 92.9 ms（无 fused）** 未解释（墙钟只差 5%，不阻塞）。

## 10. 落盘（绝对路径）
- 本件：`…\P3-20261016\回执\长prompt-3.8对拍-20261017.md`
- 日志：`C:\sp3\{q38_repro_6792, q38_nofused, q38_perf, q38_4096, lp_L1_*, lp_L2_*, lp_L3_*, p6k_sm*, ctl_6792_pb1024, onechunk_6792, tf_tokens, tf_file, rv_*, warmcache_on}.log`
- prompt：`C:\sp3\{q_prompt.txt, q700.txt, q4096.txt, q6792.txt, p6000.txt, p6792.txt, p8192.txt, p16384.txt}`
- 工具：`C:\sp3\{pcibench.exe, gatherbench.exe, gguf_vocab.py}` · nsys：`C:\nsys\{ln_correct, q38}.nsys-rep`
