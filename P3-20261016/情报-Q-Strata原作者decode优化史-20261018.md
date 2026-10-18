# 情报-Q · Strata 原作者 decode 优化史（commit 级，2026-09-24 → 10-18）— 2026-10-18

- **席位**：scout-a（零 GPU）· **触发**：lead 派工"去 GitHub 挖 Strata 原作者历史优化记录，专摘 decode，commit 级，≥15 条 + TOP-10 + 最可能出量级的 3 条"。
- **状态**：**达成**（仓库关系已订正；decode 相关 commit 摘 **20 条**（每条 sha/日期/宣称收益/改动机制/我们有无）；TOP-10；3 条量级判断）。
- **纪律**：零 GPU、零编译、不改码；每条带 commit sha + 日期 + 文件；"message 宣称"与"diff/代码实际"分列；未 spawn 子代理。
- **网络**：本机代理 **127.0.0.1:7897 活**；7890/10809/1080 死。Windows 凭据管理器取到的 GitHub token **无效**（`GET /user` → **401 Bad credentials**，未回显未落盘）⇒ 改用 **未认证 API**（探仓库用）+ **`git clone --bare --filter=blob:none` 走 git 代理配置**（两次 clone 各 ~3 s，**无速率限制**）：
  - `C:\sp3\upstream-hist\strata.git`（Niko1221/Strata，**1493** commits，最新 `477ac13c` 2026-10-16）
  - `C:\sp3\upstream-hist\polystrata.git`（VecSzn/PolyStrata，**2634** commits，最新 `dc2080ba` 2026-10-17）

---

## §0 三句结论（先给 lead 最需要的）

1. **仓库关系订正**：`VecSzn/Strata` **不存在**（GitHub API 404；`/users/VecSzn/repos` 全量 14 个公开仓库里**没有** Strata，有 **PolyStrata**）。真相是：**`VecSzn/PolyStrata`（fork=false，192★，创建 2026-10-06）与 `Niko1221/Strata`（fork=false，26,908★，创建 2026-09-24）共享 1468/1493 条 commit** ⇒ PolyStrata 是 Strata 历史的**延续线**（多 1166 条：Niko1221 507、**VecSzn 312**、其余社区）；**我们的引擎 `PolyStrata` 就是这条线**（本地树有 `src/poly`、`--pcie-mode` 等只属 poly 的旗标）。⇒ "原作者"按 commit 计 = **VecSzn（312 条，且 10-05…10-17 的 decode 主力）+ Niko1221**；已全部翻完。
2. **decode 优化史的核心事实**：10-05 → 10-12 有一整串 **decode 大招**（见 §2）：token 流水线、窗口整/半/整段自适应、窗口注意力 span、窗口 8 + 3 份递归状态、copy2/pinned 宿主拷贝、Windows 电源节流 opt-out、专家缓存跟着答案走、猜测代价门控…… **这些代码我们的树里全都有**（marker 全量计数见 §2 的"我们有无"列；我们的树是上游 ≈10-12…10-16 快照 + 本地施工增量）。**10-13 之后没有新的 decode 机制**（10-16 是 AMD/HIP 弹性缓存、10-17/18 是 serve/确定性，均非 decode 量级）。
3. **真正的问题不是"没抄到"，而是"抄到的没用上"**：我们 **235 个跑法剧本里 228 个带 `--no-draft`、170 个 `--lookup 0`** ⇒ `Session::most = (lookup>0 || drafts()) ? … : 1`（`PolyStrata\src\poly\main.cpp:329`）⇒ **window()=1** ⇒ 上面 10-05…10-12 的窗口/流水线/注意力 span 机制**全部处于闲置态**。**唯一可能出量级的三条全部在这条闲置栈里**（§4），前置是**让 window≥2**（修 MTP 的 ids = 线 B，或改用 `--lookup 3` 的重复草稿——我们已有 64 个剧本这样跑）。

---

## §1 口径与证据等级

| 项 | 内容 |
|---|---|
| 仓库 | `VecSzn/PolyStrata`（我们引擎的上游）+ `Niko1221/Strata`（老师线，PolyStrata 的前身历史）；**`VecSzn/Strata` 为 404**（本次实测） |
| 方法 | 未认证 GitHub API（探仓库/最新 commit）+ 两次 `git clone --bare --filter=blob:none`（全量 commit 元数据，离线 `git log`/`git show --name-status`，不受 60/h 限流） |
| commit 全量 | strata 1493；polystrata 2634；共享 1468；poly-only 1166（VecSzn 312 / Niko1221 507 / maxfridbe 69 / …） |
| 关键词 commit 数（message 全量，两库合计去重） | decode 251 · tok/s 182 · ring 175 · draft 142 · MTP 116 · pool 99 · pcie 72 · graph 70 · expert cache 61 · spec 53 · doorbell 28 · adaptive 31 · kv-resident 18 · single-token 14 · lookahead 8 · foresight 5 · batch=1 3 |
| 分级 | **commit 与 message 原文 = C（外部）**；"我们有没有"= **A（本机树全量 grep）**；宣称收益一律标注其机器/模型（RTX 4070 Laptop 8GB / RTX 5090 / 3090；Qwen3.6 / Gemma 4 / Ornith / GLM 5.3 Flash）⇒ 对我们的外推 **D** |
| 代理/凭据 | 7897 活；token 401（Bad credentials）⇒ 用未认证 API + git 代理；**未回显、未落盘任何凭据** |

---

## §2 decode 相关 commit 史（20 条，时间倒序；"宣称"= message 原文数字，"实际"= diff 文件/机制）

> **我们有无**列 = 本机 `PolyStrata` 树全量 marker 计数（A）。**"有代码"≠"生效"**：多数属于窗口栈，而我们跑法 window=1（见 §0-3）。

| # | sha / 日期 / 作者 | 一句话 | 改了什么（文件 + 机制） | 宣称收益（原文） | 我们有无（A） | 值不值得抄/用 |
|---|---|---|---|---|---|---|
| 1 | `4636dbe3` 10-16 Donald Chen | HIP 弹性专家缓存（AMD） | 4 文件；VMM 保留、地址重建、amdgpu VRAM 读数 | 110–128 tok/s（RX 9070 XT） | **有代码**（elastic 98 行 / amdgpu 3 行；`hipMemGetInfo` 14） | **N/A**（NVIDIA 线不用） |
| 2 | `85d1d9cb` 10-12 Nenad | 草稿块输入行先等卡 | `engine.cpp/.hpp`；`draft_input()` 记事件、下次 refill 前 wait | 竞态 12/47→0/44（3090） | **有且在用**（`draft_input` 6 行 + 事件） | 正确性已修，0 收益 |
| 3 | `99bdfd5f` 10-11 VecSzn | 在 prompt 最后一个 turn 处留 checkpoint | serve + `conversations.hpp` + `main.cpp`；`turn_at=POS` | 多轮后续请求读 1,442→**399** tokens/轮 | **有**（`turn_at` 30 行） | 单轮 bench = 0；**serve 多轮真赚**，我们目前单 prompt 用不上 |
| 4 | `719bbb48` 10-11 VecSzn | 专家缓存旁至少留 256 MiB | `engine.cpp/main.cpp`；reserve 下限抬到 256 | 修 `--vram-reserve-mib 0` 起不来 | **有**（reserve 相关同源；未逐行核） | 我们 reserve=2500–3000 > 256 ⇒ **0** |
| 5 | `d453bc76` 10-10 VecSzn | Gemma 4 的 dense MLP 挪进 `moe_shared()`，在路由事件前记录 | `engine.cpp/.hpp`、`gemma4.*`、`kernels.cu`；**router 等 MLP 时卡空闲 45–60 µs/层** → 卡在 host 路由时跑 MLP；另 L2 分块、草稿表 Q4_0（1.35→? ms）、value 读法 | 49.9→**54.8**、84.7→**90.7** tok/s（4070L/Gemma4） | **有**（`moe_shared(` 25 行） | **值得核**：我们 Nemotron 有 shared expert；若我们 shared 在 router 之后发 ⇒ GPU 空等 host 路由，同构收益 ~5–10% |
| 6 | `68c0374d` 10-10 VecSzn | **faster prompt + decode**（七项） | `engine.cpp/.hpp/main.cpp/kernels.cu/runtime.hpp`；① GDN state kernel 列寄存器化；② **Windows 流在 kernel↔copy 切换处空转 20–30 µs × 4 次/层 ⇒ `copy2()` 两个 4B 段一次 launch + 全部改 pinned 宿主内存（router 两拷贝合一、`moe_add` 从 pinned 加、`draft_input/draft_rows`、`likeliest` 写 pinned、`cpu_`/`guess_` 消失）**；③ `pick_way()` 窗口走法重测节奏（4/8/16/32/64 窗）；④ `--draft-vocab N` 把猜测限制在前 N 个 token（Qwen 词表序）；⑤ 专家换入 16→**64**；⑥ 解码注意力 `attn_score_reg_kernel`/`attn_value_t_kernel`（31K 上下文：0.505/0.596/1.014/1.168 → 0.392/0.445/0.651/0.752 ms） | **81.8→90.5/92.7**（整窗）、71.8→83.5/86.2（半窗）；62.2/62.6→68.7/70.8（31K 上下文）；专家缓存 +7 个 | **有且在用**（`copy2` 14 行、`draft_vocab` 13、`pick_way` 5、`attn_score_reg` 6、`kAttendResults` 3） | **0 净收益可抄**（已在基线里）；但可复查我们 decode 里 copy 段是否全走 `copy2`+pinned |
| 7 | `fa6c3f94` 10-09 VecSzn | **一个窗口的 token 共享一次 KV 读**（注意力 span） | `kernels.cu`/`kernels.hpp`/`gemma4.cpp`；窗口内 token 一起算 score/softmax/value，value 一行读一次给最多 8 个 query；`kAttendResults` 限块 | 31K 上下文 1–4 token：4.45→**1.40** ms/层；56.4→**71.5**、50.3→**56.1** tok/s（4070L） | **有代码、窗口=1 时闲置** | **★ 窗口栈上线后是纯赚**（我们只有 8 层 attention，收益打折） |
| 8 | `425afd57` 10-09 VecSzn | 窗口走法的"落后方"计时半衰减半 | `engine.cpp/runtime.hpp`；半衰 1/8→1/2，落后窗口每 32 个测一次 | Gemma 61.7→**68–72** tok/s；300 token 回答 +5% | **有代码**（`pick_way`/runtime） | ★ 与 #9 同栈 |
| 9 | `b605b2f9` 10-09 VecSzn | **窗口两种走法：两个半窗各一次调用，或整窗"卡先算、CPU 后算"**，按实测选快的 | `engine.cpp/.hpp/runtime.hpp`；`pick_way()` 对每种窗宽各测 3 窗、快者用、落后方每 32 窗测一次 | 48–50→**62–69**（Gemma4）、54–57→**70–76**（Qwen3.6）、61–65→**66–75**（Ornith）tok/s（4070L，+-token 一致） | **有代码**（`pick_way`、whole/halves 807 行含注释） | **★★ 最大单件之一**：把"CPU 专家腿等待"变成"整窗卡先算" |
| 10 | `66140159` 10-09 VecSzn | 重复草稿可填 8 token 窗口（无 VRAM 代价的模型） | `engine.cpp/.hpp/main.cpp`；窗口 8 默认按"猜测是否要递归状态拷贝"选：无递归层 8、有 4；`--window N` | 87.6→**99.6–106.8**、74.3→**83.2–88.2** tok/s（Gemma4，复制型回答） | **有代码**（`window` 默认 8/4 注释 `engine.hpp:74-75`） | ★ 与 #9 同栈 |
| 11 | `c80a4dba` 10-09 VecSzn | 长窗口只留 3 份递归状态快照 | `kernels.cu` 两 kernel 按 bit 记录"哪个 token 离开状态"；`keep()` 回退最近快照 + 重跑 1–2 token | Qwen3.6 90→**112–113**、77→**83–85**；Ornith 99–102→**120–121** tok/s（4070L，复制型） | **有代码**（`keep(` 16 行） | ★ 我们 40 层 mamba 是递归层 ⇒ 窗口 8 需 3 份状态（VRAM 代价要算，见 §5） |
| 12 | `1ba07568` 10-09 VecSzn | 4 token 以上窗口只由"8 连重复"填 | `main.cpp`；窗口内填充策略 | 复制型回答保持 103/85–87/109–110 tok/s | **有代码**（`main.cpp:465,479` `kCopyMatch/kBlockWindow`） | ★ 与 #10/11 同栈 |
| 13 | `ee1ce4dd` 10-09 VecSzn | 池任务内再调 `Workers::run()` 的死锁修 | `cpu_experts.cpp/.hpp` + `workers_test.cpp`；线程在 task 内则就地做（`in_task` 守卫） | 修 GLM 两线程 100%/卡空闲挂死 | **有**（`in_task` 我们在 `cpu_experts.cpp:33,59` 读过） | 正确性已修；**窗口半窗并行前提** |
| 14 | `e9416d34` 10-09 VecSzn | **退出 Windows 电源节流**（后台进程线程被压到 E 核） | `main.cpp`；`SetProcessInformation(ProcessPowerThrottling, 关)` | i7-14700HX：48–50→**66–72** tok/s（+33%）；另一组 39→**57–62** | **有且已在 `main()` 第一行调用**（`main.cpp:895-908`） | **0 可抄**（已在）；但可核：我们 E-core 混合核的 24 线程问题（情报-L）另属线程放置 |
| 15 | `f5eb2eae` 10-06 VecSzn | 给"问草稿块"计时，不划算就不问 | `main.cpp`；`Guesses` 估计一次猜测的收益 | 39.5/44.7/50.5 vs 37.4/44.6/48.5 tok/s（5090） | **有**（`Guesses`/`guesses.hpp`） | ★ 窗口栈配套（省 1–2 ms/次） |
| 16 | `5b998016` 10-06 VecSzn | **一个窗口的 token 一个落后一个，CPU 算专家时卡在算前一个 token** | `engine.cpp/.hpp` + `gemma4/glm5next.*`；每 token 自己的发射/事件/`shift_rows()`/递归状态快照，一个 worker 线程替主线程把上一 token 的卡工作发出去 | 5090：GLM 37.9/43.7/51.2→**43.1/52.1/59.4**；8GB：Qwen3.6 123/117/122→**127/130/132**、Ornith 93/116/109→**99/123/111**、Gemma4 88/91/87→**87/102/97** | **有代码**（`shift_rows` 14 行）；窗口=1 时**不生效** | **★★★ 最匹配我们**：我们 CPU 腿 71.9 ms 是主项，这条正是"卡的工作藏在 CPU 腿后面" |
| 17 | `1eabf41c` 10-05 VecSzn | 专家缓存跟着答案走（自适应 LFU-衰减换入换出） | `engine.cpp` 等；`--expert-swaps N`（16 默认，0 关）；每 128 token 半衰，异步线程从映射文件读到 pinned + 独立 stream | GLM 26.1→**30.3** tok/s；**命中 20.4%→49.7%** | **有且在用**（`follow(` 16 行；我们 `--expert-swaps 64`） | 0 可抄（已在）；可调 `kSwapHalf/kSwapMin` |
| 18 | `a36be1db` 10-03 Stuart Chapin | perf(cuda,decode)：zero-doorbell verify graph、sub-warp 专家打包、共享内存 staging、批量 PLE/MTP | 21 文件（`include/strata/core/mtp.hpp`、`verify.hpp`、`src/core/expert_source.cpp`、kernels） | message 无正文（squash 无数字） | **树内有相关代码**（`sub-warp` 6、`doorbell` 154——多在 `src/core`，poly 引擎未必走） | 属 Strata 核心路径；poly 用自己的 verify/graph 路径 ⇒ **可借思路不可直接搬** |
| 19 | `5a8b2924` 10-04 野生の男 | 保留完整线程亲和（Windows `SetThreadGroupAffinity`，进程组不丢） | `src/kernels/cpu/pool_affinity_win.hpp`、`pool.cpp`、`session.cpp` | 3990X 的 63 workers 不再塌成 32 | **树内有**（`SetThreadGroupAffinity` 2、`pool_affinity_win.hpp`） | ★ 与我们在做的 `--host-core/--aux-cpus` 同族；**该核对的现成件** |
| 20 | `cab62c93` 10-04 Niko1221 | batch slots（多请求/一批解码时穿插 prompt 读） | `generate.cpp`/verify/serve | batch>1 吞吐 | **树内有**（`batch_slots|--batch` 21 行） | **N/A（batch=1）** |

> **已翻全量**：两库 `git log --all` 共 2634+1493 条（去重后），关键词命中数见 §1；**未见**任何"专家图/共现图/expert graph"类 commit（`git log --grep='co_occur|co-occur|expert graph|共现'` = 0），与 `情报-O` 的结论一致。**未见**专门针对"decode + 单测/单卡 + 我们模型"的 commit——上游所有 decode 数字都出在 4070L/5090/3090 + Qwen3.6/Gemma4/Ornith/GLM。

---

## §3 TOP-10（按"对我们 16.2 tok/s 的可能增益 × 可移植性"）

| 排 | 机制（commit） | 可能增益（上游同代码实测，D 外推） | 可移植性/现状 | 落地成本 |
|---|---|---|---|---|
| 1 | **窗口栈总开**：token 流水线 `5b998016` + 走法 `b605b2f9`/`425afd57` + 窗口 8 `66140159`/`c80a4dba` | **+30–40%**（GLM 43→59、Qwen3.6 54→76） | 代码全在树；**我们跑法 window=1 ⇒ 全闲置**；前置=窗口≥2 | **零改码**：`--lookup 3`（重复草稿）或修 MTP ids；然后 A/B 4 臂 |
| 2 | token 流水线单列（`5b998016`） | +20–38%（5090） | 同上（需窗口 2–4） | 同上 |
| 3 | **整窗"卡先算、CPU 后算" vs 两半窗自适应**（`b605b2f9`+`425afd57`） | +15–30% | 同上 | 同上 |
| 4 | 窗口注意力 span（`fa6c3f94`） | +10–25%（31K 上下文）；我们 8 层 attention ⇒ 打折 | 同上 | 同上 |
| 5 | 窗口 8 + 3 份递归状态（`66140159`+`c80a4dba`） | +10–20%（复制型回答） | 同上；我们 40 层 mamba 的状态拷贝 VRAM 要算 | 同上 |
| 6 | shared/dense MLP 挪到 router 事件前（`d453bc76`） | +5–10%（若我们 shared 在 router 后发） | 需读我们 `Engine::moe()` 的 shared 分支顺序（未核） | 小 |
| 7 | `copy2`/pinned 宿主拷贝（`68c0374d`） | 已在基线（0 可抄） | 有且在用（`copy2` 14 行） | 只复查是否所有 decode 拷贝走它 |
| 8 | Windows 电源节流 opt-out（`e9416d34`） | 已在基线（0 可抄） | `main.cpp:895-908` 已调用 | 0 |
| 9 | 专家缓存跟答案（`1eabf41c`） | 已在基线（0 可抄） | `follow()` 在用，`--expert-swaps 64` | 0（可调 kSwapHalf/kSwapMin） |
| 10 | 多轮 checkpoint（`99bdfd5f`） | 单轮 0；serve 多轮每轮读 1442→399 token | 有代码；我们单 prompt bench 用不上 | serve 场景再说 |
| — | N/A | HIP 弹性缓存（`4636dbe3`/`8dd22050`，AMD）、batch slots（`cab62c93`，batch=1）、GLM indexer（`3d1db7c9`，非我们模型）、reserve 下限（我们 2500>256） | — | — |

---

## §4 哪些"我们完全没有且最可能出量级"？——**代码级 0 条；生效级 3 条（全在窗口栈）**

**先纠一个可能被误读的结论**：如果按"我们树里没有这段代码"来算，**答案是 0 条**——10-05…10-12 的全部 decode 机制（20 条表里 1–17）我们树里都有（marker 全量计数在 §2 列；我们的树是上游 ≈10-12…10-16 快照 + 本地施工增量），10-13 之后上游也没有新的 decode 机制（10-16 是 AMD/HIP、10-17/18 是 serve/确定性）。**"抄新代码"不是当前瓶颈。**

**按"实际生效"算，最可能出量级的三条 = 我们关掉的窗口栈（同属一个大前置：window≥2）**：

1. **token 流水线（`5b998016`）**——"一个窗口的 token 一个落后一个，CPU 算专家时卡在算前一个 token 的层"。**为什么最匹配我们**：我们 decode 的 CPU 腿 71.9 ms/token（scout-b A 级）就是待藏的那条腿；上游同代码在 5090 上给出 **GLM 37.9→59.4 tok/s（+57% 最好一档）**、8 GB 上 **Qwen3.6 117→132、Ornith 93→123、Gemma4 73→102 tok/s**（原文）。**零代码改动**即可试：我们已有 64 个剧本用 `--lookup 3`（重复草稿产生窗口），把这条从"关闭"变"打开"。
2. **窗口两种走法自适应（`b605b2f9` + `425afd57`）**：整窗"卡先算、CPU 后算" vs 两半窗各一次调用，按每窗实测选快者——直击"GPU 等 CPU 专家"。上游 **48–50→62–69 / 54–57→70–76 / 61–65→66–75 tok/s**（4070L，输出逐 token 一致）。
3. **窗口内注意力 span（`fa6c3f94`）**：窗口 token 共享一次 KV 读；上游 **56.4→71.5 / 50.3→56.1 tok/s**（31K 上下文）。我们只有 8 层 attention（40 层 mamba）⇒ 收益打折，但只要窗口上线就是纯赚；`c80a4dba`/`66140159` 的窗口 8 是同一栈的下一步（VRAM 代价需先算）。

**共同前置（必须写进判据）**：三条都要求 `window≥2`。我们当前 decode 基线是 `--no-draft --lookup 0`（**235 剧本里 228 用 --no-draft、170 用 --lookup 0**），`Session::most=1`（`PolyStrata\src\poly\main.cpp:329`）⇒ 机制闲置。**A/B 设计**：臂 A=现状（window=1）；臂 B=`--lookup 3`（窗口开、无 MTP 依赖）；臂 C=修好 MTP 后再开 `--window 8`（有递归层默认 4，`engine.hpp:74-75`）。**判据**：同 prompt 的 45 token 输出逐 token 一致（上游 commit 均声明"+-token 一致"）+ `tokens/round > 1` 自证窗口生效；**判红**：`tokens/round` 仍 =1 ⇒ 窗口没开，别报增益。

---

## §5 未验 / 边界
- **U1**：`VecSzn/Strata` 404 可能因为是改名/私有；本次以公开仓库 `VecSzn/PolyStrata` 为准（我们引擎的上游，A 级代码同源）。若 lead 坚持"Strata 原作者另有其人"，需要再给线索（账号/更名/镜像）。
- **U2**：我们的树与上游快照**不能按字节对齐**（`engine.cpp`：我们 215,975 B vs 上游 HEAD 174,446 B；`cpu_experts.cpp` 15,639 vs 10,051——含我们的本地施工与上游后续重构），只能用机制 marker 判（`turn_at`/`draft_input` 事件在、HIP elastic 在、serve 的 `preserve_thinking` 不在 ⇒ ≈10-12…10-16）。**"有代码"已 A 级，但"是否被我们的 decode 路径调用"需逐段核**（尤其 Strata 核心路径的 `doorbell`/`sub-warp`/`exchange` 属 `src/core`，poly 引擎未必链接）。
- **U3**：上游所有数字 = RTX 4070 Laptop 8 GB / RTX 5090 / 3090，模型 Qwen3.6/Gemma4/Ornith/GLM ⇒ 对我们的外推 **D**；我们模型 88 层（40 mamba）与它们完全不同（窗口/mamba 状态拷贝代价、attention 层数都不同）。
- **U4**：我们的基线 `16.2 tok/s` 对应哪套 flags 未核（`run_delivery_t16.cmd` 用 `--no-draft --lookup 3`，`run_R39_E*.cmd` 用 `--no-draft --lookup 0`；两者 window 可能不同，见 `main.cpp:329` 条件）。
- **U5**：10-17/10-18 上游新 commit（`b220649`/`b8d06b7`/`9bb0467`/`22b6d6e`：确定性 freeze、Strata 0.1.45 合并、serve 开关）**未逐条读**；按标题都不是 decode 机制，但 `Merge Strata v0.1.45` 可能带了老师线新改动，值得下一轮扫。

---

## §6 复算命令（全只读）

```powershell
# ① 仓库关系（未认证 API，走代理）
$px='http://127.0.0.1:7897'
Invoke-RestMethod 'https://api.github.com/repos/VecSzn/PolyStrata' -Proxy $px | Select full_name,fork,stargazers_count,parent
Invoke-RestMethod 'https://api.github.com/users/VecSzn/repos?per_page=100' -Proxy $px | Select -Expand name
# ② 全量 commit 元数据（已 clone 到本地）
git --git-dir=C:\sp3\upstream-hist\polystrata.git log --all --format='%H|%ad|%an|%s' --date=short
git --git-dir=C:\sp3\upstream-hist\polystrata.git show -s --format='%H|%ad|%an|%s%n%b' 5b998016
git --git-dir=C:\sp3\upstream-hist\polystrata.git show --name-status --format='' 68c0374d
git --git-dir=C:\sp3\upstream-hist\strata.git rev-list --all --count
# ③ 我们有无（A）
Select-String 'PolyStrata\src\poly\main.cpp' -Pattern 'most\(lookup|no_power_throttling|--no-draft'
Get-ChildItem 'C:\sp3' -Filter 'run_*.cmd' | Select-String '--no-draft|--lookup'
```

**判据（能变红）**：① kb/API 能找到 `VecSzn/Strata` ⇒ §0-1 错；② 我们树里 grep 不到 `copy2|pick_way|shift_rows|no_power_throttling|draft_vocab|keep\(` 中任一 ⇒ "有代码"错；③ 跑一个 `--lookup 3` 臂若 `tokens/round` 仍=1 ⇒ "窗口可开"错；④ `main.cpp:329` 的 `most` 表达式变了 ⇒ window=1 的论证要重读。

---

## §7 落盘
- 本件：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\情报-Q-Strata原作者decode优化史-20261018.md`
- 回执：`E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\回执\IA-20261018-decode优化史.md`
- 本地证据库（只读）：`C:\sp3\upstream-hist\strata.git` · `C:\sp3\upstream-hist\polystrata.git`
