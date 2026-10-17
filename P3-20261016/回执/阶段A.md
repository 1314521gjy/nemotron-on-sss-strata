# 回执 · 阶段A（工具落地）— 2026-10-17 09:1x

- **任务**（`计划-分环节定位杠杆-20261017.md` §3 阶段A）：A1 解包 0.1.43.1 原厂引擎 · A2 ncu 真采一次最小 kernel · A3 3.8-Flash 冒烟。门：**全绿才往下**。
- **状态**：**A1 ✅ / A2 ✅ / A3 ✅ 全绿**。
- **结论一句**：**三条工具链全部打通** —— 0.1.43.1 原厂引擎能跑 3.8-Flash（RC=0）、ncu 能出真计数器且**不需管理员**、nsys 的 `bad conversion` **真因 = 中文路径**（全 ASCII 下两个 exe 都能采）。

## A1 ✅ 解包 0.1.43.1（未落 `E:\strata`）

- **源**：`E:\strata\_staging\strata-windows-x64-v0.1.43.1.zip`（138,187,009 B，**只读**）⇒ 拷到 **`C:\sp3\`** 解包。**未落 `E:\strata`**。
- **核 `BUILD.json`** = **`0.1.43.1`**（`source: release` · `cuda: 13.0` · `archs: [75,86,89,120]` · `ptx: true` · `vision: gpu` · `portable: true`）。**门过**。
- **zip 只含 3 件**：`strata.exe`(168,322,560 B) · `strata-vision.exe`(108,173,312 B) · `BUILD.json`(156 B)。
- **sha256**：`strata.exe` = `aa690ae65bfd306d4c84525548a83de14e48d95b81f11ad451bda7f5f50bc13a` · `strata-vision.exe` = `501a37fed09088017c8a0885468f213a52d7e894f3335e39270f8a77c0b6cff1` · `BUILD.json` = `aab806280fdf31a92b9fb100f4a9aab9771f0f41aa82bcce703727b7bdf6fccd`。
- **跑通验证**：`C:\sp3\strata.exe --help` ⇒ **RC=0**（usage = `strata generate --pack DIR --tokens "1,2,3" [options]`）。

### ★ A1 三个坑（已修，供全队记）
1. **缺 CUDA 运行时 DLL**：zip **不带** `cublas64_13.dll` ⇒ 直接跑报 `cannot open shared object file`。**修**：从 **`E:\llama-cpp\`**（CUDA **13.3**，同大版本）拷 `cublas64_13.dll` / `cublasLt64_13.dll` / `cudart64_13.dll` 到 `C:\sp3\`（**未碰 `E:\strata`**）。
2. **没有 `generate` 子命令**：usage 里 `strata generate ...` 是它**自称**；真调用 = **`strata.exe <旗标...>`**。带 `generate` ⇒ `unknown argument: generate`。
3. **`--tokens` 必需**（逗号分隔 **token id 列表**，非文本）；另有 **`--tokens-file PATH`**（长 prompt 正路）。

## A2 ✅ ncu 真采最小 kernel（能出数 · 不需管理员）

- **命令**：`ncu.exe --launch-count 1 -f -o C:\ncu\cell --target-processes all C:\sp3\mamba2_cell_test.exe`（exe 已拷到 `C:\sp3` ASCII）。
- **结果**：`==PROF== Profiling "conv_silu_kernel" - 8 passes` · 报告 **`C:\ncu\cell.ncu-rep`（126 KB）** · **RC=0**。
- **要不要管理员**：**不要**（当前非提权 shell 直接采成功，无权限错、无提权提示）。
- **导入核验（真计数器）**：`Duration 5.12 µs` · `DRAM Frequency 11.15 GHz` · `Memory Throughput 1.27%` · `DRAM Throughput 1.03%` · `Compute (SM) 0.05%` · `Achieved Occupancy 8.12%` · `Waves 0.00` · `Grid 1 / Block 256` · `Registers 40`.
- **口径自评**：该 kernel **grid=1 block ⇒ 8% 占空、launch-bound**，**只证工具可用**，不代表引擎任何 kernel；真采留给 C2（top-N）。

## A3 ✅ 3.8-Flash 冒烟（0.1.43.1 原厂）

- **有效命令**（cwd = `C:\sp3` 纯 ASCII）：
  `C:\sp3\strata.exe --pack E:\strata-data\packs\iq3_s --native E:\strata-models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf E:\strata-models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile C:\sp3\expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp E:\strata-data\mtp\rt --max-context 65536 --kv int8 --pcie-frac 0.55 --tokens "1,2,3" --max-new 8`
- **结果（A 级）**：`RC=0` · `prompt 1 2 3` → `output 4 5 6 24218 10 4838 1665 15` · **decode 8 tok / 178.7 ms = 44.77 tok/s** · prefill 2 tok / 257 ms（**prompt 太短，无意义**）。装载 46.84 GiB @ 2.74 GiB/s · expert cache **8967 slots / 22.23 GiB** · speculation 4 rounds accepted 4/4 · adaptive tier 96 experts swapped。**真启动 = 成功完成一次最小工作 ⇒ 门过。**

### ★ A3 两个坑
1. **社区配方缺 `--expert-profile`，而 native IQ pack 强制需要**：不传 ⇒ `--spec needs the device residency table (--expert-profile, --expert-cache and the token graph)`；给**不存在的路径** ⇒ `read_expert_profile: cannot open`。**本机无 `E:\strata` 外的现成 profile**；唯一 **byte-identical** 副本 = **`E:\cyber-on-strata\data\expert-profile.bin`**（sha `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf` = `E:\strata\data\expert-profile.bin`）⇒ 只读拷到 **`C:\sp3\expert-profile.bin`**（**未碰 `E:\strata`**）。
2. **`--resident-budget-gib N` 会触发驻留模式**，进而要求 `--mmap-experts` + 静态 `--expert-profile`（报 `--resident-cpu-experts requires --mmap-experts and a static --expert-profile`）⇒ 冒烟里**去掉**该旗标。

## ★ nsys `bad conversion` 真因（决定性对照，A 级）

- **两臂**（都在 **`C:\sp3` ASCII cwd** + ASCII `-o`）：
  - `nsys profile -o C:\nsys\qa -- C:\sp3\strata.exe --help`（0.1.43.1）⇒ **成功**，`C:\nsys\qa.nsys-rep`（81,572 B）。
  - `nsys profile -o C:\nsys\ourhelp -- C:\sp3\strata-poly.exe --help`（**我们 exe**，先拷到 `C:\sp3`）⇒ **成功**，`C:\nsys\ourhelp.nsys-rep`（78,889 B），日志 grep **无 `bad conversion`**。
- **⇒ 结论**：`std::range_error: bad conversion` 的**真因 = 路径含中文**（旧 P0-d 的 cwd / 目标 exe / `-o` **全在中文路径**）；**不是我们 exe 的毛病**。**N1/C 从 `C:\sp3` 起跑、全 ASCII 即可。**

## 未验 / 待
- **prefill/decode 真数**（A3 prompt 仅 2 tok）：留 B 阶段（长 prompt，用 `--tokens-file`）。
- **ncu × CUDA graph 共存行为**（C1/C2 会撞，届时单列）。
- **`--expert-profile` 拷贝的合规性**：已报队长**待确认**。

## 落盘（绝对路径）
- 本件：`…\P3-20261016\回执\阶段A.md`
- `C:\sp3\{strata.exe, strata-vision.exe, BUILD.json, cublas64_13.dll, cublasLt64_13.dll, cudart64_13.dll, expert-profile.bin, strata-poly.exe, mamba2_cell_test.exe}`
- `C:\sp3\{smoke_a3.log…smoke_a3e.log, help.txt, flags.txt, cell_plain.log, cell_ncu.log}`
- `C:\nsys\{qa.nsys-rep, ourhelp.nsys-rep, qa_run.log, our_run.log}`
- `C:\ncu\{cell.ncu-rep, cell_report.txt, cell_ncu.log}`
