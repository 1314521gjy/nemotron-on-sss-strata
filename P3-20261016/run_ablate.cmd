@echo off
rem P3 ablation: a MoE layer contributes nothing (pure residual).  usage: run_ablate.cmd <mode> [tokens]
rem   mode = moe  -> STRATA_ABLATE_MOE=1
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
if "%1"=="moe" set STRATA_ABLATE_MOE=1
if "%1"=="attn" set STRATA_ABLATE_ATTN=1
if "%1"=="mamba" set STRATA_ABLATE_MAMBA=1
if "%1"=="moe2" set STRATA_ABLATE_MOE2=1
if "%1"=="shared" set STRATA_ABLATE_SHARED=1
echo === ablation=%1 ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens 10,3263,1010,1784,8961,1307,5498,1395,11,1010,10,1503,19464,1010 --max-new 8 ^
  --prompt-batch 8 --vram-reserve-mib 2500 --top-probs 8
echo EXIT=%errorlevel%
