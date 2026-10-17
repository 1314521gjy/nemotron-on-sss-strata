@echo off
rem P3 second run: the streaming switches on (expert-profile + cpu-share + swaps + reserve).
rem usage: run_e2e_opt.cmd <prompt_ids> [max_new]
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
set PROF=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\expert-profile-p3.json
if "%2"=="" (set N=16) else (set N=%2)
echo === U3 memory ===
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
echo === run: switches on, tokens %1, max-new %N% ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens %1 --max-new %N% ^
  --expert-profile "%PROF%" --cpu-share 0.4 --expert-swaps 64 --vram-reserve-mib 2500 --tail-skip 0
echo EXIT=%errorlevel%
