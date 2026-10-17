@echo off
rem P3 multi-token engine run (greedy = --no-draft --lookup 0).  usage: run_mt.cmd <n>
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set N=%1
if "%N%"=="" set N=8
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
echo === engine max-new=%N% no-draft lookup0 ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens 1784,8961,1307,5498,1395 --max-new %N% --prompt-batch 8 ^
  --vram-reserve-mib 2500 --tail-skip 0 --no-draft --lookup 0 --top-probs 5
echo EXIT=%errorlevel%
