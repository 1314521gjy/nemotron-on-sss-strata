@echo off
rem P3 determinism: canonical greedy (--no-draft --lookup 0).  usage: run_det.cmd <tail-skip>
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
build-p3\strata-poly.exe --model "%M%" --tokens 1784,8961,1307,5498,1395 --max-new 8 --prompt-batch 8 ^
  --vram-reserve-mib 2500 --tail-skip %1 --no-draft --lookup 0 --dump-layers --top-probs 5
echo EXIT=%errorlevel%
