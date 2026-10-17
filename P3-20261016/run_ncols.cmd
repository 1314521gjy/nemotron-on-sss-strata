@echo off
rem P3 t13: 14-token prompt (n>8).  usage: run_ncols.cmd <extra args...>  (pass --prompt-batch 8 to chunk)
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
echo === 14-token prompt, extra=%* ===
build-p3\strata-poly.exe --model "%M%" --tokens 10,3263,1010,1784,8961,1307,5498,1395,11,1010,10,1503,19464,1010 --max-new 8 ^
  --vram-reserve-mib 2500 --top-probs 5 %*
echo EXIT=%errorlevel%
