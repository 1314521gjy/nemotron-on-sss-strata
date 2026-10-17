@echo off
rem P3 t5: default path WITH draft (exercises window()/keep() = the recurrent-state snapshot path).
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
echo === engine draft max-new=8 (snap path) ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens 1784,8961,1307,5498,1395 --max-new 8 --prompt-batch 8 ^
  --vram-reserve-mib 2500 --tail-skip 0 --top-probs 5
echo EXIT=%errorlevel%
