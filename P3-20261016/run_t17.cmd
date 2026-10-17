@echo off
rem P3 t17: --expert-profile with a PURE ASCII path (the model dir), to test the non-ASCII-path hypothesis.
rem ASCII only in this file's arguments; the fact that the .cmd is UTF-8 does not matter for ASCII text.
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
set P=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\expert-profile-p3.txt
echo === t17 profile -> %P% ===
build-p3\strata-poly.exe --model "%M%" --tokens 1784,8961,1307,5498,1395 --max-new 4 --prompt-batch 8 ^
  --vram-reserve-mib 2500 --tail-skip 0 --expert-profile "%P%"
echo EXIT=%errorlevel%
