@echo off
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3in;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
build-p3\strata-poly.exe --model "%M%" --tokens 1784,8961,1307,5498,1395 --max-new 4 --prompt-batch 8 --vram-reserve-mib 2500 --tail-skip 0 --expert-profile "E:\视频\_flashnextxperiments�61015-nemotron-on-strata\P3-20261016\profile_neg.txt"
echo EXIT=%errorlevel%
