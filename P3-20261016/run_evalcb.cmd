@echo off
rem P3 G1-c: llama.cpp's own per-node ground truth (CPU-only, -ngl 0).  See 读数索引-P3.md.
setlocal
cd /d "E:\llama-cpp-pin\build-cpu-eval\bin"
set PATH=E:\llama-cpp-pin\build-cpu-eval\bin;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
.\llama-eval-callback.exe -m "%M%" -p "The capital of France is" -n 1 --seed 42 -ngl 0 --temp 0
echo EXIT=%errorlevel%
