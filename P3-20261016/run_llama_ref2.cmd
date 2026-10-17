@echo off
rem P3: reference continuation from llama.cpp b11408 (loads this GGUF; no conversation template).
cd /d "E:\llama-cpp\b11408"
set PATH=E:\llama-cpp\b11408;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
.\llama-cli.exe -m "%M%" -p "The capital of France is" -n 8 --temp 0 -ngl 99 --verbose-prompt
echo EXIT=%errorlevel%
