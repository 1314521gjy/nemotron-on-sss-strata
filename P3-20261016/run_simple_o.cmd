@echo off
rem P3 multi-token reference (llama-simple, greedy, CPU) with STDOUT separated from stderr.
rem usage: run_simple_o.cmd <n> <outfile>
setlocal
cd /d "E:\llama-cpp-pin\build-cpu-eval\bin"
set PATH=%CD%;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
.\llama-simple.exe -m "%M%" -n %1 -ngl 0 "The capital of France is" 1>"%~dp0%~2" 2>"%~dp0%~2.err"
echo EXIT=%errorlevel%
