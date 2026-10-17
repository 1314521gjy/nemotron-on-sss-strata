@echo off
rem P3 multi-token reference: llama-simple (greedy, CPU-only, from the pin).  usage: run_simple.cmd <n>
setlocal
cd /d "E:\llama-cpp-pin\build-cpu-eval\bin"
set PATH=%CD%;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
set N=%1
if "%N%"=="" set N=8
echo === llama-simple n=%N% ===
.\llama-simple.exe -m "%M%" -n %N% -ngl 0 "The capital of France is"
echo.
echo EXIT=%errorlevel%
