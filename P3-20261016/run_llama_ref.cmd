@echo off
rem P3: the llama.cpp reference for the first-token reconciliation (same GGUF, greedy, verbose prompt).
cd /d "E:\llama-cpp"
set PATH=E:\llama-cpp;%PATH%
set M=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
.\llama-cli.exe -m "%M%" -p "The capital of France is" -n 6 --temp 0 -ngl 99 -no-cnv --verbose-prompt
echo EXIT=%errorlevel%
