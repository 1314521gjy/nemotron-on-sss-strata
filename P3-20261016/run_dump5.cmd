@echo off
rem P3 macro check: one token, dump every stage.  blk=0 attn_in is the normed embedding.
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set STRATA_DUMP_ROUTE=1
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
build-p3\strata-poly.exe --model "%MODEL%" --tokens 1784,8961,1307,5498,1395 --max-new 1 --prompt-batch 8 ^
  --vram-reserve-mib 2500 --dump-layers & set STRATA_DUMP_ROUTE=1
echo EXIT=%errorlevel%
