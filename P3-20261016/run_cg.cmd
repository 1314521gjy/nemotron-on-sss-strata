@echo off
rem P3 t2 grouping-key experiment.  usage: run_cg.cmd <mode>
rem   mode = old    -> STRATA_CG_GATE=1 AND STRATA_SH_XQ=1   (both bugs, baseline)
rem   mode = cgold  -> STRATA_CG_GATE=1 only                 (fix (2) only; (1) still buggy)
rem   mode = shold  -> STRATA_SH_XQ=1 only                   ((1) fixed; (2) still buggy)
rem   mode = fix    -> neither                              (both fixed)
rem Canonical t2 run: --prompt-batch 8 --tail-skip 0 --cpu-share 0.55 --expert-swaps 64 --expert-profile.
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set STRATA_DUMP_LATENT=1
set STRATA_PROBE_HOST=1
set STRATA_DUMP_ROUTE=1
if "%1"=="old"   set STRATA_CG_GATE=1
if "%1"=="old"   set STRATA_SH_XQ=1
if "%1"=="cgold" set STRATA_CG_GATE=1
if "%1"=="shold" set STRATA_SH_XQ=1
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
echo === mode=%1 ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens 1784,8961,1307,5498,1395 --max-new 1 --prompt-batch 8 ^
  --vram-reserve-mib 2500 --cpu-share 0.55 --expert-swaps 64 --tail-skip 0 ^
  --expert-profile "E:\视频\_flashnext\experiments\20261015-nemotron-on-strata\P3-20261016\profile_p3.txt" ^
  --dump-layers --top-probs 5
echo EXIT=%errorlevel%
