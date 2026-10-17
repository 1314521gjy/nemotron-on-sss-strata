@echo off
rem P3: build llama-cli from the pinned source (CPU-only, no GPU).  Same tree as build_evalcb.
set VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo VCVARS_FAIL & exit /b 1)
set NJ=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set BLD=E:\llama-cpp-pin\build-cpu-eval
"%NJ%" -C "%BLD%" llama-simple
if errorlevel 1 (echo CLIBUILD_FAIL & exit /b 1)
echo === DONE ===
dir "%BLD%\bin\llama-simple.exe"
