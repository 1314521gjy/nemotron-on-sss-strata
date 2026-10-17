@echo off
rem P3 E1: build the strata-poly target only (kernels.cu: mamba2_scan state->shared, STRATA_M2_SCAN_SMEM).
setlocal
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake
set CMAKE="%VS%\CMake\bin\cmake.exe"
%CMAKE% --build build-p3 --target strata-poly
echo EXIT=%errorlevel%
