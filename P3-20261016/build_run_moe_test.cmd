@echo off
rem P3: build and run the non-gated relu2 expert block test with a clean object state (KP-UNI-0203).
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set D=%~dp0
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
del /q "%D%poly_nemotron_moe_test.exe" >nul 2>nul
echo CLEAN done
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include src/poly/kernels.cu src/poly/nemotron_moe_test.cu -o "%D%poly_nemotron_moe_test.exe" -lcublas
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
"%D%poly_nemotron_moe_test.exe"
echo EXIT=%errorlevel%
