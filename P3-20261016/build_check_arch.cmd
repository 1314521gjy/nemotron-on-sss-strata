@echo off
rem P3: syntax-check nemotronhmoe.cpp and engine.cpp alone, from a clean object state (KP-UNI-0203).
rem ggml.h comes from a llama.cpp tree (the real build pins its own via STRATA_ORACLE_SOURCE_DIR).
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set D=%~dp0
set GGML=E:\llama-cpp\prism-b10687\ggml\include
del /q "%D%nemotronhmoe.obj" "%D%engine.obj" "%D%cpu_experts.obj" "%D%kernels_check.obj" >nul 2>nul
echo CLEAN_OBJ= done
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include -c src/poly/nemotronhmoe.cpp -o "%D%nemotronhmoe.obj"
if errorlevel 1 ( echo ARCH_BUILD_FAILED & exit /b 1 )
echo ARCH_OK
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include -I "%GGML%" -c src/poly/engine.cpp -o "%D%engine.obj"
if errorlevel 1 ( echo ENGINE_BUILD_FAILED & exit /b 1 )
echo ENGINE_OK
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include -I "%GGML%" -c src/poly/cpu_experts.cpp -o "%D%cpu_experts.obj"
if errorlevel 1 ( echo CPUEXPERTS_BUILD_FAILED & exit /b 1 )
echo CPUEXPERTS_OK
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include -c src/poly/kernels.cu -o "%D%kernels_check.obj"
if errorlevel 1 ( echo KERNELS_BUILD_FAILED & exit /b 1 )
echo KERNELS_OK
