@echo off
rem P3: syntax-check poly kernels.cu alone, from a clean object state (KP-UNI-0203).
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set OUT=%~dp0kernels_check.obj
del /q "%OUT%" >nul 2>nul
echo CLEAN_OBJ= done, building %OUT%
"E:\cuda-13.3\bin\nvcc.exe" -std=c++20 -arch=sm_89 -I include -c src/poly/kernels.cu -o "%OUT%"
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
