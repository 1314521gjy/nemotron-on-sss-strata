@echo off
rem P2-20261016/build_run.cmd - one build, from a clean object state.
rem KP-UNI-0203: this machine's incremental C++ build is untrustworthy, so every *.obj/*.o is deleted first.
rem nvcc without -c leaves no persistent .obj here; the delete is the rule kept explicit anyway.
cd /d "%~dp0"
rem MSVC's cl.exe must be on PATH for nvcc; the toolchain here is VS18 BuildTools (Chinese MUI, KP-UNI-0203).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo CLEAN_OBJ= deleting *.obj *.o and the previous exe
del /q *.obj *.o mamba2_cell_test.exe >nul 2>nul
set NVCC=E:\cuda-13.3\bin\nvcc.exe
if not exist "%NVCC%" set NVCC=nvcc
echo BUILD: "%NVCC%" -std=c++17 -O2 -arch=sm_89
"%NVCC%" -std=c++17 -O2 -arch=sm_89 -I. mamba2_cell.cu mamba2_cell_test.cu -o mamba2_cell_test.exe
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
.\mamba2_cell_test.exe
echo EXIT=%errorlevel%
