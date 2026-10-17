@echo off
rem P3 e2e: configure and build strata-poly with the local prism ggml (team-lead's plan b).
rem KP-UNI-0203: the build tree here is fresh; a rebuild run must still clear objects.
setlocal
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake
set CMAKE="%VS%\CMake\bin\cmake.exe"
set NINJA="%VS%\Ninja\ninja.exe"
if "%1"=="build" goto build
echo CONFIGURE
%CMAKE% -S . -B build-p3 -G Ninja -DCMAKE_MAKE_PROGRAM=%NINJA% -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER=E:\cuda-13.3\bin\nvcc.exe -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DSTRATA_GGML_DIR=E:\llama-cpp-pin
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
echo CONFIGURE_OK
exit /b 0
:build
echo BUILD strata-poly
%CMAKE% --build build-p3 --target strata-poly -j
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
