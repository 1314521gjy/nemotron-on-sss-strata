@echo off
set VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo VCVARS_FAIL & exit /b 1)
set CM=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NJ=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set SRC=E:\llama-cpp-pin
set BLD=E:\llama-cpp-pin\build-cpu-eval
echo === CONFIGURE ===
"%CM%" -G Ninja -B "%BLD%" -S "%SRC%" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
if errorlevel 1 (echo CONFIGURE_FAIL & exit /b 1)
echo === BUILD ===
"%NJ%" -C "%BLD%" llama-eval-callback
if errorlevel 1 (echo BUILD_FAIL & exit /b 1)
echo === DONE ===
dir "%BLD%\bin\llama-eval-callback.exe"
