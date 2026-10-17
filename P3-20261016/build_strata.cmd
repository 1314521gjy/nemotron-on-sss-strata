@echo off
setlocal
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio8\BuildTools\VC\Auxiliary\Buildcvars64.bat" >/dev/null 2>&1
set VS=C:\Program Files (x86)\Microsoft Visual Studio8\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake
set CMAKE="%VS%\CMakein\cmake.exe"
%CMAKE% --build build-p3 --target strata
echo EXIT=%errorlevel%
