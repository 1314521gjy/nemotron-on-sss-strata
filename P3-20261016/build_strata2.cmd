@echo off
rem P3 P0-b: build the strata target only (do not touch strata-poly).
setlocal
cd /d "%~dp0..\PolyStrata"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake
set CMAKE="%VS%\CMake\bin\cmake.exe"
%CMAKE% --build build-p3 --target strata
echo EXIT=%errorlevel%
