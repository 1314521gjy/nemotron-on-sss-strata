@echo off
rem P3 e2e smoke: a short prompt through the real Nemotron GGUF, with the U3 memory readings.
setlocal
cd /d "%~dp0..\PolyStrata"
set PATH=E:\llama-cpp;E:\cuda-13.3\bin;%PATH%
set MODEL=E:\models\Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_K_M-00001-of-00002.gguf
echo === U3 memory ===
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
powershell -NoProfile -Command "$o=Get-CimInstance Win32_OperatingSystem; $c=Get-CimInstance Win32_ComputerSystem; 'usable_physical_GB={0:N1}  CommitLimit_GB={1:N1}  CommitAvail_GB={2:N1}' -f ($c.TotalPhysicalMemory/1GB),($o.TotalVirtualMemorySize/1MB),(($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB)"
echo === run: tokens 1,2,3 max-new 4 ===
build-p3\strata-poly.exe --model "%MODEL%" --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14 --max-new 2 --vram-reserve-mib 2500 --top-probs 8
echo EXIT=%errorlevel%
