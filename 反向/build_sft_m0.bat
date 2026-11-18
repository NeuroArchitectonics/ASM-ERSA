@echo off
rem SFT data stream + M0 VK trainer (ASCII only).
setlocal
chcp 65001 >nul

if "%VKSDK%"=="" set "VKSDK=C:\VulkanSDK\1.4.341.1"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "ROOT=%~dp0"
set "PRJ=%ROOT%.."

if not exist "%VCVARS%" ( echo [build] vcvars64 not found & exit /b 1 )
call "%VCVARS%" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & exit /b 1 )

if not exist "%ROOT%build" mkdir "%ROOT%build"

cl /nologo /std:c++20 /EHsc /utf-8 /O2 /MD /D_CRT_SECURE_NO_WARNINGS ^
   /I "%ROOT%." /I "%PRJ%\ASM-ERSA-Design\vk" /I "%PRJ%\spv" /I "%VKSDK%\Include" ^
   "%ROOT%train_sft_m0.cpp" ^
   /Fe:"%ROOT%build\train_sft_m0.exe" /Fo:"%ROOT%build\train_sft_m0.obj" ^
   /link /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib
if errorlevel 1 ( echo [build] compile FAILED & exit /b 1 )

pushd "%PRJ%"
"%ROOT%build\train_sft_m0.exe" "spv" ".cache/sft_110k_head100.jsonl" 100
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
