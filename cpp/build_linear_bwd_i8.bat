@echo off
setlocal
chcp 65001 >nul
if "%VKSDK%"=="" set "VKSDK=C:\VulkanSDK\1.4.341.1"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /EHsc /utf-8 /O2 /MD /D_CRT_SECURE_NO_WARNINGS /I"D:\智能体工作区\ASM-ERSA" /I"D:\智能体工作区\ASM-ERSA\cpp" /I"D:\智能体工作区\ASM-ERSA\vk" /I"%VKSDK%\Include" "cpp\linear_bwd_i8_test.cpp" /Fe:"cpp\build\linear_bwd_i8_test.exe" /link /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib
if errorlevel 1 exit /b 1
".\cpp\build\linear_bwd_i8_test.exe" "cpp\spv"
