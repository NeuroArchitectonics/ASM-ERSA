@echo off
setlocal
chcp 65001 >nul
if "%VKSDK%"=="" set "VKSDK=C:\VulkanSDK\1.4.341.1"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1
cl /nologo /EHsc /std:c++20 /utf-8 /O2 /W3 /MD /D_CRT_SECURE_NO_WARNINGS /I"D:\智能体工作区\ASM-ERSA" /I"D:\智能体工作区\ASM-ERSA\ASM-ERSA-Design\vk" /I"%VKSDK%\Include" "cpp\recall_main.cpp" /Fe:"cpp\build\recall_main.exe" /Fo:"cpp\build\recall_main.obj" /link /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib
if errorlevel 1 exit /b 1
".\cpp\build\recall_main.exe" "cpp\spv"
