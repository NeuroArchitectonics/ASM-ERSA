@echo off
rem train_m4 + Apollo solver (ASCII only on purpose).
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
   /I "%ROOT%." /I "%PRJ%\3911~1" /I "%PRJ%\ASM-ERSA-Design\vk" /I "%PRJ%\ASM-ERSA-Design\vk\spv" /I "%PRJ%\spv" /I "%VKSDK%\Include" ^
   "%ROOT%train_m4.cpp" "%PRJ%\3911~1\ApolloSolver.cpp" ^
   /Fe:"%ROOT%build\train_m4_apollo.exe" /Fo:%ROOT%build\ ^
   /link /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib
if errorlevel 1 ( echo [build] compile FAILED & exit /b 1 )
echo [build] OK: %ROOT%build\train_m4_apollo.exe
exit /b 0
