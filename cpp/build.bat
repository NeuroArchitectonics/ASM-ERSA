@echo off
rem ASM-ERSA C++ fused main model build (ASCII only).
rem usage: build.bat
setlocal enabledelayedexpansion
chcp 65001 >nul

if "%VKSDK%"=="" set "VKSDK=C:\VulkanSDK\1.4.341.1"
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set "ROOT=%~dp0"
set "PRJ=%ROOT%.."

if not exist "%VCVARS%" (
    echo [build] vcvars64 not found: %VCVARS%
    exit /b 1
)
if not exist "%VKSDK%\Bin\glslangValidator.exe" (
    echo [build] Vulkan SDK not found: %VKSDK%
    exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & exit /b 1 )

if not exist "%ROOT%spv" mkdir "%ROOT%spv"
if not exist "%ROOT%build" mkdir "%ROOT%build"

set "CXXFLAGS=/nologo /EHsc /std:c++17 /utf-8 /O2 /W3 /MD /D_CRT_SECURE_NO_WARNINGS /I"%ROOT%." /I"%PRJ%" /I"%PRJ%\vk" /I"%VKSDK%\Include""
set "LDFLAGS=/nologo /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib"

set "SHADER_TOOL=%PRJ%\ASM-ERSA-Design\vk\build\shader_tool.exe"
if not exist "%SHADER_TOOL%" (
    echo [build] shader_tool
    cl %CXXFLAGS% /c "%PRJ%\ASM-ERSA-Design\vk\tools\shader_tool.cpp" /Fo:"%ROOT%build\shader_tool.obj"
    if errorlevel 1 ( echo [build] shader_tool compile FAILED & exit /b 1 )
    link /nologo /OUT:"%ROOT%build\shader_tool.exe" "%ROOT%build\shader_tool.obj" %LDFLAGS%
    if errorlevel 1 ( echo [build] shader_tool link FAILED & exit /b 1 )
    set "SHADER_TOOL=%ROOT%build\shader_tool.exe"
)

echo [build] compiling fused shaders
"%SHADER_TOOL%" "%ROOT%glsl" "%ROOT%spv" "%VKSDK%\Bin\glslangValidator.exe" "%VKSDK%\Bin\spirv-opt.exe"
if errorlevel 1 ( echo [build] shader compile FAILED & exit /b 1 )

echo [build] copying runtime spv
copy /Y "%PRJ%\spv\linear_fwd_int8_f32.spv" "%ROOT%spv\" >nul
if exist "%ROOT%spv\linear_fwd.spv" del /Q "%ROOT%spv\linear_fwd.spv"
if exist "%ROOT%spv\rmsnorm_fwd.spv" del /Q "%ROOT%spv\rmsnorm_fwd.spv"

echo [build] compiling C++
cl %CXXFLAGS% /c "%ROOT%asm_ersa_model.cpp" /Fo:"%ROOT%build\asm_ersa_model.obj"
if errorlevel 1 ( echo [build] asm_ersa_model FAILED & exit /b 1 )
cl %CXXFLAGS% /c "%ROOT%cpu_reference.cpp" /Fo:"%ROOT%build\cpu_reference.obj"
if errorlevel 1 ( echo [build] cpu_reference FAILED & exit /b 1 )
cl %CXXFLAGS% /c "%ROOT%main.cpp" /Fo:"%ROOT%build\main.obj"
if errorlevel 1 ( echo [build] main FAILED & exit /b 1 )

echo [build] linking
link /nologo /OUT:"%ROOT%build\asm_ersa_model.exe" ^
    "%ROOT%build\asm_ersa_model.obj" "%ROOT%build\cpu_reference.obj" "%ROOT%build\main.obj" %LDFLAGS%
if errorlevel 1 ( echo [build] link FAILED & exit /b 1 )

echo [run] asm_ersa_model
pushd "%ROOT%"
"%ROOT%build\asm_ersa_model.exe" "%ROOT%spv"
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" ( echo [run] FAILED & exit /b %RC% )
echo [run] PASS
exit /b 0
