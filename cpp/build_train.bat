@echo off
rem M-B2 main-model VK trainer build (ASCII only).
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

rem backward .spv are produced in the shared 反向/spv directory
for %%F in (linear_bwd_dx linear_bwd_dwdb rmsnorm_bwd_dx rmsnorm_bwd_dw ^
            fused_ce_bwd embedding_bwd moe_bwd_dx moe_bwd_dw_expert ^
            moe_bwd_dw_shared adamw_update) do (
    copy /Y "%PRJ%\spv\%%F.spv" "%ROOT%spv\%%F.spv" >nul
    if errorlevel 1 ( echo [build] missing backward spv: %%F & exit /b 1 )
)

cl /nologo /std:c++20 /EHsc /utf-8 /O2 /MD /D_CRT_SECURE_NO_WARNINGS ^
   /I "%ROOT%." /I "%ROOT%spv" /I "%PRJ%\spv" /I "%PRJ%" /I "%PRJ%\3911~1" /I "%PRJ%\vk" /I "%VKSDK%\Include" ^
   "%ROOT%train_main.cpp" "%ROOT%asm_ersa_model.cpp" "%ROOT%cpu_reference.cpp" "%PRJ%\3911~1\ApolloMini.cpp" "%PRJ%\3911~1\ApolloMiniI8.cpp" ^
   /Fe:"%ROOT%build\train_main.exe" ^
   /Fo:"%ROOT%build\\" ^
   /link /LIBPATH:"%VKSDK%\Lib" vulkan-1.lib
if errorlevel 1 ( echo [build] compile FAILED & exit /b 1 )

pushd "%PRJ%"
"%ROOT%build\train_main.exe" "cpp/spv" %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
