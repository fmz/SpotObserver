@echo off
REM Host-side test harness for StreamingONNXModel. Runs without a GPU or the CUDA
REM toolkit: it extracts the real class and the real kernels from src/ and compiles
REM them against the real ONNX Runtime headers, stubbing only CUDA allocation and
REM the CUDA execution provider.
REM
REM Requires: MSVC, python with onnx + numpy, extern/onnxruntime-win-x64-gpu-1.22.0.
REM Usage: run_all.bat [path-to-python]

setlocal
set HERE=%~dp0
set REPO=%HERE%..\..
set ORT=%REPO%\extern\onnxruntime-win-x64-gpu-1.22.0
set PY=%~1
if "%PY%"=="" set PY=python

if not exist "%ORT%\include\onnxruntime_cxx_api.h" (
  echo ERROR: ONNX Runtime not found at %ORT%
  echo Download onnxruntime-win-x64-gpu-1.22.0 and extract it under extern\.
  exit /b 1
)

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo ERROR: vcvars64.bat failed & exit /b 1 )

echo === generating harnesses from src/ ===
"%PY%" "%HERE%make_stub_model.py"        || exit /b 1
"%PY%" "%HERE%make_stub_model.py" --fp16 || exit /b 1
"%PY%" "%HERE%gen_ort_check.py"          || exit /b 1
"%PY%" "%HERE%gen_kernel_test.py"        || exit /b 1
"%PY%" "%HERE%gen_integ_test.py"         || exit /b 1

set B=%HERE%build

echo.
echo === 1/3 ORT API compile check ===
cl /c /nologo /std:c++20 /EHsc /permissive- /I "%ORT%\include" ^
   /Fo:"%B%\ort_check.obj" "%B%\ort_check.cpp" || exit /b 1

echo.
echo === 2/3 kernel math tests ===
cl /nologo /std:c++20 /EHsc /O2 /fp:precise ^
   /Fe:"%B%\kern_test.exe" /Fo:"%B%\kern_test.obj" "%B%\kern_test.cpp" || exit /b 1
"%B%\kern_test.exe" || exit /b 1

echo.
echo === 3/3 integration tests (fp32 + fp16 caches) ===
if not exist "%B%\onnxruntime.dll" copy /y "%ORT%\lib\onnxruntime.dll" "%B%\" >nul
cl /nologo /std:c++20 /EHsc /O2 /fp:precise /I "%ORT%\include" ^
   /Fe:"%B%\integ_test.exe" /Fo:"%B%\integ_test.obj" "%B%\integ_test.cpp" ^
   /link "%ORT%\lib\onnxruntime.lib" || exit /b 1
"%B%\integ_test.exe" || exit /b 1

echo.
echo ALL HARNESS SUITES PASSED
endlocal
