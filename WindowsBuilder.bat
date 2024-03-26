@ECHO off

if [%1] == [test] shift & goto :test
if [%1] == [install] shift & goto :install
if [%1] == [clean] shift & goto :clean

:parseArgs
if [%1] == [WORKERS] set NATTEN_N_WORKERS=%2 & shift & shift & goto :parseargs
if [%1] == [CUDA_ARCH] set NATTEN_CUDA_ARCH=%2 & shift & shift & goto :parseargs
if [%1] == [VERBOSE] set NATTEN_VERBOSE=%2 & shift & shift & goto :parseargs
goto :installContinue
:end

:test
echo "Testing NATTEN"
pip install -r requirements-dev.txt
pytest -v -x ./tests
goto :eof
:end

:install
goto :installStart
:end

:installStart
goto :parseargs
:end

:installFinalize
set NATTEN_N_WORKERS=
set NATTEN_CUDA_ARCH=
set NATTEN_VERBOSE=
goto :eof
:end

:installContinue
echo NATTEN_N_WORKERS: %NATTEN_N_WORKERS%
echo NATTEN_CUDA_ARCH: %NATTEN_CUDA_ARCH%
echo NATTEN_VERBOSE: %NATTEN_VERBOSE%
pip install -r requirements.txt
python setup.py install
goto :installFinalize
:end

:clean
echo Cleaning up
echo "Removing %CD%\build"
del %CD%\build
goto :eof
:end