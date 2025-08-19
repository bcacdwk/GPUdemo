@echo off
REM =============================================================================
REM CUDA Compile Script - Windows Version
REM Function: Auto setup VS environment, compile CUDA program, and run result
REM Usage: compile_cuda.bat <source.cu> [output_name] [compile_flags]
REM Example: compile_cuda.bat vector_add.cu
REM          compile_cuda.bat vector_add.cu my_program "-O3"
REM =============================================================================

REM Set UTF-8 encoding for proper Chinese character display
chcp 65001 >nul 2>&1

setlocal EnableDelayedExpansion

REM Check if source file parameter is provided
if "%~1"=="" (
    echo Error: Please provide CUDA source file
    echo Usage: %0 ^<source.cu^> [output_name] [compile_flags]
    echo Example: %0 vector_add.cu
    echo          %0 vector_add.cu my_program "-O3"
    pause
    exit /b 1
)

REM Get input parameters
set "SOURCE_FILE=%~1"
set "OUTPUT_FILE=%~2"
set "COMPILE_FLAGS=%~3"

REM Check if source file exists
if not exist "%SOURCE_FILE%" (
    echo Error: File "%SOURCE_FILE%" does not exist!
    pause
    exit /b 1
)

REM If no output file specified, use source filename without extension
if "%OUTPUT_FILE%"=="" (
    for /f "delims=" %%i in ("%SOURCE_FILE%") do set "OUTPUT_FILE=%%~ni"
)

REM If no compile flags specified, use default debug flags
if "%COMPILE_FLAGS%"=="" (
    set "COMPILE_FLAGS=-g -G"
)

echo.
echo ========================================
echo CUDA Compilation Script
echo ========================================
echo Source File: %SOURCE_FILE%
echo Output File: %OUTPUT_FILE%.exe
echo Compile Options: %COMPILE_FLAGS%
echo ========================================
echo.

REM Setup Visual Studio environment
echo [1/4] Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if errorlevel 1 (
    echo Error: Cannot setup Visual Studio environment
    echo Please ensure Visual Studio 2022 Community is installed
    pause
    exit /b 1
)
echo     √ Visual Studio environment setup successful

REM Compile CUDA program
echo [2/4] Compiling CUDA program...
nvcc %COMPILE_FLAGS% -o "%OUTPUT_FILE%" "%SOURCE_FILE%"

if errorlevel 1 (
    echo     × Compilation failed!
    echo Please check source code or compilation options
    pause
    exit /b 1
) else (
    echo     √ Compilation successful!
)

REM Check if executable file is generated
echo [3/4] Checking generated executable...
if exist "%OUTPUT_FILE%.exe" (
    echo     √ Executable "%OUTPUT_FILE%.exe" generated successfully
    echo [4/4] Running program...
    echo.
    echo ================== Program Output ==================
    "%OUTPUT_FILE%.exe"
    set "EXIT_CODE=!ERRORLEVEL!"
    echo ================================================
    echo.
    if !EXIT_CODE! equ 0 (
        echo √ Program executed successfully ^(Exit code: !EXIT_CODE!^)
    ) else (
        echo × Program execution failed ^(Exit code: !EXIT_CODE!^)
    )
) else (
    echo     × Executable generation failed
)

echo.
echo Compilation and execution completed.
pause
