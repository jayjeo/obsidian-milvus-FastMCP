@echo off
echo ================================================================
echo          Direct Conda NumPy Fix (Miniconda 환경)
echo ================================================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Miniconda가 이미 설치되어 있으므로 conda를 직접 사용합니다.
echo 이 스크립트는 다음을 수행합니다:
echo 1. 현재 패키지 상태 확인
echo 2. pip 대신 conda로 NumPy 설치
echo 3. sentence-transformers 재설치
echo.

echo 계속하려면 아무 키나 누르세요...
pause

echo.
echo ================================================================
echo Step 1: 현재 환경 확인
echo ================================================================

echo Python 위치:
where python
echo.

echo 현재 NumPy 상태:
python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>nul || echo NumPy not installed

echo 현재 conda 환경:
conda info --envs

echo.
echo ================================================================
echo Step 2: 기존 패키지 제거 (conda 사용)
echo ================================================================

echo conda로 패키지 제거 중...
conda remove -y numpy sentence-transformers --force-remove
echo.

echo pip로도 제거 (혹시 남아있을 수 있음)...
python -m pip uninstall -y numpy sentence-transformers

echo.
echo ================================================================
echo Step 3: conda로 NumPy 설치
echo ================================================================

echo 기본 채널에서 NumPy 설치 시도...
conda install -y numpy=1.26.4

if %errorlevel% neq 0 (
    echo.
    echo 기본 채널 실패, conda-forge 채널 시도...
    conda install -y -c conda-forge numpy=1.26.4
    
    if %errorlevel% neq 0 (
        echo.
        echo 특정 버전 실패, 호환 버전 범위로 시도...
        conda install -y -c conda-forge "numpy>=1.21.0,<2.0.0"
        
        if %errorlevel% neq 0 (
            echo ERROR: conda로도 NumPy 설치 실패
            echo.
            echo 대안: 새로운 conda 환경 생성
            echo conda create -n numpy_fix python=3.9 numpy=1.26.4
            echo conda activate numpy_fix
            pause
            exit /b 1
        )
    )
)

echo.
echo ================================================================
echo Step 4: sentence-transformers 설치
echo ================================================================

echo conda로 sentence-transformers 설치 시도...
conda install -y -c conda-forge sentence-transformers

if %errorlevel% neq 0 (
    echo.
    echo conda 설치 실패, conda 환경에서 pip 사용...
    python -m pip install sentence-transformers>=2.2.2
    
    if %errorlevel% neq 0 (
        echo ERROR: sentence-transformers 설치 실패
        pause
        exit /b 1
    )
)

echo.
echo ================================================================
echo Step 5: 설치 확인
echo ================================================================

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import numpy; print(f'NumPy location: {numpy.__file__}')"
python -c "import sentence_transformers; print('Sentence Transformers 성공적으로 import됨')"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo 성공! NumPy 호환성 문제가 해결되었습니다!
    echo ================================================================
    echo.
    echo 이제 incremental embedding 옵션을 다시 실행할 수 있습니다.
    echo.
    echo 설치된 패키지 정보:
    conda list numpy
    conda list sentence-transformers
) else (
    echo.
    echo ERROR: 설치 확인 실패
    echo 위의 오류 메시지를 확인해주세요.
)

echo.
pause