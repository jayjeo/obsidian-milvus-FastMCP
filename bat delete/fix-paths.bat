@echo off
echo ================================================================
echo          Milvus 경로 문제 해결 스크립트
echo ================================================================
echo.
echo 이 스크립트는 다음을 수행합니다:
echo 1. 모든 기존 Milvus 컨테이너 및 볼륨 완전 삭제
echo 2. 올바른 위치에 새로 설치
echo.
echo 현재 프로젝트 디렉토리: %CD%
echo 예상 데이터 위치:
echo   - 컨테이너 데이터: %CD%\volumes
echo   - 임베딩 데이터: %CD%\MilvusData
echo.
set /p confirm="계속하시겠습니까? (y/n): "
if not "%confirm%"=="y" (
    echo 취소되었습니다.
    pause
    exit /b 0
)

echo.
echo [1/6] 모든 Milvus 컨테이너 중지 및 삭제...
podman stop milvus-standalone milvus-minio milvus-etcd 2>nul
podman rm -f milvus-standalone milvus-minio milvus-etcd 2>nul

echo.
echo [2/6] 모든 Milvus 볼륨 삭제...
podman volume rm -f milvus-etcd-data milvus-minio-data milvus-db-data 2>nul

echo.
echo [3/6] 네트워크 정리...
podman network rm -f milvus milvus-network 2>nul

echo.
echo [4/6] 시스템 정리...
podman system prune -f --volumes 2>nul

echo.
echo [5/6] 기존 로컬 데이터 폴더 확인 및 정리...
if exist "MilvusData" (
    echo 기존 MilvusData 폴더를 백업합니다...
    if exist "MilvusData_backup" rmdir /s /q "MilvusData_backup" 2>nul
    move "MilvusData" "MilvusData_backup" 2>nul
    echo MilvusData 폴더가 MilvusData_backup으로 백업되었습니다.
)

if exist "volumes" (
    echo 기존 volumes 폴더를 백업합니다...
    if exist "volumes_backup" rmdir /s /q "volumes_backup" 2>nul
    move "volumes" "volumes_backup" 2>nul
    echo volumes 폴더가 volumes_backup으로 백업되었습니다.
)

echo.
echo [6/6] 새로운 폴더 생성...
mkdir "MilvusData" 2>nul
mkdir "volumes" 2>nul
mkdir "MilvusData\minio" 2>nul
mkdir "MilvusData\milvus" 2>nul
mkdir "volumes\etcd" 2>nul

echo.
echo ================================================================
echo                  정리 완료!
echo ================================================================
echo.
echo 다음 단계:
echo 1. start-milvus.bat 실행
echo 2. 또는 python main.py 실행 후 옵션 2 선택
echo.
echo 데이터가 다음 위치에 설치될 것입니다:
echo   - 컨테이너 데이터: %CD%\volumes
echo   - 임베딩 데이터: %CD%\MilvusData
echo.
echo 기존 데이터는 다음 위치에 백업되었습니다:
echo   - MilvusData_backup (있는 경우)
echo   - volumes_backup (있는 경우)
echo.
pause