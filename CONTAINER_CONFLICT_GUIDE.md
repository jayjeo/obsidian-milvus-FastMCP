# 🔧 컨테이너 충돌 문제 해결 가이드

이 문제는 Milvus 컨테이너들이 완전히 정리되지 않아 새 컨테이너 시작 시 이름 충돌이 발생하는 것입니다.

## 🚨 즉시 해결 방법

### 1단계: 응급 정리 실행
```bash
emergency-reset.bat
```
이 스크립트를 실행하면 모든 컨테이너와 데이터가 강제로 삭제됩니다.

### 2단계: 정상 재시작
```bash
python main.py
```
프로그램을 다시 실행합니다.

### 3단계: 전체 임베딩 실행
- 옵션 2 선택 (전체 embedding)
- "erase all?" 질문에 Y 선택

## 🔍 문제 진단

문제가 지속되면 `diagnose-podman.bat`를 실행하여 시스템 상태를 확인하세요.

## 📋 향후 방지법

1. **정상 종료**: Ctrl+C로 프로그램을 종료할 때 컨테이너가 자동으로 정리됩니다.

2. **수동 정리**: 필요시 다음 명령으로 수동 정리:
   ```bash
   podman stop milvus-standalone milvus-minio milvus-etcd
   podman rm milvus-standalone milvus-minio milvus-etcd
   ```

3. **완전 정리**: 모든 것을 정리하려면:
   ```bash
   emergency-reset.bat
   ```

## ❓ 여전히 문제가 있다면

1. `diagnose-podman.bat` 실행하여 상태 확인
2. Podman Desktop 재시작
3. Windows 재부팅 후 다시 시도

---
**주의**: `emergency-reset.bat`는 모든 Podman 컨테이너를 삭제하므로 다른 프로젝트가 있다면 주의하세요.
