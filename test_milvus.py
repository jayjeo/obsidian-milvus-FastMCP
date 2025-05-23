#!/usr/bin/env python3
"""
간단한 Milvus 연결 테스트
"""

def test_milvus_connection():
    try:
        from pymilvus import connections, utility
        
        print("🔗 Milvus 서버 연결 시도 중...")
        connections.connect(
            alias="default",
            host='localhost',
            port='19530'
        )
        
        if connections.has_connection("default"):
            print("✅ Milvus 연결 성공!")
            
            # 서버 정보 확인
            try:
                collections = utility.list_collections()
                print(f"📊 기존 컬렉션 수: {len(collections)}")
                if collections:
                    for col in collections:
                        print(f"   - {col}")
                else:
                    print("   (컬렉션 없음)")
                
                print("🎉 Milvus 서버가 정상적으로 작동 중입니다!")
                return True
                
            except Exception as e:
                print(f"⚠️ 컬렉션 조회 중 오류: {e}")
                print("그러나 기본 연결은 성공했습니다.")
                return True
                
        else:
            print("❌ Milvus 연결 실패")
            return False
            
    except ImportError as e:
        print(f"❌ pymilvus 패키지를 찾을 수 없습니다: {e}")
        print("💡 해결방법: pip install pymilvus")
        return False
        
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        print("💡 해결방법:")
        print("1. Milvus 서버가 실행 중인지 확인")
        print("2. 포트 19530이 사용 가능한지 확인")
        print("3. 방화벽 설정 확인")
        return False

if __name__ == "__main__":
    print("="*50)
    print("    Milvus 연결 테스트")
    print("="*50)
    
    result = test_milvus_connection()
    
    print("\n" + "="*50)
    if result:
        print("✅ 테스트 완료: Milvus 서버 정상 작동")
        print("💡 이제 setup.py에서 2번(Milvus connection test)을 실행하세요!")
    else:
        print("❌ 테스트 실패: Milvus 서버 문제")
        print("💡 네트워크 설정을 다시 확인하거나 컨테이너를 재시작하세요")
    print("="*50)
