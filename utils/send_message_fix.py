def send_message(self, chat_id, query):
    """질문에 대한 컨텍스트를 검색하고 메시지 전송"""
    # 검색 전 로깅
    print(f"\n[OpenWebUI] 검색 시작: '{query}'")
    
    # 임베딩 데이터베이스 연결 확인
    if not self.milvus_manager.is_connected():
        try:
            print("[OpenWebUI] Milvus 데이터베이스에 연결 시도 중...")
            self.milvus_manager.connect()
            print("[OpenWebUI] Milvus 데이터베이스 연결 성공")
        except Exception as e:
            print(f"[OpenWebUI] Milvus 데이터베이스 연결 오류: {e}")
            return {
                "success": False,
                "message": f"Milvus 데이터베이스 연결 오류: {str(e)}"
            }
    
    # 콜렉션 존재 확인
    if not self.milvus_manager.has_collection():
        print("[OpenWebUI] 오류: Milvus 콜렉션이 존재하지 않습니다.")
        return {
            "success": False,
            "message": "Milvus 콜렉션이 존재하지 않습니다. 임베딩을 먼저 실행해주세요."
        }
    
    try:
        # 임베딩 데이터 확인
        doc_count = self.milvus_manager.count_entities()
        if doc_count == 0:
            print("[OpenWebUI] 오류: 임베딩된 문서가 없습니다.")
            return {
                "success": False,
                "message": "임베딩된 문서가 없습니다. 먼저 문서를 임베딩해주세요."
            }
        
        print(f"[OpenWebUI] 임베딩된 문서 수: {doc_count}")
        
        # 1. 벡터 검색 실행
        print(f"[OpenWebUI] 벡터 검색 시작: '{query}'")
        try:
            vector_results, search_info = self.search_engine.hybrid_search(query=query, limit=10)
            print(f"[OpenWebUI] 벡터 검색 결과: {len(vector_results)} 개")
        except Exception as e:
            print(f"[OpenWebUI] 벡터 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            vector_results = []
            search_info = {"error": str(e)}
        
        # 2. 키워드 검색 실행 (모든 문서 가져와서 메모리에서 필터링)
        print(f"[OpenWebUI] 키워드 검색 시작...")
        try:
            all_docs = self.milvus_manager.query(
                "id >= 0", 
                output_fields=["id", "path", "title", "chunk_text", "file_type", "tags", "created_at", "updated_at"],
                limit=1000
            )
            print(f"[OpenWebUI] 전체 문서 수: {len(all_docs)}")
        except Exception as e:
            print(f"[OpenWebUI] 문서 조회 오류: {e}")
            all_docs = []
        
        # 검색어로 필터링
        import re
        keyword_results = []
        query_terms = re.findall(r'[\w가-힣]+', query.lower())
        
        for doc in all_docs:
            path = doc.get('path', '').lower()
            title = doc.get('title', '').lower()
            content = doc.get('chunk_text', '').lower()
            
            # 점수 계산
            score = 0
            for term in query_terms:
                # 경로에 검색어가 있으면 높은 점수
                if term in path:
                    score += 5
                # 제목에 검색어가 있으면 중간 점수
                if term in title:
                    score += 3
                # 내용에 검색어가 있으면 낮은 점수
                if term in content:
                    score += 1
            
            if score > 0:
                doc['score'] = score
                keyword_results.append(doc)
        
        # 점수 기준으로 정렬
        keyword_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        keyword_results = keyword_results[:10]
        print(f"[OpenWebUI] 키워드 검색 결과: {len(keyword_results)} 개")
        
        # 3. 결과 병합 및 중복 제거
        all_results = []
        seen_paths = set()
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            path = result.get('path')
            if path and path not in seen_paths:
                seen_paths.add(path)
                # 결과 형식 확인 및 정규화
                normalized_result = {
                    "id": result.get("id"),
                    "path": path,
                    "title": result.get("title", "제목 없음"),
                    "chunk_text": result.get("chunk_text", ""),
                    "score": result.get("score", 0),
                    "source": "vector"
                }
                all_results.append(normalized_result)
        
        # 키워드 검색 결과 추가
        for doc in keyword_results:
            path = doc.get('path')
            if path and path not in seen_paths:
                seen_paths.add(path)
                all_results.append({
                    "id": doc.get('id'),
                    "path": path,
                    "title": doc.get('title', '제목 없음'),
                    "chunk_text": doc.get('chunk_text', ''),
                    "score": doc.get('score', 0),
                    "source": "keyword"
                })
        
        print(f"[OpenWebUI] 최종 결과 수: {len(all_results)} 개")
        
        # 검색 결과가 없는 경우 처리
        if not all_results:
            context = "관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해보세요."
        else:
            # 컨텍스트 구성
            context = "다음은 관련 문서에서 찾은 정보입니다:\n\n"
            for i, result in enumerate(all_results[:5], 1):
                # 파일 경로와 제목이 있는지 확인
                path = result.get('path', '파일 경로 없음')
                title = result.get('title', '제목 없음')
                chunk_text = result.get('chunk_text', '')
                source = result.get('source', '알 수 없음')
                score = result.get('score', 0)
                
                print(f"[OpenWebUI] 결과 {i}: {title} (출처: {source}, 점수: {score})")
                
                # 결과 포맷팅
                context += f"[{i}] {title} (파일: {path})\n"
                context += f"{chunk_text}\n\n"
    except Exception as e:
        print(f"[OpenWebUI] 검색 오류: {e}")
        context = f"검색 중 오류가 발생했습니다: {str(e)}"
        all_results = []
    
    # 시스템 프롬프트 구성
    system_prompt = (
        "당신은 사용자의 Obsidian 노트와 문서를 분석하고 질문에 답변하는 전문 도우미입니다. "
        "주어진 컨텍스트 정보는 Milvus 벡터 데이터베이스에서 검색된 사용자의 개인 노트와 지식 정보입니다. "
        "이 정보를 바탕으로 정확하고 유용한 답변을 제공하세요. "
        "컨텍스트에 관련 정보가 없는 경우, 그 사실을 정직하게 알려주세요. "
        "답변은 명확하고 간결하게 작성하되, 필요한 모든 정보를 포함해야 합니다. "
        "출처 정보가 제공된 경우, 답변에 해당 출처를 인용하세요."
    )
    
    # 최종 메시지 구성
    user_message = f"컨텍스트: {context}\n\n질문: {query}"
    
    print(f"[OpenWebUI] 최종 메시지 구성 완료: {len(user_message)} 자")
    
    # Ollama 직접 호출
    try:
        # Ollama 클라이언트 가져오기
        from ollama_client import OllamaClient
        ollama_client = OllamaClient()
        
        # Ollama 응답 생성
        print(f"[OpenWebUI] Ollama 직접 호출: 모델={config.OLLAMA_MODEL}")
        response = ollama_client.generate_response(
            model=config.OLLAMA_MODEL,
            system_prompt=system_prompt,
            user_message=user_message
        )
        
        if not response:
            print("[OpenWebUI] 오류: Ollama 응답이 비어 있습니다.")
            response = "응답을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요."
        else:
            print(f"[OpenWebUI] Ollama 응답 생성 완료: {len(response)} 자")
        
        # 성공 응답 반환
        return {
            "success": True,
            "message": response,
            "results": all_results,
            "search_info": search_info,
            "model": config.OLLAMA_MODEL,
            "source": "direct_ollama"
        }
    except Exception as e:
        print(f"[OpenWebUI] Ollama 호출 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # OpenWebUI API를 통한 응답 생성 시도
        print("[OpenWebUI] OpenWebUI API를 통한 응답 생성 시도...")
    
    # OpenWebUI API 호출 시도
    try:
        # 인증 정보 설정
        auth = None
        if hasattr(config, 'OPENWEBUI_AUTH_ENABLED') and config.OPENWEBUI_AUTH_ENABLED:
            auth = (config.OPENWEBUI_USERNAME, config.OPENWEBUI_PASSWORD)
        
        # API 키 헤더 갱신
        headers = self.headers.copy()
        headers['Content-Type'] = 'application/json'
        
        # 요청 로깅
        print(f"[OpenWebUI] API 요청: {self.base_url}/api/chats/{chat_id}/messages")
        print(f"[OpenWebUI] 헤더: {headers}")
        
        # 메시지 데이터 준비
        message = f"컨텍스트 정보: {context}\n\n질문: {query}"
        
        # API 요청 전송
        response = requests.post(
            f"{self.base_url}/api/chats/{chat_id}/messages",
            json={"content": message},
            auth=auth,  # 기본 인증 정보
            headers=headers,  # API 키 인증 헤더
            timeout=30  # 타임아웃 증가
        )
        
        # 응답 헤더 확인
        print(f"[OpenWebUI] 응답 상태 코드: {response.status_code}")
        print(f"[OpenWebUI] 응답 헤더: {response.headers}")
        
        # 응답 내용 확인
        content_type = response.headers.get('Content-Type', '')
        print(f"[OpenWebUI] 응답 컨텐트 타입: {content_type}")
        
        # HTML 응답 체크 (API 오류 시 HTML이 반환될 수 있음)
        if 'text/html' in content_type:
            print("[OpenWebUI] 오류: API가 HTML 응답을 반환했습니다. API 엔드포인트가 올바른지 확인하세요.")
            return {
                "success": False,
                "message": "OpenWebUI API가 HTML 응답을 반환했습니다. API 엔드포인트가 올바른지 확인하세요.",
                "results": all_results,
                "search_info": search_info
            }
        
        # JSON 응답 파싱
        try:
            result = response.json()
            print(f"[OpenWebUI] 응답 JSON: {result}")
            
            # 응답 메시지 추출
            if 'message' in result:
                ai_message = result.get('message')
            elif 'content' in result:
                ai_message = result.get('content')
            else:
                ai_message = "응답 형식이 예상과 다릅니다. API 응답을 확인하세요."
                print(f"[OpenWebUI] 오류: 예상치 못한 응답 형식: {result}")
            
            return {
                "success": True,
                "message": ai_message,
                "results": all_results,
                "search_info": search_info,
                "model": result.get('model', 'unknown'),
                "source": "openwebui_api"
            }
        except json.JSONDecodeError as e:
            print(f"[OpenWebUI] JSON 파싱 오류: {e}")
            print(f"[OpenWebUI] 응답 내용: {response.text[:500]}...")
            
            # 응답 내용이 JSON이 아닌 경우 직접 결과 구성
            return {
                "success": False,
                "message": "OpenWebUI API 응답을 파싱할 수 없습니다. API 호환성을 확인하세요.",
                "results": all_results,
                "search_info": search_info,
                "raw_response": response.text[:1000]  # 디버깅을 위해 원시 응답 일부 포함
            }
    except Exception as e:
        print(f"[OpenWebUI] API 요청 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시 검색 결과만 반환
        return {
            "success": False,
            "message": f"API 오류: {str(e)}",
            "results": all_results,
            "search_info": search_info
        }
