import re
import json
import time
from datetime import datetime
from collections import defaultdict
from embeddings import EmbeddingModel
import config

class SearchEngine:
    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        # 사용자 피드백 저장소 (문서 ID -> 피드백 점수)
        self.feedback_store = defaultdict(float)
        # 최근 검색 쿼리 캐싱
        self.recent_queries = []
    
    def hybrid_search(self, query, limit=None, filter_params=None):
        # limit이 없으면 config에서 가져옴
        if limit is None:
            limit = config.get_search_results_limit()
        """하이브리드 검색 (벡터 + 키워드 + 피드백 반영)"""
        # 필터 표현식 설정
        filter_expr = None
        if filter_params:
            filter_expr = filter_params.get('filter_expr')
        
        # 안전한 검색을 위해 limit 값 제한 - config에서 최대값의 2배까지 허용
        max_limit = config.get_search_results_limit() * 2  # 내부 처리를 위해 요청된 limit의 2배까지 허용
        safe_limit = min(limit, max_limit)
        
        # 검색 정보 초기화
        search_info = {
            "query": query,
            "requested_limit": limit,
            "actual_limit": safe_limit,
            "filter": filter_expr,
            "vector_count": 0,
            "keyword_count": 0,
            "total_count": 0
        }
        
        try:
            # 벡터 검색 실행 - 안전한 제한값 사용
            vector_results = self._vector_search(query, limit=safe_limit*2, filter_expr=filter_expr)
            search_info["vector_count"] = len(vector_results)
        except Exception as e:
            print(f"[검색 엔진] 벡터 검색 오류: {e}")
            vector_results = []
            search_info["vector_count"] = 0
            search_info["vector_error"] = str(e)
        
        try:
            # 키워드 검색 실행 - 안전한 제한값 사용
            keyword_results = self._keyword_search(query, limit=safe_limit*2, filter_expr=filter_expr)
            search_info["keyword_count"] = len(keyword_results)
        except Exception as e:
            print(f"[검색 엔진] 키워드 검색 오류: {e}")
            keyword_results = []
            search_info["keyword_count"] = 0
            search_info["keyword_error"] = str(e)
        
        # 결과 병합 및 중복 제거
        all_results = []
        seen_ids = set()
        
        # 벡터 검색 결과 추가 (우선순위 높음)
        for result in vector_results:
            result_id = result.get('id')
            if result_id is not None and result_id not in seen_ids:
                seen_ids.add(result_id)
                # 결과 형식 정규화
                normalized_result = {
                    "id": result_id,
                    "path": result.get('path', ''),
                    "title": result.get('title', '제목 없음'),
                    "chunk_text": result.get('chunk_text', ''),
                    "score": result.get('score', 0),
                    "source": 'vector',
                    "file_type": result.get('file_type', ''),
                    "tags": result.get('tags', []),
                    "created_at": result.get('created_at', ''),
                    "updated_at": result.get('updated_at', '')
                }
                all_results.append(normalized_result)
        
        # 키워드 검색 결과 추가
        for result in keyword_results:
            result_id = result.get('id')
            if result_id is not None and result_id not in seen_ids:
                seen_ids.add(result_id)
                # 결과 형식 정규화
                normalized_result = {
                    "id": result_id,
                    "path": result.get('path', ''),
                    "title": result.get('title', '제목 없음'),
                    "chunk_text": result.get('chunk_text', ''),
                    "score": result.get('score', 0),
                    "source": 'keyword',
                    "file_type": result.get('file_type', ''),
                    "tags": result.get('tags', []),
                    "created_at": result.get('created_at', ''),
                    "updated_at": result.get('updated_at', '')
                }
                all_results.append(normalized_result)
        
        try:
            # 점수 기준으로 정렬 및 제한
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # 요청한 제한값과 안전한 제한값 중 작은 값 사용
            actual_limit = min(safe_limit, limit)
            all_results = all_results[:actual_limit]
            
            # 검색 정보 업데이트
            search_info["total_count"] = len(all_results)
            search_info["final_limit_applied"] = actual_limit
        except Exception as e:
            print(f"[검색 엔진] 결과 정렬 및 제한 오류: {e}")
            # 오류 발생 시 기본 처리
            search_info["sort_error"] = str(e)
        
        return all_results, search_info
    
    def _vector_search(self, query, limit, filter_expr):
        """벡터 유사도 검색"""
        query_vector = self.embedding_model.get_embedding(query)
        results = self.milvus_manager.search(
            vector=query_vector,
            limit=limit,
            filter_expr=filter_expr
        )
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "id": hit.id,
                "path": hit.entity.get('path', ''),
                "title": hit.entity.get('title', '제목 없음'),
                "chunk_text": hit.entity.get('chunk_text', ''),
                "score": hit.score,
                "source": "vector",
                "file_type": hit.entity.get('file_type', ''),
                "tags": hit.entity.get('tags', []),
                "created_at": hit.entity.get('created_at', ''),
                "updated_at": hit.entity.get('updated_at', '')
            })
        
        return formatted_results
    
    def _keyword_search(self, query, limit, filter_expr):
        """키워드 기반 텍스트 검색 (한글 파일명 지원)"""
        # 쿼리 용어 정리 (한글 지원을 위해 정규식 패턴 수정)
        terms = re.findall(r'[\w가-힣]+', query.lower())
        if not terms:
            return []
        
        # 모든 청크에 대해 검색 (한글 파일명 처리를 위해 id >= 0 사용)
        try:
            results = self.milvus_manager.query(
                "id >= 0", 
                output_fields=["id", "path", "title", "chunk_text", "file_type", "tags", "created_at", "updated_at"],
                limit=1000
            )
            print(f"[검색엔진] 키워드 검색: {len(results)}개 청크 조회됨")
        except Exception as e:
            print(f"[검색엔진] 키워드 검색 오류: {e}")
            return []
        
        # 결과 점수 매기기
        scored_results = []
        for result in results:
            # 필드 존재 확인 및 기본값 설정
            chunk_text = result.get('chunk_text', '')
            if chunk_text is None:
                chunk_text = ''
            chunk_text = chunk_text.lower()
            
            path = result.get('path', '')
            if path is None:
                path = ''
                
            title = result.get('title', '')
            if title is None:
                title = ''
            
            # 파일 경로나 제목에 쿼리 용어가 포함되어 있는지 확인
            path_match = any(term.lower() in path.lower() for term in terms)
            title_match = any(term.lower() in title.lower() for term in terms)
            
            # 간단한 BM25 스타일 스코어링
            score = 0
            for term in terms:
                # 청크 텍스트에서 용어 빈도 계산
                if len(chunk_text) > 0:  # 0으로 나누기 방지
                    count = chunk_text.count(term)
                    if count > 0:
                        score += count * (1.0 / len(chunk_text))
                
                # 경로 일치에 가중치
                if term in path.lower():
                    score += 5
                
                # 제목 일치에 가중치
                if term in title.lower():
                    score += 3
            
            # 점수가 0보다 크면 결과에 추가
            if score > 0 or path_match or title_match:
                result['score'] = score
                scored_results.append(result)
        
        # 점수순 정렬 및 상위 결과 반환
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:limit]
    
    def _merge_results(self, vector_results, keyword_results, limit):
        """벡터 검색과 키워드 검색 결과 병합 (사용자 피드백 반영)"""
        # 결과 ID별로 인덱싱
        result_map = {}
        
        # 벡터 결과 처리 (높은 가중치)
        for result in vector_results:
            result_id = f"{result['path']}_{result['chunk_index']}"
            result_map[result_id] = {
                **result,
                "final_score": result['score'] * 0.7  # 벡터 검색에 70% 가중치
            }
        
        # 키워드 결과 처리 및 병합
        for result in keyword_results:
            result_id = f"{result['path']}_{result['chunk_index']}"
            if result_id in result_map:
                # 기존 결과에 키워드 점수 추가
                result_map[result_id]['final_score'] += result['score'] * 0.3  # 키워드 검색에 30% 가중치
            else:
                # 새 결과 추가
                result_map[result_id] = {
                    **result,
                    "final_score": result['score'] * 0.3
                }
        
        # 사용자 피드백 반영
        for result_id, result in result_map.items():
            doc_id = result['id']
            if doc_id in self.feedback_store:
                # 피드백 점수 반영 (최대 20% 영향)
                feedback_boost = self.feedback_store[doc_id] * 0.2
                result_map[result_id]['final_score'] += feedback_boost
                result_map[result_id]['boosted_by_feedback'] = True
        
        # 최종 점수로 정렬
        all_results = list(result_map.values())
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return all_results[:limit]
    
    def _build_filter_expr(self, filter_params):
        """필터 표현식 구성"""
        if not filter_params:
            return None
        
        expressions = []
        
        # 파일 타입 필터
        if filter_params.get('file_types'):
            file_types = filter_params['file_types']
            if len(file_types) == 1:
                expressions.append(f"file_type == '{file_types[0]}'")
            else:
                file_type_expr = " || ".join([f"file_type == '{ft}'" for ft in file_types])
                expressions.append(f"({file_type_expr})")
        
        # 태그 필터
        if filter_params.get('tags'):
            tag_expressions = []
            for tag in filter_params['tags']:
                tag_expressions.append(f"tags like '%{tag}%'")
            
            if tag_expressions:
                expressions.append("(" + " && ".join(tag_expressions) + ")")
        
        # 날짜 범위 필터
        if filter_params.get('date_from'):
            try:
                date_from = filter_params['date_from']
                if isinstance(date_from, str):
                    date_from = datetime.fromisoformat(date_from).timestamp()
                expressions.append(f"created_at >= '{date_from}'")
            except:
                pass
        
        if filter_params.get('date_to'):
            try:
                date_to = filter_params['date_to']
                if isinstance(date_to, str):
                    date_to = datetime.fromisoformat(date_to).timestamp()
                expressions.append(f"created_at <= '{date_to}'")
            except:
                pass
        
        return " && ".join(expressions) if expressions else None
    
    def add_feedback(self, doc_id, is_relevant, feedback_strength=1.0):
        """사용자 피드백 추가
        
        Args:
            doc_id: 문서 ID
            is_relevant: 관련성 여부 (True/False)
            feedback_strength: 피드백 강도 (0.0~1.0)
        """
        if is_relevant:
            # 관련성이 높은 문서는 점수 증가
            self.feedback_store[doc_id] += feedback_strength
        else:
            # 관련성이 낮은 문서는 점수 감소
            self.feedback_store[doc_id] -= feedback_strength
        
        # 점수 범위 제한 (-1.0 ~ 1.0)
        self.feedback_store[doc_id] = max(-1.0, min(1.0, self.feedback_store[doc_id]))
        
        return self.feedback_store[doc_id]
    
    def get_feedback(self, doc_id):
        """특정 문서에 대한 피드백 점수 조회"""
        return self.feedback_store.get(doc_id, 0.0)
    
    def clear_feedback(self, doc_id=None):
        """피드백 정보 초기화
        
        Args:
            doc_id: 특정 문서 ID (없으면 모든 피드백 초기화)
        """
        if doc_id is not None:
            if doc_id in self.feedback_store:
                del self.feedback_store[doc_id]
        else:
            self.feedback_store.clear()
    
    def get_similar_queries(self, query, threshold=0.7):
        """유사한 최근 쿼리 찾기"""
        if not self.recent_queries or not query:
            return []
            
        query_vector = self.embedding_model.get_embedding(query)
        similar_queries = []
        
        for past_query in self.recent_queries:
            if past_query == query:
                continue
                
            past_vector = self.embedding_model.get_embedding(past_query)
            
            # 코사인 유사도 계산
            similarity = self._calculate_similarity(query_vector, past_vector)
            if similarity > threshold:
                similar_queries.append({
                    "query": past_query,
                    "similarity": similarity
                })
        
        # 유사도 순 정렬
        similar_queries.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_queries
    
    def _calculate_similarity(self, vec1, vec2):
        """두 벡터 간의 코사인 유사도 계산"""
        # 벡터의 길이가 다를 수 있으므로 최소 길이로 조정
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # 두 벡터가 0인 경우 처리
        if sum(vec1) == 0 or sum(vec2) == 0:
            return 0.0
            
        # 내적 계산
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # 벡터 크기 계산
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        # 코사인 유사도
        return dot_product / (magnitude1 * magnitude2)