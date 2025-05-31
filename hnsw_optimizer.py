#!/usr/bin/env python3
"""
HNSW 최적화 모듈 - 기존 milvus_manager.py의 성능 기능들과 통합
Milvus HNSW 인덱스의 성능을 최대화하는 최적화 모듈
"""

import time
import logging
from typing import Dict, Any, List
import config
from logger import get_logger

# 중앙 집중식 로깅 시스템 사용
logger = get_logger('HNSWOptimizer')

class HNSWOptimizer:
    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager
        
    def create_optimized_index(self, collection_name=None):
        """최적화된 HNSW 인덱스 생성 - milvus_manager의 기존 로직 활용"""
        start_time = time.time()
        logger.info(f"최적화된 인덱스 생성 시작 - 커렉션: {collection_name or self.milvus_manager.collection_name}")
        
        if collection_name is None:
            collection_name = self.milvus_manager.collection_name
            
        try:
            # 기존 인덱스 확인
            collection = self.milvus_manager.collection
            logger.debug(f"커렉션 접근 성공: {collection_name}")
            
            if hasattr(collection, 'indexes') and collection.indexes:
                logger.info(f"기존 인덱스가 존재합니다: {len(collection.indexes)}개. 최적화된 인덱스로 재구성을 권장합니다.")
                
                # 기존 인덱스 정보 반환
                existing_index = collection.indexes[0]
                index_type = existing_index.params.get("index_type", "Unknown")
                metric_type = existing_index.params.get("metric_type", "Unknown")
                
                logger.debug(f"기존 인덱스 정보 - 타입: {index_type}, 메트릭: {metric_type}")
                
                elapsed_time = time.time() - start_time
                logger.info(f"기존 인덱스 확인 완료 ({elapsed_time:.3f}초)")
                
                return {
                    "index_type": index_type,
                    "metric_type": metric_type,
                    "status": "existing"
                }
            
            # GPU 사용 여부 확인
            gpu_available = False
            if config.USE_GPU and hasattr(self.milvus_manager, '_is_gpu_available'):
                gpu_start = time.time()
                try:
                    gpu_available = self.milvus_manager._is_gpu_available()
                    logger.debug(f"GPU 사용 가능 여부 확인: {gpu_available} ({time.time() - gpu_start:.3f}초)")
                except Exception as gpu_err:
                    logger.warning(f"GPU 사용 가능 여부 확인 오류: {gpu_err}")
                    gpu_available = False
            
            # GPU 사용 시 최적화된 설정
            if config.USE_GPU and gpu_available:
                gpu_index_type = getattr(config, 'GPU_INDEX_TYPE', 'GPU_IVF_FLAT')
                logger.debug(f"GPU 인덱스 타입 선택: {gpu_index_type}")
                
                if gpu_index_type == "GPU_IVF_FLAT":
                    index_params = {
                        "metric_type": "COSINE",
                        "index_type": "GPU_IVF_FLAT",
                        "params": {
                            "nlist": 2048,  # 클러스터 수 최적화
                        }
                    }
                    logger.debug("GPU_IVF_FLAT 인덱스 파라미터 사용, nlist=2048")
                elif gpu_index_type == "GPU_CAGRA":
                    index_params = {
                        "metric_type": "COSINE",
                        "index_type": "GPU_CAGRA",
                        "params": {
                            "search_width": 32,
                            "build_algo": "IVF_PQ"
                        }
                    }
                    logger.debug("GPU_CAGRA 인덱스 파라미터 사용, search_width=32, build_algo=IVF_PQ")
                else:  # 기본 GPU_IVF_FLAT
                    index_params = {
                        "metric_type": "COSINE",
                        "index_type": "GPU_IVF_FLAT",
                        "params": {"nlist": 2048}
                    }
                    logger.debug("기본 GPU_IVF_FLAT 인덱스 파라미터 사용, nlist=2048")
            else:
                # CPU용 HNSW 최적화 설정
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,        # 연결 수 (메모리 vs 성능 균형)
                        "efConstruction": 256,  # 구축 시 후보 수
                    }
                }
                logger.debug("CPU HNSW 인덱스 파라미터 사용, M=16, efConstruction=256")
            
            # 인덱스 생성 시작
            index_start = time.time()
            logger.info(f"인덱스 생성 시작: {index_params['index_type']}, 벡터 필드: 'vector'")
            
            try:
                collection.create_index(
                    field_name="vector",  # 벡터 필드명
                    index_params=index_params
                )
                index_time = time.time() - index_start
                logger.info(f"인덱스 생성 성공: {index_params['index_type']} ({index_time:.3f}초)")
            except Exception as index_err:
                logger.error(f"인덱스 생성 오류: {index_err}", exc_info=True)
                # 간단한 파라미터로 재시도
                logger.warning("단순 파라미터로 인덱스 생성 재시도 중...")
                try:
                    simple_params = {
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT" if gpu_available else "HNSW",
                        "params": {"nlist": 1024} if gpu_available else {"M": 12, "efConstruction": 128}
                    }
                    collection.create_index(
                        field_name="vector",
                        index_params=simple_params
                    )
                    index_params = simple_params
                    logger.info(f"단순 파라미터로 인덱스 생성 성공: {index_params['index_type']}")
                except Exception as retry_err:
                    logger.error(f"인덱스 재시도 오류: {retry_err}", exc_info=True)
                    raise retry_err
            
            total_time = time.time() - start_time
            entity_count = self.milvus_manager.count_entities() if hasattr(self.milvus_manager, 'count_entities') else 'unknown'
            logger.info(f"최적화된 인덱스 생성 완료: {index_params['index_type']}, 엔티티 수: {entity_count} ({total_time:.3f}초)")
            
            # 생성된 인덱스 파라미터 반환
            index_params["creation_time"] = total_time
            index_params["entity_count"] = entity_count
            return index_params
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"최적화된 인덱스 생성 실패 ({total_time:.3f}초): {e}", exc_info=True)
            return None
    
    def optimize_search_params(self, query_complexity="medium"):
        """검색 복잡도에 따른 최적화된 검색 파라미터"""
        
        # GPU vs CPU 파라미터 선택
        if config.USE_GPU and self.milvus_manager._is_gpu_available():
            # GPU 파라미터
            if query_complexity == "simple":
                return {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 8}  # 낮은 nprobe로 빠른 검색
                }
            elif query_complexity == "medium":
                return {
                    "metric_type": "COSINE", 
                    "params": {"nprobe": 16}  # 중간 nprobe로 균형
                }
            elif query_complexity == "complex":
                return {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 32}  # 높은 nprobe로 정확한 검색
                }
        else:
            # CPU HNSW 파라미터
            if query_complexity == "simple":
                return {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}  # 낮은 ef로 빠른 검색
                }
            elif query_complexity == "medium":
                return {
                    "metric_type": "COSINE", 
                    "params": {"ef": 128}  # 중간 ef로 균형
                }
            elif query_complexity == "complex":
                return {
                    "metric_type": "COSINE",
                    "params": {"ef": 512}  # 높은 ef로 정확한 검색
                }
        
        # 기본값
        return {
            "metric_type": "COSINE",
            "params": {"ef": 128} if not config.USE_GPU else {"nprobe": 16}
        }
        
    def adaptive_search(self, query_vector, initial_limit=20):
        """적응적 검색 - 결과 품질에 따라 검색 파라미터 조정"""
        start_time = time.time()
        logger.info(f"적응적 검색 시작 - 초기 한계: {initial_limit}")
        
        try:
            # 1단계: 빠른 검색으로 시작
            fast_params = self.optimize_search_params("simple")
            logger.debug(f"빠른 검색 파라미터 사용: {fast_params}")
            
            initial_start = time.time()
            try:
                if hasattr(self.milvus_manager, 'search_with_params'):
                    logger.debug(f"search_with_params 메서드 사용, 한계: {initial_limit}")
                    initial_results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=initial_limit,
                        search_params=fast_params
                    )
                else:
                    logger.debug(f"기본 search 메서드 사용, 한계: {initial_limit}")
                    initial_results = self.milvus_manager.search(
                        query_vector, initial_limit
                    )
                    
                initial_search_time = time.time() - initial_start
                logger.debug(f"초기 검색 완료: {len(initial_results)}개 결과 ({initial_search_time:.3f}초)")
                
                # 특수 문자 경로 처리 확인
                special_char_paths = []
                excalidraw_files = []
                
                for result in initial_results:
                    if hasattr(result, 'entity') and 'path' in result.entity:
                        path = result.entity['path']
                        if any(c in path for c in "'\"()[]{},;"):
                            special_char_paths.append(path)
                            logger.debug(f"특수 문자가 포함된 경로 발견: {path}")
                        if "excalidraw" in path.lower():
                            excalidraw_files.append(path)
                            logger.debug(f"Excalidraw 파일 발견: {path}")
                
                if special_char_paths:
                    logger.info(f"초기 검색에서 {len(special_char_paths)}개의 특수 문자 경로 발견")
                if excalidraw_files:
                    logger.info(f"초기 검색에서 {len(excalidraw_files)}개의 Excalidraw 파일 발견")
                
            except Exception as e:
                initial_search_time = time.time() - initial_start
                logger.error(f"초기 검색 오류 ({initial_search_time:.3f}초): {e}", exc_info=True)
                return []
            
            # 결과 품질 평가
            quality_start = time.time()
            quality_score = 0
            results_ratio = len(initial_results) / initial_limit if initial_limit > 0 else 0
            top_score = initial_results[0].score if initial_results else 0
            
            logger.debug(f"결과 품질 평가 - 결과 비율: {results_ratio:.2f}, 최고 점수: {top_score:.4f}")
            
            need_precise_search = False
            
            if len(initial_results) < initial_limit * 0.7:
                logger.debug(f"결과 수 부족: {len(initial_results)}/{initial_limit} ({results_ratio:.2f}), 정밀 검색 필요")
                need_precise_search = True
            elif initial_results and initial_results[0].score < 0.7:
                logger.debug(f"최고 점수 부족: {top_score:.4f} < 0.7, 정밀 검색 필요")
                need_precise_search = True
            else:
                logger.debug(f"초기 검색 결과 품질 양호: 결과 수 {len(initial_results)}, 최고 점수 {top_score:.4f}")
            
            if need_precise_search:
                try:
                    # 2단계: 더 정확한 검색
                    logger.info("초기 검색 결과 불충분: 정밀 모드로 전환 중")
                    precise_params = self.optimize_search_params("complex")
                    precise_start = time.time()
                    precise_limit = initial_limit * 2
                    
                    logger.debug(f"정밀 검색 파라미터: {precise_params}, 한계: {precise_limit}")
                    
                    if hasattr(self.milvus_manager, 'search_with_params'):
                        enhanced_results = self.milvus_manager.search_with_params(
                            vector=query_vector,
                            limit=precise_limit,
                            search_params=precise_params
                        )
                    else:
                        enhanced_results = self.milvus_manager.search(
                            query_vector, precise_limit
                        )
                    
                    precise_search_time = time.time() - precise_start
                    logger.debug(f"정밀 검색 완료: {len(enhanced_results)}개 결과 ({precise_search_time:.3f}초)")
                    
                    # 특수 문자 경로 처리 확인 (정밀 검색)
                    precise_special_paths = []
                    precise_excalidraw_files = []
                    
                    for result in enhanced_results:
                        if hasattr(result, 'entity') and 'path' in result.entity:
                            path = result.entity['path']
                            if any(c in path for c in "'\"()[]{},;"):
                                if path not in special_char_paths:
                                    special_char_paths.append(path)
                                    precise_special_paths.append(path)
                                    logger.debug(f"정밀 검색에서 특수 문자 경로 발견: {path}")
                            if "excalidraw" in path.lower():
                                if path not in excalidraw_files:
                                    excalidraw_files.append(path)
                                    precise_excalidraw_files.append(path)
                                    logger.debug(f"정밀 검색에서 Excalidraw 파일 발견: {path}")
                    
                    if precise_special_paths:
                        logger.info(f"정밀 검색에서 추가 {len(precise_special_paths)}개의 특수 문자 경로 발견")
                    if precise_excalidraw_files:
                        logger.info(f"정밀 검색에서 추가 {len(precise_excalidraw_files)}개의 Excalidraw 파일 발견")
                        
                    # 성능 향상 정도 계산
                    quality_improvement = 0
                    if enhanced_results and initial_results:
                        quality_improvement = enhanced_results[0].score - initial_results[0].score
                    
                    total_time = time.time() - start_time
                    logger.info(f"정밀 검색 성공 ({total_time:.3f}초): 초기 {len(initial_results)}개 결과 -> 정밀 {len(enhanced_results)}개 결과, 점수 향상: {quality_improvement:.4f}")
                    
                    return {
                        "results": enhanced_results,
                        "stats": {
                            "total_time": total_time,
                            "initial_search_time": initial_search_time,
                            "precise_search_time": precise_search_time,
                            "quality_improvement": quality_improvement,
                            "special_paths_count": len(special_char_paths),
                            "excalidraw_files_count": len(excalidraw_files)
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"정밀 검색 오류: {e}", exc_info=True)
                    total_time = time.time() - start_time
                    logger.warning(f"정밀 검색 실패, 초기 결과 사용 ({total_time:.3f}초)")
                    
                    return {
                        "results": initial_results,
                        "stats": {
                            "total_time": total_time,
                            "initial_search_time": initial_search_time,
                            "special_paths_count": len(special_char_paths),
                            "excalidraw_files_count": len(excalidraw_files),
                            "error": str(e)
                        }
                    }
            
            # 초기 검색 결과만 사용
            total_time = time.time() - start_time
            logger.info(f"적응적 검색 완료: 초기 검색만으로 충분함 ({total_time:.3f}초), {len(initial_results)}개 결과")
            
            return {
                "results": initial_results,
                "stats": {
                    "total_time": total_time,
                    "initial_search_time": initial_search_time,
                    "special_paths_count": len(special_char_paths),
                    "excalidraw_files_count": len(excalidraw_files)
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"적응적 검색 실패 ({total_time:.3f}초): {e}", exc_info=True)
            return {"results": [], "error": str(e)}
    
    def bulk_search_optimization(self, query_vectors, batch_size=100):
        """대량 검색 최적화"""
        start_time = time.time()
        total_vectors = len(query_vectors)
        logger.info(f"대량 검색 최적화 시작 - 총 {total_vectors}개 벡터, 배치 크기: {batch_size}")
        
        # 결과 및 통계 초기화
        all_results = []
        success_count = 0
        error_count = 0
        special_path_count = 0
        excalidraw_file_count = 0
        batch_times = []
        
        # 검색 파라미터 최적화
        param_start = time.time()
        optimized_params = self.optimize_search_params("medium")
        logger.debug(f"최적화된 검색 파라미터: {optimized_params} ({time.time() - param_start:.3f}초)")
        
        # 배치 완료 비율을 위한 변수
        num_batches = (total_vectors + batch_size - 1) // batch_size  # 올림 나누기
        
        for i in range(0, total_vectors, batch_size):
            batch_start = time.time()
            batch_vectors = query_vectors[i:i+batch_size]
            current_batch_size = len(batch_vectors)
            batch_num = i // batch_size + 1
            
            logger.debug(f"배치 {batch_num}/{num_batches} 검색 시작 ({current_batch_size}개 벡터)")
            
            # 병렬 검색 실행
            batch_results = []
            batch_success = 0
            batch_errors = 0
            batch_special_paths = []
            batch_excalidraw_files = []
            
            for v_idx, vector in enumerate(batch_vectors):
                vector_start = time.time()
                try:
                    if hasattr(self.milvus_manager, 'search_with_params'):
                        results = self.milvus_manager.search_with_params(
                            vector=vector,
                            limit=20,
                            search_params=optimized_params
                        )
                    else:
                        results = self.milvus_manager.search(vector, 20)
                        
                    # 검색 결과 분석
                    for result in results:
                        if hasattr(result, 'entity') and 'path' in result.entity:
                            path = result.entity['path']
                            if any(c in path for c in "'\"()[]{},;"):
                                batch_special_paths.append(path)
                            if "excalidraw" in path.lower():
                                batch_excalidraw_files.append(path)
                    
                    batch_results.append(results)
                    batch_success += 1
                    success_count += 1
                    
                    if v_idx % 10 == 0 and v_idx > 0:  # 로그 양을 줄이기 위해 10개마다 로깅
                        logger.debug(f"  - 배치 내 진행상황: {v_idx+1}/{current_batch_size} ({(v_idx+1)/current_batch_size*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"배치 검색 오류 (vector {i+v_idx}): {e}")
                    batch_results.append([])
                    batch_errors += 1
                    error_count += 1
            
            # 배치 통계 업데이트
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            special_path_count += len(set(batch_special_paths))
            excalidraw_file_count += len(set(batch_excalidraw_files))
            
            # 배치 결과 추가
            all_results.extend(batch_results)
            
            # 진행상황 로깅
            progress = (i + current_batch_size) / total_vectors * 100
            avg_batch_time = sum(batch_times) / len(batch_times)
            est_remaining = avg_batch_time * (num_batches - batch_num)
            
            logger.info(f"배치 {batch_num}/{num_batches} 완료: {batch_success}성공/{batch_errors}실패 ({batch_time:.3f}초), 진행률: {progress:.1f}%, 예상 남은 시간: {est_remaining:.1f}초")
            
            if batch_special_paths:
                logger.debug(f"  - 특수 문자 경로 {len(set(batch_special_paths))}개 발견")
            if batch_excalidraw_files:
                logger.debug(f"  - Excalidraw 파일 {len(set(batch_excalidraw_files))}개 발견")
        
        # 최종 통계 및 결과 반환
        total_time = time.time() - start_time
        success_rate = success_count / total_vectors * 100 if total_vectors > 0 else 0
        
        logger.info(f"대량 검색 최적화 완료 ({total_time:.3f}초) - 총 {total_vectors}개 중 {success_count}개 성공 ({success_rate:.1f}%), {error_count}개 오류")
        
        if special_path_count > 0 or excalidraw_file_count > 0:
            logger.info(f"  - 특수 경로 통계: {special_path_count}개 특수 문자 경로, {excalidraw_file_count}개 Excalidraw 파일")
        
        return {
            "results": all_results,
            "stats": {
                "total_time": total_time,
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_rate,
                "special_path_count": special_path_count,
                "excalidraw_file_count": excalidraw_file_count,
                "avg_batch_time": sum(batch_times) / len(batch_times) if batch_times else 0
            }
        }
    
    def index_performance_monitoring(self):
        """인덱스 성능 모니터링 - milvus_manager의 기존 기능 활용"""
        start_time = time.time()
        logger.info("인덱스 성능 모니터링 시작")
        
        try:
            # 기존 성능 통계 기능 활용
            if hasattr(self.milvus_manager, 'get_performance_stats'):
                logger.debug("get_performance_stats 메서드 사용 중")
                stats_start = time.time()
                try:
                    base_stats = self.milvus_manager.get_performance_stats()
                    stats_time = time.time() - stats_start
                    logger.debug(f"성능 통계 가져오기 성공 ({stats_time:.3f}초)")
                except Exception as stats_err:
                    logger.error(f"성능 통계 가져오기 오류: {stats_err}", exc_info=True)
                    raise
            else:
                # 폴백: 기본 통계
                logger.debug("기본 통계 수집 시작 (get_performance_stats 메서드 없음)")
                
                load_start = time.time()
                try:
                    collection = self.milvus_manager.collection
                    collection.load()
                    logger.debug(f"커렉션 로드 성공 ({time.time() - load_start:.3f}초)")
                except Exception as load_err:
                    logger.error(f"커렉션 로드 오류: {load_err}", exc_info=True)
                    raise
                
                count_start = time.time()
                try:
                    row_count = self.milvus_manager.count_entities()
                    logger.debug(f"엔티티 수 계산 성공: {row_count} ({time.time() - count_start:.3f}초)")
                except Exception as count_err:
                    logger.error(f"엔티티 수 계산 오류: {count_err}", exc_info=True)
                    row_count = "unknown"
                
                types_start = time.time()
                try:
                    file_types = self.milvus_manager.get_file_type_counts()
                    logger.debug(f"파일 타입 계산 성공: {len(file_types)} 개 타입 ({time.time() - types_start:.3f}초)")
                except Exception as types_err:
                    logger.error(f"파일 타입 계산 오류: {types_err}", exc_info=True)
                    file_types = {}
                
                base_stats = {
                    "total_entities": row_count,
                    "file_types": file_types
                }
            
            # 추가 성능 메트릭
            logger.debug("추가 성능 메트릭 계산 중...")
            metrics_start = time.time()
            
            # 메모리 사용량 추정
            memory_start = time.time()
            memory_usage = self._estimate_memory_usage(base_stats.get("total_entities", 0))
            logger.debug(f"메모리 사용량 추정 완료: {memory_usage} ({time.time() - memory_start:.3f}초)")
            
            # 검색 지연시간 측정
            latency_start = time.time()
            search_latency = self._measure_search_latency()
            logger.debug(f"검색 지연시간 측정 완료: {search_latency} ({time.time() - latency_start:.3f}초)")
            
            # 최적화 권장사항 생성
            recommendations_start = time.time()
            recommendations = self._generate_optimization_recommendations(base_stats.get("total_entities", 0))
            logger.debug(f"최적화 권장사항 생성 완료: {len(recommendations)}개 항목 ({time.time() - recommendations_start:.3f}초)")
            
            performance_metrics = {
                "collection_size": base_stats.get("total_entities", 0),
                "index_type": base_stats.get("index_type", "Unknown"),
                "memory_usage": memory_usage,
                "search_latency": search_latency,
                "recommendations": recommendations
            }
            
            # 기존 통계와 병합
            performance_metrics.update(base_stats)
            
            metrics_time = time.time() - metrics_start
            total_time = time.time() - start_time
            logger.info(f"성능 모니터링 완료 ({total_time:.3f}초): {len(base_stats)}개 기본 통계, {len(recommendations)}개 권장사항")
            
            return performance_metrics
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"성능 모니터링 오류 ({total_time:.3f}초): {e}", exc_info=True)
            return {
                "error": str(e),
                "collection_size": 0,
                "recommendations": ["성능 모니터링 중 오류 발생"],
                "monitoring_time": total_time
            }
    
    def _estimate_memory_usage(self, row_count):
        """메모리 사용량 추정"""
        logger.debug(f"메모리 사용량 추정 시작 - 행 수: {row_count}")
        
        try:
            vector_size = config.VECTOR_DIM * 4  # float32 = 4 bytes
            logger.debug(f"벡터 디멘전: {config.VECTOR_DIM}, 요소당 크기: 4 bytes")
            
            estimated_mb = (row_count * vector_size) / (1024 * 1024)
            result = f"{estimated_mb:.2f} MB"
            
            logger.debug(f"메모리 사용량 추정 결과: {result} (raw: {estimated_mb:.6f} MB)")
            return result
            
        except Exception as e:
            logger.error(f"메모리 사용량 추정 오류: {e}", exc_info=True)
            return "N/A"
    
    def _measure_search_latency(self):
        """검색 지연시간 측정"""
        logger.debug("검색 지연시간 측정 시작")
        
        try:
            # 샘플 벡터로 검색 시간 측정
            sample_vector = [0.1] * config.VECTOR_DIM
            logger.debug(f"측정용 샘플 벡터 생성 - 차원: {config.VECTOR_DIM}")
            
            # 검색 파라미터 설정
            search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
            limit = 10
            logger.debug(f"검색 파라미터: {search_params}, 한계: {limit}")
            
            # 검색 실행 및 시간 측정
            latency_results = []
            num_tests = 3  # 정확한 평균을 위해 여러 번 테스트
            
            for i in range(num_tests):
                start_time = time.time()
            # 기존 벤치마크 기능이 있으면 사용
            if hasattr(self.milvus_manager, 'benchmark_search_strategies'):
                return self.milvus_manager.benchmark_search_strategies(test_queries)
            
            # 없으면 직접 구현
            sample_vector = [0.1] * config.VECTOR_DIM
            
            results = {
                "simple_search": [],
                "medium_search": [],
                "complex_search": []
            }
            
            complexities = ["simple", "medium", "complex"]
            
            for complexity in complexities:
                search_params = self.optimize_search_params(complexity)
                
                for i in range(test_queries):
                    start_time = time.time()
                    
                    if hasattr(self.milvus_manager, 'search_with_params'):
                        self.milvus_manager.search_with_params(
                            vector=sample_vector,
                            limit=20,
                            search_params=search_params
                        )
                    else:
                        self.milvus_manager.search(sample_vector, 20)
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # ms
                    results[f"{complexity}_search"].append(latency)
            
            # 통계 계산
            benchmark_stats = {}
            for complexity in complexities:
                latencies = results[f"{complexity}_search"]
                benchmark_stats[complexity] = {
                    "avg_latency": sum(latencies) / len(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "test_count": len(latencies)
                }
            
            return benchmark_stats
            
        except Exception as e:
            logger.error(f"벤치마크 오류: {e}")
            return {"error": str(e)}
    
    def auto_tune_parameters(self):
        """자동 파라미터 튜닝"""
        try:
            collection = self.milvus_manager.collection
            
            # is_loaded 속성 접근을 안전하게 처리
            try:
                # 새로운 방식 시도
                if hasattr(collection, 'load_state') and collection.load_state.name != 'Loaded':
                    collection.load()
                elif hasattr(collection, 'is_loaded') and not collection.is_loaded:
                    collection.load()
                elif not hasattr(collection, 'is_loaded') and not hasattr(collection, 'load_state'):
                    # 안전을 위해 load 시도
                    try:
                        collection.load()
                    except Exception:
                        pass  # 이미 로드되었을 수 있음
            except Exception as e:
                logger.debug(f"컬렉션 로드 상태 확인 중 오류 (무시): {e}")
                # 컬렉션이 이미 로드되었을 가능성이 높으므로 계속 진행
                
            row_count = self.milvus_manager.count_entities()
            
            # 컬렉션 크기에 따른 자동 튜닝
            if row_count < 10000:
                recommended_params = {
                    "search_complexity": "simple",
                    "ef": 64 if not config.USE_GPU else None,
                    "nprobe": 8 if config.USE_GPU else None,
                    "batch_size": 50
                }
            elif row_count < 100000:
                recommended_params = {
                    "search_complexity": "medium", 
                    "ef": 128 if not config.USE_GPU else None,
                    "nprobe": 16 if config.USE_GPU else None,
                    "batch_size": 100
                }
            else:
                recommended_params = {
                    "search_complexity": "complex",
                    "ef": 256 if not config.USE_GPU else None,
                    "nprobe": 32 if config.USE_GPU else None,
                    "batch_size": 200
                }
            
            # GPU 설정 반영
            if config.USE_GPU:
                recommended_params["gpu_optimization"] = True
                recommended_params["gpu_available"] = self.milvus_manager._is_gpu_available()
            
            logger.info(f"자동 튜닝 완료: {recommended_params}")
            return recommended_params
            
        except Exception as e:
            logger.error(f"자동 튜닝 오류: {e}")
            return {"error": str(e)}
