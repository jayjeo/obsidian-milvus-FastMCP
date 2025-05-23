#!/usr/bin/env python3
"""
Milvus HNSW 인덱스의 성능을 최대화하는 최적화 모듈
"""

import time
import logging
from typing import Dict, Any, List
import config

logger = logging.getLogger('HNSWOptimizer')

class HNSWOptimizer:
    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager
        
    def create_optimized_index(self, collection_name):
        """최적화된 HNSW 인덱스 생성"""
        
        # GPU 사용 시 최적화된 설정
        if config.USE_GPU:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "GPU_IVF_FLAT",  # GPU 최적화
                "params": {
                    "nlist": 2048,  # 클러스터 수 최적화
                    "cache_dataset_on_device": "true",  # GPU 메모리 캐싱
                }
            }
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
        
        try:
            # 기존 인덱스 확인
            collection = self.milvus_manager.collection
            indexes = collection.indexes
            
            if indexes:
                logger.info("기존 인덱스가 존재합니다. 최적화된 인덱스로 재구성을 권장합니다.")
                return index_params
            
            # 인덱스 생성
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"최적화된 인덱스 생성 완료: {index_params['index_type']}")
            return index_params
            
        except Exception as e:
            logger.error(f"인덱스 생성 오류: {e}")
            return None
    
    def optimize_search_params(self, query_complexity="medium"):
        """검색 복잡도에 따른 최적화된 검색 파라미터"""
        
        if query_complexity == "simple":
            # 빠른 검색 (약간의 정확도 희생)
            return {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # 낮은 ef로 빠른 검색
            }
        elif query_complexity == "medium":
            # 균형잡힌 검색
            return {
                "metric_type": "COSINE", 
                "params": {"ef": 128}  # 중간 ef로 균형
            }
        elif query_complexity == "complex":
            # 높은 정확도 검색
            return {
                "metric_type": "COSINE",
                "params": {"ef": 512}  # 높은 ef로 정확한 검색
            }
        else:
            # 기본값
            return {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
        
    def adaptive_search(self, query_vector, initial_limit=20):
        """적응적 검색 - 결과 품질에 따라 검색 파라미터 조정"""
        
        # 1단계: 빠른 검색으로 시작
        fast_params = self.optimize_search_params("simple")
        try:
            initial_results = self.milvus_manager.search(
                vector=query_vector,
                limit=initial_limit,
                search_params=fast_params
            )
        except Exception as e:
            logger.error(f"초기 검색 오류: {e}")
            return []
        
        # 결과 품질 평가
        if len(initial_results) < initial_limit * 0.7 or \
           (initial_results and initial_results[0].score < 0.7):
            
            try:
                # 2단계: 더 정확한 검색
                precise_params = self.optimize_search_params("complex")
                enhanced_results = self.milvus_manager.search(
                    vector=query_vector,
                    limit=initial_limit * 2,
                    search_params=precise_params
                )
                logger.info("적응적 검색: 정밀 모드로 전환")
                return enhanced_results
            except Exception as e:
                logger.error(f"정밀 검색 오류: {e}")
                return initial_results
        
        return initial_results
    
    def bulk_search_optimization(self, query_vectors, batch_size=100):
        """대량 검색 최적화"""
        
        # 배치 단위로 검색 실행
        all_results = []
        optimized_params = self.optimize_search_params("medium")
        
        for i in range(0, len(query_vectors), batch_size):
            batch_vectors = query_vectors[i:i+batch_size]
            
            # 병렬 검색 실행
            batch_results = []
            for vector in batch_vectors:
                try:
                    results = self.milvus_manager.search(
                        vector=vector,
                        limit=20,
                        search_params=optimized_params
                    )
                    batch_results.append(results)
                except Exception as e:
                    logger.error(f"배치 검색 오류: {e}")
                    batch_results.append([])
            
            all_results.extend(batch_results)
            logger.info(f"배치 검색 진행: {i + len(batch_vectors)}/{len(query_vectors)}")
        
        return all_results
    
    def index_performance_monitoring(self):
        """인덱스 성능 모니터링"""
        try:
            collection = self.milvus_manager.collection
            
            # 컬렉션 통계
            collection.load()
            stats = collection.get_stats()
            
            # 인덱스 정보
            indexes = collection.indexes
            
            # 기본 성능 메트릭
            row_count = int(stats.get('row_count', 0))
            
            performance_metrics = {
                "collection_size": row_count,
                "index_type": indexes[0].params.get("index_type") if indexes else None,
                "memory_usage": self._estimate_memory_usage(row_count),
                "search_latency": self._measure_search_latency(),
                "recommendations": self._generate_optimization_recommendations(row_count)
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"성능 모니터링 오류: {e}")
            return {
                "error": str(e),
                "collection_size": 0,
                "recommendations": ["성능 모니터링 중 오류 발생"]
            }
    
    def _estimate_memory_usage(self, row_count):
        """메모리 사용량 추정"""
        vector_size = config.VECTOR_DIM * 4  # float32 = 4 bytes
        estimated_mb = (row_count * vector_size) / (1024 * 1024)
        return f"{estimated_mb:.2f} MB"
    
    def _measure_search_latency(self):
        """검색 지연시간 측정"""
        try:
            # 샘플 벡터로 검색 시간 측정
            sample_vector = [0.1] * config.VECTOR_DIM
            
            start_time = time.time()
            self.milvus_manager.search(
                vector=sample_vector,
                limit=10,
                search_params={"metric_type": "COSINE", "params": {"ef": 128}}
            )
            end_time = time.time()
            
            return f"{(end_time - start_time) * 1000:.2f} ms"
            
        except Exception as e:
            logger.error(f"지연시간 측정 오류: {e}")
            return "측정 불가"
    
    def _generate_optimization_recommendations(self, row_count):
        """최적화 권장사항 생성"""
        recommendations = []
        
        if row_count > 100000:
            recommendations.append("대용량 컬렉션: GPU 인덱스 사용 고려")
            recommendations.append("배치 검색으로 처리량 향상 가능")
        
        if row_count < 10000:
            recommendations.append("소용량 컬렉션: FLAT 인덱스가 더 효율적일 수 있음")
        
        if config.USE_GPU and row_count > 50000:
            recommendations.append("GPU 메모리 캐싱 활성화로 성능 향상 가능")
        
        if row_count > 1000000:
            recommendations.append("초대용량 컬렉션: 샤딩 및 분산 처리 고려")
        
        # GPU 설정 기반 권장사항
        if config.USE_GPU:
            recommendations.append("GPU 가속 활성화됨 - 대용량 벡터 검색에 최적화")
        else:
            recommendations.append("CPU 모드 - GPU 활성화 시 성능 대폭 향상 가능")
        
        return recommendations
    
    def benchmark_search_performance(self, test_queries=10):
        """검색 성능 벤치마크"""
        try:
            # 다양한 복잡도의 테스트 쿼리 생성
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
                    
                    self.milvus_manager.search(
                        vector=sample_vector,
                        limit=20,
                        search_params=search_params
                    )
                    
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
            collection.load()
            stats = collection.get_stats()
            row_count = int(stats.get('row_count', 0))
            
            # 컬렉션 크기에 따른 자동 튜닝
            if row_count < 10000:
                recommended_params = {
                    "search_complexity": "simple",
                    "ef": 64,
                    "batch_size": 50
                }
            elif row_count < 100000:
                recommended_params = {
                    "search_complexity": "medium", 
                    "ef": 128,
                    "batch_size": 100
                }
            else:
                recommended_params = {
                    "search_complexity": "complex",
                    "ef": 256,
                    "batch_size": 200
                }
            
            # GPU 설정 반영
            if config.USE_GPU:
                recommended_params["gpu_optimization"] = True
                recommended_params["cache_dataset"] = True
            
            logger.info(f"자동 튜닝 완료: {recommended_params}")
            return recommended_params
            
        except Exception as e:
            logger.error(f"자동 튜닝 오류: {e}")
            return {"error": str(e)}
