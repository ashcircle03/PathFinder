"""
Prometheus 메트릭 수집
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable


# ===== 요청 메트릭 =====
request_count = Counter(
    'pathfinder_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'pathfinder_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# ===== RAG 메트릭 =====
rag_retrieval_duration = Histogram(
    'pathfinder_rag_retrieval_duration_seconds',
    'RAG retrieval duration in seconds',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

rag_generation_duration = Histogram(
    'pathfinder_rag_generation_duration_seconds',
    'RAG LLM generation duration in seconds',
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
)

rag_similarity_score = Histogram(
    'pathfinder_rag_similarity_score',
    'Average similarity score of retrieved documents',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

rag_retrieved_count = Histogram(
    'pathfinder_rag_retrieved_count',
    'Number of documents retrieved',
    buckets=(1, 2, 3, 5, 10, 20)
)

# ===== LLM 메트릭 =====
llm_tokens_generated = Counter(
    'pathfinder_llm_tokens_generated_total',
    'Total number of tokens generated',
    ['model']
)

llm_requests = Counter(
    'pathfinder_llm_requests_total',
    'Total number of LLM requests',
    ['model', 'status']
)

# ===== 피드백 메트릭 =====
feedback_count = Counter(
    'pathfinder_feedback_total',
    'Total feedback received',
    ['rating']
)

feedback_helpful = Counter(
    'pathfinder_feedback_helpful_total',
    'Helpful feedback count',
    ['is_helpful']
)

# ===== 시스템 메트릭 =====
active_users = Gauge(
    'pathfinder_active_users',
    'Number of active users'
)

system_info = Info(
    'pathfinder_system',
    'System information'
)

# 시스템 정보 설정
system_info.info({
    'version': '0.3.0',
    'llm_model': 'qwen2.5:32b',
    'embedding_model': 'jhgan/ko-sroberta-multitask'
})


# ===== 데코레이터 =====
def track_time(metric: Histogram):
    """시간 측정 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.observe(duration)

        # 함수가 코루틴인지 확인
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_rag_metrics(retrieval_time: float, generation_time: float,
                     similarity_scores: list, num_retrieved: int):
    """RAG 메트릭 기록"""
    rag_retrieval_duration.observe(retrieval_time)
    rag_generation_duration.observe(generation_time)

    if similarity_scores:
        avg_score = sum(similarity_scores) / len(similarity_scores)
        rag_similarity_score.observe(avg_score)

    rag_retrieved_count.observe(num_retrieved)


def track_llm_metrics(model: str, status: str, tokens_generated: int = 0):
    """LLM 메트릭 기록"""
    llm_requests.labels(model=model, status=status).inc()
    if tokens_generated > 0:
        llm_tokens_generated.labels(model=model).inc(tokens_generated)


def track_feedback_metrics(rating: int, is_helpful: bool):
    """피드백 메트릭 기록"""
    feedback_count.labels(rating=str(rating)).inc()
    feedback_helpful.labels(is_helpful=str(is_helpful)).inc()
