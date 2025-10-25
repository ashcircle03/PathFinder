from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import ollama
import os
import time
import uuid
from typing import List, Dict, Any, Optional
from prometheus_client import make_asgi_app

# LLMOps 모듈
from src.logging_config import logger
from src.metrics import (
    request_count, request_duration, track_rag_metrics,
    track_llm_metrics, track_feedback_metrics
)
from src.mlflow_tracker import get_mlflow_tracker
from src.prompt_manager import get_prompt_manager
from src.feedback_db import get_feedback_db

app = FastAPI(
    title="PathFinder API",
    description="고등학생 대학 학과 매칭 서비스 - LLMOps Edition",
    version="0.3.0"
)

# Prometheus 메트릭 엔드포인트
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 환경 변수
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

# Ollama 클라이언트 설정
client = ollama.Client(host=OLLAMA_HOST)

# LLMOps 모듈 초기화
mlflow_tracker = get_mlflow_tracker()
prompt_manager = get_prompt_manager()
feedback_db = get_feedback_db()

# RAG 시스템 (지연 로딩)
rag_system = None


# 요청 추적 미들웨어
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """모든 요청을 추적하고 메트릭 수집"""
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    # 요청 로깅
    logger.info(
        "request_started",
        trace_id=trace_id,
        path=request.url.path,
        method=request.method
    )

    response = await call_next(request)

    # 응답 시간 계산
    duration = time.time() - start_time

    # 메트릭 기록
    request_count.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()

    request_duration.labels(
        endpoint=request.url.path,
        method=request.method
    ).observe(duration)

    # 응답 로깅
    logger.info(
        "request_completed",
        trace_id=trace_id,
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=duration
    )

    return response


class InterestRequest(BaseModel):
    interests: str

    class Config:
        json_schema_extra = {
            "example": {
                "interests": "프로그래밍, 게임 개발, 수학"
            }
        }


class RecommendationResponse(BaseModel):
    recommended_majors: List[str]
    reasoning: str


class RAGRecommendationResponse(BaseModel):
    recommendation_id: str
    recommended_majors: List[str]
    reasoning: str
    retrieved_context: Optional[List[Dict[str, Any]]] = None


class FeedbackRequest(BaseModel):
    recommendation_id: str
    rating: int  # 1-5
    is_helpful: bool
    selected_majors: Optional[List[str]] = None
    comment: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "recommendation_id": "550e8400-e29b-41d4-a716-446655440000",
                "rating": 5,
                "is_helpful": True,
                "selected_majors": ["컴퓨터공학과", "소프트웨어학과"],
                "comment": "매우 도움이 되었습니다!"
            }
        }


@app.get("/")
def root():
    return {
        "service": "PathFinder",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """서비스 헬스 체크"""
    try:
        # Ollama 연결 확인
        models = client.list()
        return {
            "status": "healthy",
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL,
            "available_models": [m['name'] for m in models.get('models', [])]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama 연결 실패: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_major(request: InterestRequest):
    """학생의 관심사를 기반으로 대학 학과를 추천"""

    prompt = f"""학생의 관심사: {request.interests}

위 관심사를 가진 고등학생에게 적합한 대학 학과를 3-5개 추천하고, 각 학과가 학생의 관심사와 어떻게 연결되는지 구체적으로 설명해주세요.

다음 형식으로 답변해주세요:
1. 추천 학과: [학과명1, 학과명2, 학과명3, ...]
2. 추천 이유: [각 학과를 추천하는 구체적인 이유]

중요: 반드시 한국어로만 답변해주세요.

추천 학과:"""

    try:
        # Ollama를 통해 LLM 호출
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': '당신은 한국의 고등학생을 위한 진로 상담 전문가입니다. 학생의 관심사를 분석하여 적합한 대학 학과를 추천해주세요. 반드시 한국어로만 답변해주세요.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.7,
                'num_predict': 512
            }
        )

        # 응답 파싱
        llm_response = response['message']['content']

        # 간단한 파싱 (실제로는 더 정교하게 구현 필요)
        lines = llm_response.strip().split('\n')

        # 추천 학과 추출 (첫 번째 줄 또는 관련 줄)
        majors_line = ""
        reasoning = llm_response

        for line in lines:
            if "추천 학과" in line or any(keyword in line for keyword in ["학과", "전공", ","]):
                majors_line = line
                break

        # 쉼표로 구분된 학과 추출
        if majors_line:
            majors = [m.strip() for m in majors_line.replace("추천 학과:", "").split(",")]
            majors = [m for m in majors if m and len(m) > 0]
        else:
            majors = ["컴퓨터공학과", "소프트웨어학과"]  # 기본값

        return RecommendationResponse(
            recommended_majors=majors[:5],  # 최대 5개
            reasoning=reasoning
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"학과 추천 중 오류 발생: {str(e)}"
        )


@app.post("/pull-model")
async def pull_model():
    """Ollama 모델 다운로드 (초기 설정용)"""
    try:
        client.pull(OLLAMA_MODEL)
        return {
            "status": "success",
            "message": f"모델 {OLLAMA_MODEL} 다운로드 완료"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 다운로드 실패: {str(e)}"
        )


def get_rag_system():
    """RAG 시스템을 초기화하고 반환합니다 (싱글톤 패턴)"""
    global rag_system
    if rag_system is None:
        try:
            from src.rag import MajorRecommendationRAG
            rag_system = MajorRecommendationRAG()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG 시스템 초기화 실패: {str(e)}"
            )
    return rag_system


@app.post("/recommend-rag", response_model=RAGRecommendationResponse)
async def recommend_major_rag(request: InterestRequest):
    """RAG 기반 학과 추천 (검색 증강 생성) with LLMOps"""
    recommendation_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "rag_recommendation_started",
        recommendation_id=recommendation_id,
        interests=request.interests
    )

    try:
        # RAG 시스템 가져오기
        rag = get_rag_system()

        # 검색 시작
        retrieval_start = time.time()
        search_results = rag.search_similar_majors(request.interests, top_k=5)
        retrieval_time = time.time() - retrieval_start

        logger.info(
            "rag_retrieval_completed",
            recommendation_id=recommendation_id,
            retrieval_time=retrieval_time,
            num_retrieved=len(search_results)
        )

        # 생성 시작
        generation_start = time.time()
        result = rag.generate_recommendation(request.interests, search_results)
        generation_time = time.time() - generation_start

        logger.info(
            "rag_generation_completed",
            recommendation_id=recommendation_id,
            generation_time=generation_time
        )

        total_time = time.time() - start_time

        # 메트릭 기록
        similarity_scores = [r['score'] for r in search_results]
        track_rag_metrics(
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            similarity_scores=similarity_scores,
            num_retrieved=len(search_results)
        )

        track_llm_metrics(
            model=OLLAMA_MODEL,
            status="success"
        )

        # MLflow 로깅
        try:
            avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            mlflow_tracker.log_recommendation_run(
                interests=request.interests,
                model=OLLAMA_MODEL,
                prompt_version=prompt_manager.get_current_version("rag_recommendation"),
                temperature=0.7,
                num_predict=512,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                recommended_majors=result["recommended_majors"],
                avg_similarity_score=avg_score,
                num_retrieved=len(search_results)
            )
        except Exception as e:
            logger.warning("mlflow_logging_failed", error=str(e))

        logger.info(
            "rag_recommendation_completed",
            recommendation_id=recommendation_id,
            total_time=total_time,
            num_recommended=len(result["recommended_majors"])
        )

        return RAGRecommendationResponse(
            recommendation_id=recommendation_id,
            recommended_majors=result["recommended_majors"],
            reasoning=result["reasoning"],
            retrieved_context=result.get("retrieved_context")
        )

    except Exception as e:
        track_llm_metrics(
            model=OLLAMA_MODEL,
            status="error"
        )

        logger.error(
            "rag_recommendation_failed",
            recommendation_id=recommendation_id,
            error=str(e),
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"RAG 추천 중 오류 발생: {str(e)}"
        )


@app.get("/rag-health")
async def rag_health_check():
    """RAG 시스템 헬스 체크"""
    try:
        rag = get_rag_system()
        return rag.health_check()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG 시스템 상태 확인 실패: {str(e)}"
        )


@app.post("/initialize-db")
async def initialize_database():
    """벡터 DB 초기화 (학과 데이터 임베딩 및 저장)"""
    try:
        from src.initialize_db import initialize_qdrant
        initialize_qdrant()
        return {
            "status": "success",
            "message": "벡터 DB 초기화 및 학과 데이터 저장 완료"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DB 초기화 실패: {str(e)}"
        )


# ===== LLMOps 엔드포인트 =====

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """사용자 피드백 수집"""
    logger.info(
        "feedback_received",
        recommendation_id=feedback.recommendation_id,
        rating=feedback.rating,
        is_helpful=feedback.is_helpful
    )

    try:
        # 메트릭 기록
        track_feedback_metrics(
            rating=feedback.rating,
            is_helpful=feedback.is_helpful
        )

        # 데이터베이스에 저장 (임시로 interests와 recommended_majors는 빈 값)
        # 실제로는 recommendation_id로 조회하여 가져와야 함
        success = feedback_db.save_feedback(
            recommendation_id=feedback.recommendation_id,
            interests="",  # TODO: recommendation_id로 조회
            recommended_majors=[],  # TODO: recommendation_id로 조회
            rating=feedback.rating,
            is_helpful=feedback.is_helpful,
            selected_majors=feedback.selected_majors,
            comment=feedback.comment
        )

        if success:
            logger.info(
                "feedback_saved",
                recommendation_id=feedback.recommendation_id
            )
            return {
                "status": "success",
                "message": "피드백이 성공적으로 저장되었습니다"
            }
        else:
            return {
                "status": "warning",
                "message": "피드백이 메트릭에는 기록되었으나 DB 저장 실패"
            }

    except Exception as e:
        logger.error(
            "feedback_save_failed",
            recommendation_id=feedback.recommendation_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"피드백 저장 실패: {str(e)}"
        )


@app.get("/llmops/stats")
async def get_llmops_stats():
    """LLMOps 통계 조회"""
    try:
        # 피드백 통계
        feedback_stats = feedback_db.get_feedback_stats(days=30)

        # 최근 피드백
        recent_feedback = feedback_db.get_recent_feedback(limit=5)

        return {
            "feedback_stats": feedback_stats,
            "recent_feedback": recent_feedback,
            "system_info": {
                "model": OLLAMA_MODEL,
                "version": "0.3.0",
                "llmops_enabled": True
            }
        }
    except Exception as e:
        logger.error("llmops_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"통계 조회 실패: {str(e)}"
        )


@app.get("/llmops/prompts")
async def list_prompts():
    """사용 가능한 프롬프트 버전 목록"""
    try:
        return {
            "basic_recommendation": {
                "versions": prompt_manager.list_versions("basic_recommendation"),
                "current": prompt_manager.get_current_version("basic_recommendation")
            },
            "rag_recommendation": {
                "versions": prompt_manager.list_versions("rag_recommendation"),
                "current": prompt_manager.get_current_version("rag_recommendation")
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"프롬프트 목록 조회 실패: {str(e)}"
        )
