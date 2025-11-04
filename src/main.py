"""
PathFinder API - LangChain 기반 학과 추천 서비스
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="PathFinder API",
    description="LangChain 기반 고등학생 대학 학과 추천 서비스",
    version="2.0.0"
)

# RAG 시스템 (지연 로딩)
rag_system = None


def get_rag_system():
    """RAG 시스템을 초기화하고 반환합니다 (싱글톤 패턴)"""
    global rag_system
    if rag_system is None:
        try:
            from src.rag import get_rag_system as _get_rag
            rag_system = _get_rag()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG 시스템 초기화 실패: {str(e)}"
            )
    return rag_system


# ===== Request/Response Models =====

class InterestRequest(BaseModel):
    """학과 추천 요청"""
    interests: str

    class Config:
        json_schema_extra = {
            "example": {
                "interests": "프로그래밍, 게임 개발, 수학"
            }
        }


class RecommendationResponse(BaseModel):
    """학과 추천 응답"""
    recommendation_id: str
    recommended_majors: List[str]
    reasoning: str
    retrieved_context: Optional[List[Dict[str, Any]]] = None


# ===== API Endpoints =====

@app.get("/")
def root():
    """서비스 정보"""
    return {
        "service": "PathFinder",
        "version": "2.0.0",
        "description": "LangChain 기반 학과 추천 서비스",
        "framework": "LangChain + Ollama + Qdrant",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """서비스 헬스 체크"""
    try:
        rag = get_rag_system()
        health = rag.health_check()

        if health.get("status") == "healthy":
            return {
                "status": "healthy",
                "rag_system": health
            }
        else:
            raise HTTPException(status_code=503, detail="RAG 시스템 불안정")

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"헬스 체크 실패: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_major(request: InterestRequest):
    """
    LangChain RAG 기반 학과 추천

    - 사용자의 관심사를 벡터 검색
    - 유사한 학과 5개 검색
    - LLM으로 맞춤형 추천 생성
    """
    recommendation_id = str(uuid.uuid4())

    try:
        # RAG 시스템 가져오기
        rag = get_rag_system()

        # 학과 추천
        result = rag.recommend_majors(request.interests, top_k=5)

        return RecommendationResponse(
            recommendation_id=recommendation_id,
            recommended_majors=result["recommended_majors"],
            reasoning=result["reasoning"],
            retrieved_context=result.get("retrieved_context")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 생성 실패: {str(e)}"
        )


@app.post("/search")
async def search_majors(request: InterestRequest):
    """
    벡터 검색만 수행 (LLM 생성 없음)

    - 빠른 응답이 필요한 경우
    - 검색 결과만 확인하고 싶은 경우
    """
    try:
        rag = get_rag_system()
        search_results = rag.search_similar_majors(request.interests, top_k=5)

        return {
            "query": request.interests,
            "results": search_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"검색 실패: {str(e)}"
        )


@app.post("/initialize-db")
async def initialize_database():
    """벡터 DB 초기화 (학과 데이터 임베딩 및 저장)"""
    try:
        from src.initialize_db import initialize_qdrant
        result = initialize_qdrant()

        return {
            "status": "success",
            "message": "벡터 DB 초기화 완료",
            "details": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DB 초기화 실패: {str(e)}"
        )


# ===== Development Endpoints =====

@app.get("/debug/prompt")
async def get_prompt_template():
    """현재 사용 중인 프롬프트 템플릿 확인 (개발용)"""
    try:
        rag = get_rag_system()
        return {
            "template": rag.rag_template,
            "model": rag.llm_model
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"프롬프트 조회 실패: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
