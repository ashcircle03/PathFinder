"""
PathFinder API - LangChain 기반 학과 추천 서비스
"""
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# LangSmith 추적 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "pathfinder-production"
# os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"  # 실제 키로 교체 필요

# RAG 시스템 (지연 로딩)
rag_system = None
rag_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 라이프사이클 관리"""
    # Startup
    global rag_system, rag_initialized
    try:
        print("🚀 서버 시작 중... RAG 시스템 초기화")
        print("📊 LangSmith 추적 활성화됨 (프로젝트: pathfinder-production)")
        from src.rag import get_rag_system as _get_rag
        rag_system = _get_rag()
        rag_initialized = True
        print("✅ RAG 시스템 초기화 완료!")
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        rag_initialized = False

    yield

    # Shutdown
    print("🛑 서버 종료 중...")


app = FastAPI(
    title="PathFinder API",
    description="LangChain 기반 고등학생 대학 학과 추천 서비스",
    version="2.0.0",
    lifespan=lifespan
)


def get_rag_system():
    """RAG 시스템을 반환합니다"""
    global rag_system
    if not rag_initialized or rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 아직 초기화되지 않았습니다"
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


class ChatRequest(BaseModel):
    """대화형 상담 요청"""
    message: str
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "저는 프로그래밍과 게임 만드는 것에 관심이 많아요",
                "session_id": "user-12345"
            }
        }


class ChatResponse(BaseModel):
    """대화형 상담 응답"""
    session_id: str
    response: str
    is_ready_to_recommend: bool
    conversation_count: int
    collected_info: Optional[Dict[str, Any]] = None


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
    if rag_initialized and rag_system is not None:
        return {
            "status": "healthy",
            "rag_initialized": True
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 초기화되지 않았습니다"
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


# ===== Conversation Endpoints =====

@app.post("/chat", response_model=ChatResponse)
async def chat_with_counselor(request: ChatRequest):
    """
    대화형 진로 상담

    - 학생과 자연스러운 대화를 통해 관심사를 파악
    - 3-5회 대화 후 학과 추천 준비 완료
    - session_id로 대화 이력 관리
    """
    try:
        from src.conversation import get_conversation_session

        # 세션 ID 생성 또는 기존 세션 사용
        session_id = request.session_id or str(uuid.uuid4())

        # 대화 세션 가져오기
        conversation = get_conversation_session(session_id)

        # 대화 진행
        result = conversation.chat(request.message)

        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            is_ready_to_recommend=result["is_ready_to_recommend"],
            conversation_count=result["conversation_count"],
            collected_info=result.get("collected_info")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"대화 처리 실패: {str(e)}"
        )


@app.post("/chat/{session_id}/recommend", response_model=RecommendationResponse)
async def recommend_from_conversation(session_id: str):
    """
    대화 세션을 기반으로 학과 추천

    - 대화를 통해 수집된 정보를 바탕으로 RAG 기반 추천
    """
    try:
        from src.conversation import get_conversation_session

        # 세션 가져오기
        conversation = get_conversation_session(session_id)

        # 수집된 관심사 추출
        interests = conversation.get_collected_interests()

        if not interests:
            raise HTTPException(
                status_code=400,
                detail="아직 충분한 정보가 수집되지 않았습니다. 대화를 더 진행해주세요."
            )

        # RAG 시스템으로 추천 생성
        rag = get_rag_system()
        result = rag.recommend_majors(interests, top_k=5)

        recommendation_id = str(uuid.uuid4())

        return RecommendationResponse(
            recommendation_id=recommendation_id,
            recommended_majors=result["recommended_majors"],
            reasoning=result["reasoning"],
            retrieved_context=result.get("retrieved_context")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 생성 실패: {str(e)}"
        )


@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """대화 세션 삭제"""
    try:
        from src.conversation import delete_conversation_session

        success = delete_conversation_session(session_id)

        if success:
            return {"status": "success", "message": f"세션 {session_id} 삭제 완료"}
        else:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"세션 삭제 실패: {str(e)}"
        )


@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """대화 세션의 히스토리 조회"""
    try:
        from src.conversation import get_conversation_session

        conversation = get_conversation_session(session_id)
        history = conversation.get_conversation_history()

        return {
            "session_id": session_id,
            "history": history,
            "conversation_count": conversation.collected_info["conversation_count"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"히스토리 조회 실패: {str(e)}"
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
