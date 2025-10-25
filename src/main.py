from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import os
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="PathFinder API",
    description="고등학생 대학 학과 매칭 서비스",
    version="0.2.0"
)

# 환경 변수
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Ollama 클라이언트 설정
client = ollama.Client(host=OLLAMA_HOST)

# RAG 시스템 (지연 로딩)
rag_system = None


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
    recommended_majors: List[str]
    reasoning: str
    retrieved_context: Optional[List[Dict[str, Any]]] = None


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
    """RAG 기반 학과 추천 (검색 증강 생성)"""
    try:
        # RAG 시스템 가져오기
        rag = get_rag_system()

        # RAG 기반 추천 실행
        result = rag.recommend_majors(request.interests, top_k=5)

        return RAGRecommendationResponse(
            recommended_majors=result["recommended_majors"],
            reasoning=result["reasoning"],
            retrieved_context=result.get("retrieved_context")
        )

    except Exception as e:
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
