from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import os
from typing import List

app = FastAPI(
    title="PathFinder API",
    description="고등학생 대학 학과 매칭 서비스",
    version="0.1.0"
)

# 환경 변수
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Ollama 클라이언트 설정
client = ollama.Client(host=OLLAMA_HOST)


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

    prompt = f"""당신은 진로 상담 전문가입니다. 고등학생의 관심사를 분석하여 적합한 대학 학과를 추천해주세요.

학생의 관심사: {request.interests}

다음 형식으로 답변해주세요:
1. 추천 학과 (3-5개, 쉼표로 구분)
2. 추천 이유 (각 학과가 학생의 관심사와 어떻게 연결되는지 설명)

추천 학과:"""

    try:
        # Ollama를 통해 LLM 호출
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
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
