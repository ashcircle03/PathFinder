# PathFinder 아키텍처 설명

## 🎯 핵심 문제 정의

> **"고등학생은 자신이 뭘 원하는지, 뭘 잘하는지 모른다"**

이 문제를 해결하기 위해 3가지 접근 방식을 구현했습니다.

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (React)                          │
│                     pathfinder-frontend:v2.3.2                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP (Port 8000)
┌─────────────────────────────────────────────────────────────────────┐
│                          API Server (FastAPI)                        │
│                       pathfinder-api:v2.5.0                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │  /recommend │  │    /chat    │  │  /analyze-and-recommend     │  │
│  │  직접 입력  │  │  대화 상담  │  │    PDF 자동 분석            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                   │                        │
          └───────────────────┼────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG System (rag.py)                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Ensemble Retriever                         │   │
│  │         Vector Search (60%) + BM25 Keyword (40%)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
┌─────────────────────┐                  ┌─────────────────────┐
│   Qdrant (Vector)   │                  │   Ollama (LLM)      │
│  qdrant/qdrant      │                  │  ollama/ollama      │
│  - 84개 학과 벡터   │                  │  - exaone3.5:7.8b   │
│  - ko-sroberta      │                  │  - 추천 생성        │
└─────────────────────┘                  └─────────────────────┘
```

---

## 📁 파일별 역할과 필요성

### 1. `src/rag.py` - RAG 핵심 엔진

**필요성**: LLM만으로는 최신/정확한 학과 정보를 제공할 수 없음 (환각 문제)

**해결**: 검색 → 컨텍스트 → LLM 생성 (RAG 패턴)

```python
# 구성요소와 도입 이유
class MajorRecommendationRAG:
    
    # 1. 임베딩 모델: ko-sroberta-multitask
    #    → 필요성: 한국어 의미 검색 (영어 모델로는 "프로그래밍" ≠ "코딩" 연결 불가)
    
    # 2. Ensemble Retriever (Vector 60% + BM25 40%)
    #    → 필요성: 
    #       - Vector: "프로그래밍 좋아해요" → 컴퓨터공학과 (의미 유사도)
    #       - BM25: "AI" → 인공지능학과 (정확한 키워드 매칭)
    #       - 두 방식 결합으로 정확도 향상
    
    # 3. Ollama LLM
    #    → 필요성: 검색 결과를 자연어로 설명 + 맞춤 추천
```

### 2. `src/school_record_parser.py` - PDF 파서

**필요성**: 학생이 자신의 관심사를 모를 때 → 학교생활기록부에서 자동 추출

**해결**: 규칙 기반 텍스트 분석 (LLM 의존성 최소화)

```python
# 구성요소와 도입 이유
class SchoolRecordParser:
    
    # 1. PyPDF2 → pdf2image + pytesseract (OCR)
    #    → 필요성: 대부분의 학생부가 스캔 PDF (텍스트 추출 불가)
    
    # 2. 규칙 기반 추출 (LLM 대신)
    #    → 필요성: 
    #       - LLM 호출 비용 절감
    #       - 응답 속도 향상 (2초 → 0.5초)
    #       - 일관된 결과 보장
    
    # 3. _clean_text() 메서드
    #    → 필요성: OCR 오류 정제 (불필요한 공백, 특수문자)
```

### 3. `src/conversation.py` - 대화형 상담

**필요성**: 직접 입력도 어려운 학생을 위한 점진적 정보 수집

**해결**: 상담사 역할 LLM + Memory로 맥락 유지

```python
# 구성요소와 도입 이유
class CareerCounselorConversation:
    
    # 1. ConversationSummaryBufferMemory
    #    → 필요성: 긴 대화에서 컨텍스트 유지 + 토큰 절약
    
    # 2. 구조화된 프로필 추출 (Pydantic)
    #    → 필요성: "게임 좋아해요" 같은 막연한 답변을 구조화
    
    # 3. confidence_score 기반 추천 판단
    #    → 필요성: 충분한 정보 없이 추천하면 부정확
```

### 4. `src/initialize_db.py` - 벡터 DB 초기화

**필요성**: 학과 정보를 벡터로 변환하여 Qdrant에 저장

```python
# 데이터 흐름
majors.json (28개 일반 학과)
    +
university_departments.json (56개 대학별 학과)
    ↓
Document 변환 (page_content + metadata)
    ↓
HuggingFace 임베딩 (768차원 벡터)
    ↓
Qdrant 저장 (84개 벡터)
```

---

## 🔄 데이터 흐름

### 시나리오 1: 직접 입력 (`/recommend`)
```
"프로그래밍, 수학 좋아해요"
    ↓
Ensemble Retriever (Vector + BM25)
    ↓
검색 결과: 컴퓨터공학과, 소프트웨어학과, 데이터사이언스학과...
    ↓
LLM 프롬프트: "다음 학과 중 추천하세요: {...}"
    ↓
JSON 응답: {"recommended_majors": [...], "reasoning": "..."}
```

### 시나리오 2: PDF 분석 (`/analyze-and-recommend`)
```
학교생활기록부.pdf
    ↓
PyPDF2 텍스트 추출 (실패 시 OCR)
    ↓
규칙 기반 프로필 추출:
  - 학업 강점: 수학, 과학
  - 비교과: 수학동아리, 올림피아드
  - 성격: 리더십, 창의성
    ↓
프로필 → 관심사 텍스트 변환
    ↓
/recommend와 동일한 RAG 흐름
```

### 시나리오 3: 대화 상담 (`/chat`)
```
학생: "진로 모르겠어요"
상담사: "최근에 재밌었던 게 뭐에요?"
학생: "게임이요"
상담사: "어떤 게임? 전략? 롤플레잉?"
    ...
(5-10회 대화 후)
    ↓
구조화된 프로필 추출
    ↓
confidence_score >= 0.75 확인
    ↓
/recommend와 동일한 RAG 흐름
```

---

## ⚙️ K8s 배포 구조

```yaml
namespace: pathfinder

pods:
  - api (x2): 
      image: pathfinder-api:v2.5.0
      통신: → qdrant-service:6333, → ollama-service:11434
      
  - frontend (x2): 
      image: pathfinder-frontend:v2.3.2
      통신: → api-service:8000
      
  - qdrant (x1): 
      image: qdrant/qdrant
      storage: PVC (10Gi)
      
  - ollama (x1): 
      image: ollama/ollama
      storage: PVC (50Gi, 모델 저장)

services:
  - api-service: ClusterIP:8000
  - frontend-service: NodePort:30080
  - qdrant-service: ClusterIP:6333
  - ollama-service: ClusterIP:11434
```

---

## 📊 설계 결정 및 트레이드오프

| 결정 | 이유 | 트레이드오프 |
|------|------|-------------|
| Ensemble Retriever | Vector만 사용 시 키워드 매칭 약함 | 처리 시간 증가 |
| 규칙 기반 PDF 파싱 | LLM 비용/속도 절약 | 복잡한 문서 분석 제한 |
| ko-sroberta 임베딩 | 한국어 특화 | 이미지 크기 증가 (8GB) |
| Ollama 로컬 LLM | API 비용 없음, 프라이버시 | GPU 필요, 응답 느림 |
| ConversationSummaryBufferMemory | 긴 대화 지원 | 요약 시 정보 손실 가능 |

---

## 🔧 향후 개선 가능 영역

1. **이미지 크기 최적화**: CPU 전용 PyTorch로 교체 (8GB → 3GB)
2. **캐싱**: 동일 쿼리에 대한 검색 결과 캐싱
3. **비동기 처리**: PDF OCR을 백그라운드 작업으로 분리
4. **A/B 테스트**: 다양한 Retriever 가중치 실험
