# PathFinder 아키텍처

## 시스템 구성

```
Frontend (React + Nginx)
        │ HTTP
        ▼
API Server (FastAPI)
        │
        ▼
RAG System (rag.py)
   EnsembleRetriever: Vector(60%) + BM25(40%)
        │                     │
        ▼                     ▼
Qdrant (벡터 검색)      Ollama (LLM 생성)
84개 학과 벡터          exaone3.5:7.8b
```

## 입력 방식별 처리 흐름

### 1. 직접 입력 (`/recommend`)
```
텍스트 입력
→ HuggingFaceEmbeddings (jhgan/ko-sroberta-multitask)
→ EnsembleRetriever (Qdrant 벡터 60% + BM25 키워드 40%)
→ 상위 5개 학과 컨텍스트 구성
→ PromptTemplate 포맷
→ OllamaLLM 생성
→ PydanticOutputParser 파싱
→ JSON 응답
```

### 2. 학생부 PDF 분석 (`/analyze-and-recommend`)
```
PDF 업로드
→ PyPDF2 텍스트 추출 (실패 시 pdf2image + pytesseract OCR)
→ 규칙 기반 프로필 파싱 (학업 강점, 비교과, 성격, 진로)
→ 프로필 → 관심사 텍스트 변환
→ /recommend 와 동일한 RAG 흐름
```

### 3. 대화형 상담 (`/chat`)
```
사용자 메시지
→ ConversationChain (LangChain)
   - PromptTemplate (상담사 페르소나, 4가지 정보 수집 목표)
   - ConversationSummaryBufferMemory (max_token_limit=500)
   - OllamaLLM
→ 2회차부터: PydanticOutputParser로 StudentProfile 추출
→ confidence_score >= 0.75 && categories >= 3 && 대화 >= 5회 → 추천 가능
→ 10회 대화 달성 시 강제 추천
→ /recommend 와 동일한 RAG 흐름
```

## 파일별 역할

| 파일 | 역할 |
|------|------|
| `src/main.py` | FastAPI 앱, 라우터, lifespan 관리 |
| `src/rag.py` | EnsembleRetriever + OllamaLLM + PydanticOutputParser |
| `src/conversation.py` | ConversationChain + ConversationSummaryBufferMemory |
| `src/school_record_parser.py` | PyPDF2/OCR 텍스트 추출, 규칙 기반 프로필 파싱 |
| `src/initialize_db.py` | majors.json + university_departments.json → Qdrant 임베딩 저장 |

## 검색 설계 (EnsembleRetriever)

| 방식 | 비중 | 특징 |
|------|------|------|
| Qdrant 벡터 검색 | 60% | 의미 유사도 (예: "코딩 좋아해요" → 컴퓨터공학과) |
| BM25 키워드 검색 | 40% | 정확한 키워드 매칭 (예: "AI" → 인공지능학과) |

벡터 데이터: 일반 학과 34개 + 대학별 학과 50개 = 84개

## 배포 구조 (Kubernetes)

```
namespace: pathfinder

Deployment: ollama  (1 replica) → Service: ClusterIP:11434  PVC: 20Gi
Deployment: qdrant  (1 replica) → Service: ClusterIP:6333   PVC: 5Gi
Deployment: api     (2 replicas) → Service: ClusterIP:8000
Deployment: frontend(2 replicas) → Service: NodePort:30080
```

## 설계 결정

| 결정 | 이유 |
|------|------|
| EnsembleRetriever (벡터+BM25) | 벡터 단독보다 키워드 매칭 정확도 향상 |
| 규칙 기반 PDF 파싱 | LLM 호출 없이 빠른 처리, 일관된 결과 |
| ko-sroberta 임베딩 | 한국어 의미 검색 (영어 모델로는 한국어 동의어 연결 불가) |
| Ollama 로컬 LLM | API 비용 없음, 데이터 외부 전송 없음 (단: GPU 필요) |
| ConversationSummaryBufferMemory | 긴 대화에서 토큰 절약, 최근 500토큰 + 이전 내용 요약 유지 |
