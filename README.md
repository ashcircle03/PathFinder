# PathFinder

고등학생 대상 대학 학과 추천 서비스. 관심사 직접 입력, 학생부 PDF 분석, 대화형 상담 세 가지 방식으로 LangChain RAG 파이프라인을 통해 학과를 추천한다.

## 기술 스택

**Backend**
- Python, FastAPI 0.115, uvicorn, Pydantic v2
- LangChain 0.3 (langchain-core, langchain-community, langchain-qdrant, langchain-ollama, langchain-huggingface)

**LLM / Embedding**
- Ollama (로컬 LLM 서버), 기본 모델: `exaone3.5:7.8b` (`OLLAMA_MODEL` 환경변수로 변경 가능)
- HuggingFace sentence-transformers: `jhgan/ko-sroberta-multitask` (768차원, 한국어 특화)

**Vector DB / Retrieval**
- Qdrant – 84개 학과 벡터 (일반 34개 + 대학별 50개)
- EnsembleRetriever: Qdrant 벡터 검색 60% + BM25 키워드 검색 40% (rank-bm25)

**PDF 처리**
- PyPDF2 (텍스트 추출), pdf2image + pytesseract (OCR 폴백)

**Frontend**
- React 18, Vite, axios, Nginx

**인프라**
- Docker Compose (로컬), Kubernetes / Minikube (배포)

**평가 / 추적**
- ragas 0.1.17 (RAG 평가), LangSmith (LLM 추적, 선택)

## 주요 기능

### 1. 직접 입력 추천 (`POST /recommend`)
관심사 텍스트 → EnsembleRetriever(벡터+BM25) 검색 → Ollama LLM 생성 → JSON 응답

### 2. 학생부 PDF 분석 (`POST /analyze-and-recommend`)
PDF 업로드 → PyPDF2/OCR 텍스트 추출 → 규칙 기반 프로필 파싱 → 동일한 RAG 흐름

### 3. 대화형 상담 (`POST /chat`)
ConversationChain + ConversationSummaryBufferMemory로 세션 유지, 5~10회 대화 후 Pydantic 기반 프로필 추출, confidence_score ≥ 0.75 시 추천 전환

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 헬스 체크 |
| POST | `/recommend` | 관심사 직접 입력 추천 |
| POST | `/initialize-db` | Qdrant 벡터 DB 초기화 |
| POST | `/chat` | 대화형 상담 |
| POST | `/chat/{session_id}/recommend` | 대화 기반 추천 |
| GET | `/chat/{session_id}/history` | 대화 이력 조회 |
| DELETE | `/chat/{session_id}` | 세션 삭제 |
| POST | `/analyze-school-record` | 학생부 PDF 분석 |
| POST | `/analyze-and-recommend` | 학생부 분석 + 추천 (원스텝) |

Swagger UI: `http://localhost:8000/docs`

## 프로젝트 구조

```
PathFinder/
├── src/
│   ├── main.py                # FastAPI 앱, 라우터
│   ├── rag.py                 # RAG 시스템 (EnsembleRetriever + OllamaLLM)
│   ├── conversation.py        # 대화형 상담 (ConversationChain + Memory)
│   ├── school_record_parser.py # 학생부 PDF 파싱 (PyPDF2 + OCR)
│   └── initialize_db.py       # Qdrant 초기화 (학과 데이터 임베딩)
├── data/
│   ├── majors.json            # 일반 학과 34개
│   └── university_departments.json  # 대학별 학과 50개
├── frontend/                  # React + Vite
├── k8s/                       # Kubernetes 매니페스트
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 실행 방법

### Docker Compose (로컬)

```bash
docker-compose up -d

# 최초 1회: 모델 다운로드
docker exec -it pathfinder-ollama ollama pull exaone3.5:7.8b

# 최초 1회: Vector DB 초기화
curl -X POST http://localhost:8000/initialize-db
```

서비스 포트:
- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Qdrant: `http://localhost:6333`
- Ollama: `http://localhost:11434`

시스템 요구사항:
- RAM 16GB 이상
- NVIDIA GPU, VRAM 6GB 이상 (권장)

### Kubernetes (Minikube)

```bash
minikube start --driver=docker --memory=12288 --cpus=4
eval $(minikube docker-env)
docker build -t pathfinder-api:latest .
docker build -t pathfinder-frontend:latest ./frontend
cd k8s && ./deploy.sh
minikube service frontend-service -n pathfinder
```

상세 가이드: [k8s/README.md](k8s/README.md)

### 로컬 개발 (Docker 없이)

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Ollama 별도 설치 후 모델 다운로드
ollama pull exaone3.5:7.8b

# Qdrant 실행
docker run -p 6333:6333 qdrant/qdrant

# Vector DB 초기화
python src/initialize_db.py

# 서버 실행
uvicorn src.main:app --reload
```

## 환경 변수

```
QDRANT_HOST=http://qdrant:6333
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=exaone3.5:7.8b
LANGCHAIN_API_KEY=              # LangSmith 사용 시 설정
```

## 트러블슈팅

- **Ollama 연결 실패**: `docker logs pathfinder-ollama` 확인, 시작 후 1~2분 대기
- **VRAM 부족**: `exaone3.5:7.8b`는 약 6GB VRAM 필요, 부족 시 더 작은 모델로 `OLLAMA_MODEL` 변경
- **모델 다운로드**: 최초 실행 시 약 5GB 다운로드, 네트워크 환경에 따라 수십 분 소요
- **GPU 확인**: `docker logs pathfinder-ollama | grep GPU`, NVIDIA Docker Runtime 필요

## 라이선스

MIT
