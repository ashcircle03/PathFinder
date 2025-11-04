# PathFinder

**LangChain 기반 고등학생 대학 학과 추천 서비스**

## 프로젝트 개요

PathFinder는 고등학생들의 관심사를 분석하여 적합한 대학 학과를 추천하는 AI 기반 진로 상담 서비스입니다.

### 기술 스택

- **Frontend**: React + Vite (Nginx)
- **API**: FastAPI
- **LangChain**: RAG 파이프라인 구성
- **LLM**: Ollama + EXAONE-3.5-7.8B (LG AI 한국어 네이티브)
- **Vector DB**: Qdrant
- **Embeddings**: HuggingFace Sentence-Transformers (한국어 특화)
- **컨테이너**: Docker Compose (4개 서비스)

---

## 주요 기능

### 🎯 LangChain RAG 시스템

- **Retrieval (검색)**: 사용자 관심사와 유사한 학과를 벡터 검색
- **Augmentation (증강)**: 검색된 학과 정보를 컨텍스트로 제공
- **Generation (생성)**: LLM이 맞춤형 추천 생성

### ✨ 핵심 특징

1. **환각 방지**: RAG로 실제 학과 정보만 추천
2. **한국어 최적화**: 한국어 특화 임베딩 + LLM
3. **구조화된 출력**: Pydantic Output Parser 사용
4. **유연한 프롬프트**: LangChain PromptTemplate

---

## 시작하기

### 사전 요구사항

- **Docker & Docker Compose** 설치
- **GPU 권장**: NVIDIA GPU (8GB VRAM 이상)
  - RTX 3060 12GB, RTX 4070, RTX 5070 등 최적
  - NVIDIA Container Toolkit 설치 필요
- 최소 16GB RAM (32GB 권장)
- 디스크 공간: 최소 10GB

### 설치 및 실행

#### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd PathFinder
```

#### 2. Docker Compose로 서비스 실행

```bash
docker-compose up -d
```

서비스가 시작됩니다:
- `frontend`: React 웹 UI (포트 3000) 🌐
- `api`: FastAPI 서버 (포트 8000)
- `ollama`: LLM 서버 (포트 11434)
- `qdrant`: Vector DB (포트 6333, 6334)

#### 3. LLM 모델 다운로드 (최초 1회)

```bash
# EXAONE-3.5-7.8B 모델 다운로드 (한국어 네이티브, 12GB GPU 최적화)
docker exec -it pathfinder-ollama ollama pull exaone3.5:7.8b

# 모델 확인
docker exec -it pathfinder-ollama ollama list
```

**VRAM 요구사항**: ~6GB (RTX 3060 12GB, RTX 4070, RTX 5070 등에 최적)

#### 4. Vector DB 초기화 (최초 1회)

```bash
curl -X POST http://localhost:8000/initialize-db
```

34개 학과 데이터가 임베딩되어 Qdrant에 저장됩니다.

#### 5. 웹 브라우저에서 접속

```
http://localhost:3000
```

브라우저에서 바로 학과 추천을 받을 수 있습니다! 🎓

또는 API를 직접 호출:

```bash
curl http://localhost:8000/health
```

---

## 사용 방법

### 💻 웹 UI 사용 (추천)

1. 브라우저에서 `http://localhost:3000` 접속
2. 관심사 입력 (예: "프로그래밍, AI, 수학")
3. "학과 추천 받기" 버튼 클릭
4. AI가 분석한 추천 학과와 상세 설명 확인

### 🔧 API 직접 호출

#### API 문서

서비스 실행 후:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트

#### 1. 학과 추천 (RAG)

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "interests": "프로그래밍, 게임 개발, 수학"
  }'
```

**응답 예시:**

```json
{
  "recommendation_id": "550e8400-e29b-41d4-a716-446655440000",
  "recommended_majors": [
    "컴퓨터공학과",
    "소프트웨어학과",
    "게임공학과",
    "인공지능학과",
    "데이터사이언스학과"
  ],
  "reasoning": "학생의 관심사인 프로그래밍과 게임 개발에 가장 적합한 학과들입니다...",
  "retrieved_context": [
    {
      "score": 0.85,
      "major_name": "컴퓨터공학과",
      "category": "공학",
      "description": "...",
      "keywords": ["프로그래밍", "코딩", "..."],
      "career_paths": ["소프트웨어 엔지니어", "..."]
    }
  ]
}
```

#### 2. 벡터 검색만 수행 (LLM 없음)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "interests": "예술, 미술, 디자인"
  }'
```

빠른 응답이 필요하거나 검색 결과만 확인하고 싶을 때 유용합니다.

#### 3. 개발자 도구

```bash
# 현재 프롬프트 템플릿 확인
curl http://localhost:8000/debug/prompt
```

---

## 프로젝트 구조

```
PathFinder/
├── src/
│   ├── main.py              # FastAPI 애플리케이션 (189줄)
│   ├── rag.py              # LangChain RAG 시스템 (258줄)
│   └── initialize_db.py    # Vector DB 초기화 (117줄)
├── data/
│   └── majors.json         # 학과 정보 (34개)
├── docker-compose.yml      # 서비스 오케스트레이션 (3개 서비스)
├── Dockerfile              # API 서버 이미지
├── requirements.txt        # Python 의존성
└── README.md
```

---

## LangChain 아키텍처

### RAG 파이프라인

```
사용자 관심사 입력
       ↓
[임베딩 변환] (HuggingFace Embeddings)
       ↓
[벡터 검색] (Qdrant VectorStore)
       ↓
검색된 학과 정보 (Top 5)
       ↓
[프롬프트 구성] (PromptTemplate)
       ↓
[LLM 생성] (Ollama)
       ↓
[출력 파싱] (PydanticOutputParser)
       ↓
구조화된 추천 결과
```

### 주요 컴포넌트

#### 1. **VectorStore** (langchain_qdrant)

```python
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="majors",
    url=qdrant_host
)
```

#### 2. **Embeddings** (HuggingFace)

```python
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",  # 한국어 특화
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 3. **LLM** (Ollama)

```python
llm = OllamaLLM(
    model="exaone3.5:7.8b",  # LG AI 한국어 네이티브
    base_url=ollama_host,
    temperature=0.7,
    system="Korean university counselor (pure Hangul)"
)
```

#### 4. **Output Parser** (Pydantic)

```python
class MajorRecommendation(BaseModel):
    recommended_majors: List[str]
    reasoning: str

parser = PydanticOutputParser(pydantic_object=MajorRecommendation)
```

---

## 학과 데이터

**34개 학과** 포함:
- 공학 (컴퓨터, 소프트웨어, AI, 전자, 기계 등)
- 상경 (경영, 경제, 회계 등)
- 의료 (의학, 간호 등)
- 교육 (교육학 등)
- 예술 (디자인, 음악 등)

각 학과 정보:
- 이름, 분야, 설명
- 키워드 (10개)
- 진로 (5개)
- 관련 과목, 필요 역량

---

## 개발

### 로컬 개발 (Docker 없이)

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Ollama 설치 및 모델 다운로드
# https://ollama.ai에서 Ollama 설치
ollama pull exaone3.5:7.8b

# Qdrant 실행 (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Vector DB 초기화
python src/initialize_db.py

# 서버 실행
uvicorn src.main:app --reload
```

### 로그 확인

```bash
# 전체 로그
docker-compose logs -f

# API 서버만
docker-compose logs -f api

# Ollama만
docker-compose logs -f ollama
```

### 서비스 중지

```bash
docker-compose down

# 볼륨까지 삭제 (모델 캐시, Vector DB 포함)
docker-compose down -v
```

---

## LangChain의 장점

### ✅ 이 프로젝트에서 활용한 기능

1. **VectorStore 추상화**
   - Qdrant, Pinecone, Chroma 등 쉽게 교체 가능
   - 일관된 인터페이스

2. **PromptTemplate**
   - 재사용 가능한 프롬프트
   - 변수 주입, 검증

3. **Output Parser**
   - 구조화된 출력 보장
   - 자동 재시도 (파싱 실패 시)

4. **Embeddings 통합**
   - OpenAI, HuggingFace, Cohere 등
   - 통일된 인터페이스

5. **Document 모델**
   - `page_content` + `metadata` 구조
   - 검색 및 필터링 용이

### 🎯 향후 확장 가능성

- **ConversationChain**: 대화형 상담
- **Agent**: 여러 도구 조합 (웹 검색, 계산기 등)
- **Memory**: 대화 이력 관리
- **LangSmith**: 프로덕션 모니터링

---

## 트러블슈팅

### Ollama 연결 실패

```bash
# Ollama 상태 확인
docker logs pathfinder-ollama

# 헬스체크 대기 (최대 1-2분)
```

### 메모리/VRAM 부족

- Docker Desktop 메모리 할당 증가 (최소 16GB)
- 더 작은 모델 사용 (`KOREAN_LLM_GUIDE.md` 참고):
  - `yanolja/EEVE-Korean-10.8B` (8GB VRAM)
  - `exaone3.5:7.8b` (6GB VRAM)

### 모델 다운로드 느림

- 첫 실행 시 18-20GB 모델 다운로드
- 네트워크에 따라 30분~1시간 소요

### GPU 사용 확인

```bash
# Ollama GPU 사용 확인
docker logs pathfinder-ollama | grep GPU

# NVIDIA Docker Runtime 필요
nvidia-smi
```

---

## 로드맵

### ✅ Phase 1: LangChain RAG (완료)
- [x] LangChain 기반 RAG 시스템
- [x] Qdrant VectorStore 통합
- [x] Pydantic Output Parser
- [x] PromptTemplate 관리

### 🚧 Phase 2: 기능 확장 (진행 중)
- [ ] 대화형 상담 (ConversationChain)
- [ ] 프롬프트 버전 관리
- [ ] 사용자 피드백 수집

### 📅 Phase 3: 프로덕션 (계획)
- [ ] LangSmith 통합
- [ ] 캐싱 및 성능 최적화
- [ ] A/B 테스팅

---

## 라이선스

MIT License

---

## 기술 블로그

이 프로젝트는 다음 개념을 학습하기 위한 목적으로 만들어졌습니다:

- **RAG (Retrieval-Augmented Generation)**
- **LangChain 프레임워크**
- **Vector Databases**
- **한국어 NLP**
- **Docker 기반 마이크로서비스**

---

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [Qdrant 공식 문서](https://qdrant.tech/)
- [Ollama 공식 사이트](https://ollama.ai/)
- [HuggingFace Sentence-Transformers](https://www.sbert.net/)
