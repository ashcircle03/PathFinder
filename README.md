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
- **컨테이너**: Docker Compose + Kubernetes (Minikube)

---

## 주요 기능

### 🎯 LangChain RAG 시스템

- **Retrieval (검색)**: 사용자 관심사와 유사한 학과를 벡터 검색
- **Augmentation (증강)**: 검색된 학과 정보를 컨텍스트로 제공
- **Generation (생성)**: LLM이 맞춤형 추천 생성

### 💬 대화형 진로 상담 (NEW!)

- **ConversationChain**: 학생과 자연스러운 대화를 통한 관심사 파악
- **Memory 관리**: 이전 대화 내용을 기억하며 맥락있는 상담 제공
- **지능형 정보 수집**: 관심사, 과목, 성격, 진로 목표 자동 추출
- **세션 관리**: 각 사용자별 독립적인 대화 세션 유지

### 🏫 실제 대학 정보 기반 추천 (NEW!)

- **대학 크롤러**: 주요 대학 입학처 정보 수집
- **상세 정보**: 커리큘럼, 입학 정원, 졸업 요건, 웹사이트 링크
- **10개 대학 x 5개 학과**: 총 50개 실제 대학 학과 정보
- **Vector DB 통합**: 기본 학과(34개) + 대학 학과(50개) = 84개 벡터

### ✨ 핵심 특징

1. **환각 방지**: RAG로 실제 학과 정보만 추천
2. **한국어 최적화**: 한국어 특화 임베딩 + LLM
3. **구조화된 출력**: Pydantic Output Parser 사용
4. **유연한 프롬프트**: LangChain PromptTemplate
5. **대화형 UX**: 자연스러운 대화를 통한 사용자 친화적 인터페이스

---

## 배포 방법

PathFinder는 두 가지 배포 방식을 지원합니다:

### 옵션 1: Docker Compose (로컬 개발용)

**사전 요구사항:**
- Docker & Docker Compose 설치
- GPU 권장: NVIDIA GPU (8GB VRAM 이상)
- 최소 16GB RAM (32GB 권장)

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

### 옵션 2: Kubernetes (Minikube)

**사전 요구사항:**
- Minikube (v1.30 이상)
- kubectl
- Docker
- 최소 12GB RAM 할당

#### 빠른 시작

```bash
# Minikube 시작
minikube start --driver=docker --memory=12288 --cpus=4

# Docker 환경 설정
eval $(minikube docker-env)

# 이미지 빌드
docker build -t pathfinder-api:latest .
docker build -t pathfinder-frontend:latest ./frontend

# 배포
cd k8s
./deploy.sh

# Frontend 접속
minikube service frontend-service -n pathfinder
```

**상세 가이드**: [k8s/README.md](k8s/README.md)

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

#### 3. 대화형 진로 상담 (NEW!)

**대화 시작:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요! 진로 상담 받고 싶어요",
    "session_id": "user-12345"
  }'
```

**응답 예시:**

```json
{
  "session_id": "user-12345",
  "response": "안녕하세요! 진로 상담을 도와드리겠습니다. 먼저 어떤 분야에 관심이 있으신가요?",
  "is_ready_to_recommend": false,
  "conversation_count": 1,
  "collected_info": {
    "interests": [],
    "subjects": [],
    "conversation_count": 1
  }
}
```

**대화 계속:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "저는 프로그래밍하고 게임 만드는 걸 좋아해요",
    "session_id": "user-12345"
  }'
```

**학과 추천 받기 (3-5회 대화 후):**

```bash
curl -X POST http://localhost:8000/chat/user-12345/recommend
```

**대화 이력 조회:**

```bash
curl http://localhost:8000/chat/user-12345/history
```

**세션 삭제:**

```bash
curl -X DELETE http://localhost:8000/chat/user-12345
```

#### 4. 개발자 도구

```bash
# 현재 프롬프트 템플릿 확인
curl http://localhost:8000/debug/prompt
```

---

## 프로젝트 구조

```
PathFinder/
├── src/
│   ├── main.py                # FastAPI 애플리케이션 (337줄)
│   ├── rag.py                 # LangChain RAG 시스템 (262줄)
│   ├── conversation.py        # 대화형 상담 시스템 (336줄) NEW!
│   ├── university_crawler.py  # 대학 정보 크롤러 (312줄) NEW!
│   └── initialize_db.py       # Vector DB 초기화 (192줄)
├── data/
│   ├── majors.json            # 기본 학과 정보 (34개)
│   └── university_departments.json  # 대학 학과 정보 (50개) NEW!
├── frontend/
│   ├── src/
│   │   ├── components/        # React 컴포넌트 NEW!
│   │   │   ├── ChatInterface.jsx       # 대화형 상담 UI
│   │   │   └── RecommendationDetail.jsx # 추천 결과 상세
│   │   ├── App.jsx
│   │   └── App.css
│   ├── nginx.conf             # Nginx 프록시 설정
│   └── Dockerfile
├── k8s/                       # Kubernetes 매니페스트 NEW!
│   ├── 00-namespace.yaml      # 네임스페이스
│   ├── 01-configmap.yaml      # 환경 설정
│   ├── 02-pvc-ollama.yaml     # Ollama 영구 볼륨
│   ├── 03-pvc-qdrant.yaml     # Qdrant 영구 볼륨
│   ├── 04-deployment-ollama.yaml
│   ├── 05-service-ollama.yaml
│   ├── 06-deployment-qdrant.yaml
│   ├── 07-service-qdrant.yaml
│   ├── 08-deployment-api.yaml
│   ├── 09-service-api.yaml
│   ├── 10-deployment-frontend.yaml
│   ├── 11-service-frontend.yaml
│   ├── deploy.sh              # 자동 배포 스크립트
│   ├── cleanup.sh             # 정리 스크립트
│   └── README.md              # Kubernetes 배포 가이드
├── docker-compose.yml         # 서비스 오케스트레이션 (4개 서비스)
├── Dockerfile                 # API 서버 이미지
├── requirements.txt           # Python 의존성
└── README.md
```

---

## LangChain 아키텍처

### RAG 파이프라인

```
사용자 관심사 입력
       ↓
[임베딩 변환] (HuggingFaceEmbeddings)
       ↓
[벡터 검색] (Qdrant VectorStore)
       ↓
검색된 학과 정보 (Top 5) - 기본 34개 + 대학 50개
       ↓
[프롬프트 구성] (PromptTemplate)
       ↓
[LLM 생성] (Ollama - EXAONE-3.5-7.8B)
       ↓
[출력 파싱] (PydanticOutputParser)
       ↓
구조화된 추천 결과
```

### 대화형 상담 파이프라인 (NEW!)

```
사용자 메시지
       ↓
[ConversationChain] (LangChain)
   ├─ PromptTemplate (상담사 페르소나)
   ├─ ConversationBufferMemory (대화 이력 저장)
   └─ OllamaLLM (한국어 대화 생성)
       ↓
상담사 응답 + 정보 추출
   ├─ 관심사 키워드 추출
   ├─ 과목 정보 수집
   └─ 대화 횟수 카운트
       ↓
충분한 정보 수집 판단
   ├─ NO → 대화 계속
   └─ YES → RAG 파이프라인으로 학과 추천
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

6. **ConversationChain + Memory** (NEW!)
   - 대화 이력 관리
   - 맥락있는 상담 제공
   - 세션별 독립적인 메모리

### 🎯 향후 확장 가능성

- **LangSmith**: 프로덕션 모니터링 및 추적
- **Agent**: 여러 도구 조합 (웹 검색, 계산기 등)
- **ConversationSummaryMemory**: 긴 대화 요약
- **실제 웹 크롤링**: 주요 대학 입학처 실시간 정보 수집
- **Redis 기반 세션 관리**: 분산 환경 지원

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
- [x] 기본 학과 정보 (34개) 벡터화

### ✅ Phase 2: 기능 확장 (완료)
- [x] 대화형 상담 (ConversationChain + Memory)
- [x] 대학 입학처 정보 크롤러
- [x] 실제 대학 학과 데이터 통합 (50개)
- [x] 세션 기반 대화 관리

### ✅ Phase 3: UX 및 인프라 개선 (완료)
- [x] Frontend 채팅 UI 구현 (ChatInterface)
- [x] 대화 히스토리 시각화
- [x] 추천 결과 상세 페이지 (RecommendationDetail)
- [x] Kubernetes 배포 환경 구축 (Minikube)
- [x] FastAPI lifespan 패턴 적용
- [x] 반응형 디자인 적용

### 📅 Phase 4: 프로덕션 (계획)
- [ ] LangSmith 통합
- [ ] Redis 기반 세션 관리
- [ ] 실시간 웹 크롤링
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
