# PathFinder

고등학생 대학 학과 매칭 서비스 - RAG & LLMOps 기반

## 프로젝트 개요

PathFinder는 고등학생들의 관심사를 분석하여 적합한 대학 학과를 추천하는 AI 기반 진로 상담 서비스입니다.

### 기술 스택

- **LLM**: Ollama + Qwen2.5:32b (한국어 성능 우수)
- **RAG**: Qdrant (Vector DB) + Sentence-Transformers (한국어 임베딩)
- **API**: FastAPI
- **컨테이너**: Docker, Docker Compose
- **향후 계획**: LLMOps, Kubernetes

## 시작하기

### 사전 요구사항

- Docker & Docker Compose 설치
- **GPU 필수**: NVIDIA GPU (RTX 3060 12GB 이상 권장, RTX 4070 최적)
- **NVIDIA Container Toolkit** 설치 (GPU 사용)
  ```bash
  # Ubuntu/Debian
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```
- 최소 16GB RAM (32GB 권장)
- 디스크 공간: 최소 20GB (모델 다운로드용)

### 설치 및 실행

1. **프로젝트 클론**
```bash
git clone <repository-url>
cd PathFinder
```

2. **환경 변수 설정** (선택사항)
```bash
cp .env.example .env
```

3. **Docker Compose로 서비스 실행**
```bash
docker-compose up -d
```

4. **LLM 모델 다운로드** (최초 1회)
```bash
curl -X POST http://localhost:8000/pull-model
```

5. **벡터 DB 초기화** (최초 1회, RAG 기능 사용 시)
```bash
curl -X POST http://localhost:8000/initialize-db
```

6. **서비스 상태 확인**
```bash
# 기본 헬스 체크
curl http://localhost:8000/health

# RAG 시스템 헬스 체크
curl http://localhost:8000/rag-health
```

### API 사용 예시

#### 1. 기본 학과 추천 (LLM만 사용)
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "interests": "프로그래밍, 게임 개발, 수학"
  }'
```

#### 2. RAG 기반 학과 추천 (검색 증강 생성) ⭐ 추천
```bash
curl -X POST http://localhost:8000/recommend-rag \
  -H "Content-Type: application/json" \
  -d '{
    "interests": "프로그래밍, 게임 개발, 수학"
  }'
```

#### 응답 예시
```json
{
  "recommended_majors": [
    "컴퓨터공학과",
    "소프트웨어학과",
    "게임공학과",
    "인공지능학과",
    "데이터사이언스학과"
  ],
  "reasoning": "검색된 학과 정보를 바탕으로 학생의 관심사와 가장 잘 맞는 학과를 추천합니다...",
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

### API 문서

서비스 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
PathFinder/
├── src/
│   ├── main.py          # FastAPI 애플리케이션
│   ├── rag.py           # RAG 시스템 (검색 증강 생성)
│   └── initialize_db.py # Vector DB 초기화 스크립트
├── data/
│   └── majors.json      # 학과 정보 데이터 (34개 학과)
├── models/              # LLM 모델 캐시 (자동 생성)
├── Dockerfile           # API 서버 이미지
├── docker-compose.yml   # 서비스 오케스트레이션 (Ollama + Qdrant + API)
├── requirements.txt     # Python 의존성
└── README.md
```

## 로드맵

### Phase 1: 기본 LLM 구동 ✅
- [x] Docker 환경 구축
- [x] Ollama + Llama 3.2 연동
- [x] FastAPI 서버 구현
- [x] 기본 학과 추천 기능

### Phase 2: RAG 구현 ✅ (현재)
- [x] Vector DB (Qdrant) 연동
- [x] 학과 정보 임베딩 (34개 학과, 한국어 모델)
- [x] 검색 기반 추천 개선
- [x] RAG 엔드포인트 구현
- [x] 벡터 검색 + LLM 생성 파이프라인

### Phase 3: LLMOps (예정)
- [ ] MLflow 실험 추적
- [ ] 프롬프트 버전 관리
- [ ] 모니터링 (Prometheus + Grafana)

### Phase 4: Kubernetes (예정)
- [ ] vLLM 전환
- [ ] K8s 매니페스트 작성
- [ ] HPA 오토스케일링

## RAG 시스템 소개

PathFinder는 **RAG (Retrieval-Augmented Generation)** 기술을 활용하여 더 정확하고 신뢰할 수 있는 학과 추천을 제공합니다.

### RAG의 장점

1. **정확성 향상**: 벡터 DB에 저장된 실제 학과 정보를 기반으로 추천
2. **환각(Hallucination) 방지**: LLM이 존재하지 않는 학과나 잘못된 정보를 생성하는 것을 방지
3. **맥락 기반 추천**: 학생의 관심사와 유사도가 높은 학과를 우선적으로 추천
4. **확장 가능성**: 새로운 학과 정보를 쉽게 추가하고 업데이트 가능

### 작동 원리

1. **임베딩**: 학과 정보를 벡터로 변환하여 Qdrant에 저장
2. **검색**: 학생의 관심사를 벡터로 변환하고 유사한 학과 검색
3. **생성**: 검색된 학과 정보를 컨텍스트로 LLM에 전달하여 맞춤형 추천 생성

### 사용된 모델

- **LLM**: Qwen2.5:32b (32B 파라미터, 한국어 성능 우수)
  - 다국어 지원 (한국어, 영어, 중국어 등)
  - 4bit 양자화 버전 자동 사용 (~10GB VRAM)
  - RTX 4070 12GB에 최적화
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (한국어 특화)
- **Vector DB**: Qdrant

## 개발

### 로컬 개발 (Docker 없이)
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Ollama 로컬 설치 필요 (https://ollama.ai)
ollama pull qwen2.5:32b

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

# 볼륨까지 삭제 (모델 캐시 포함)
docker-compose down -v
```

## 트러블슈팅

### Ollama 연결 실패
- `docker-compose logs ollama`로 Ollama 상태 확인
- 헬스체크 대기 (최대 1-2분 소요)

### 메모리/VRAM 부족
- Docker Desktop 메모리 할당 증가 (최소 16GB)
- GPU 메모리 부족 시 더 작은 모델 사용:
  - `qwen2.5:14b` (8GB VRAM)
  - `qwen2.5:7b` (4GB VRAM)
  - `llama3.2:3b` (2GB VRAM, 한국어 성능 낮음)

### 모델 다운로드 느림
- 첫 실행 시 18-20GB 모델 다운로드로 시간 소요 (양자화 버전)
- `/pull-model` 엔드포인트로 수동 다운로드
- 네트워크 상태에 따라 30분~1시간 소요 가능

### GPU 사용 확인
- Ollama는 자동으로 GPU 감지 및 사용
- `docker logs pathfinder-ollama`로 GPU 사용 확인
- NVIDIA Docker Runtime 설치 필요 (GPU 사용 시)

## 라이선스

MIT License