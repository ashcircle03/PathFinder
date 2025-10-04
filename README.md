# PathFinder

고등학생 대학 학과 매칭 서비스 - RAG & LLMOps 기반

## 프로젝트 개요

PathFinder는 고등학생들의 관심사를 분석하여 적합한 대학 학과를 추천하는 AI 기반 진로 상담 서비스입니다.

### 기술 스택

- **LLM**: Ollama + Llama 3.2:3b
- **API**: FastAPI
- **컨테이너**: Docker, Docker Compose
- **향후 계획**: RAG, LLMOps, Kubernetes

## 시작하기

### 사전 요구사항

- Docker & Docker Compose 설치
- 최소 8GB RAM (LLM 실행용)

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

5. **서비스 상태 확인**
```bash
curl http://localhost:8000/health
```

### API 사용 예시

#### 학과 추천 요청
```bash
curl -X POST http://localhost:8000/recommend \
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
    "인공지능학과"
  ],
  "reasoning": "학생의 프로그래밍과 게임 개발 관심사는 컴퓨터공학 계열과 잘 맞습니다..."
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
│   └── main.py          # FastAPI 애플리케이션
├── models/              # LLM 모델 캐시 (자동 생성)
├── data/                # 데이터 저장소 (향후 RAG용)
├── Dockerfile           # API 서버 이미지
├── docker-compose.yml   # 서비스 오케스트레이션
├── requirements.txt     # Python 의존성
└── README.md
```

## 로드맵

### Phase 1: 기본 LLM 구동 ✅ (현재)
- [x] Docker 환경 구축
- [x] Ollama + Llama 3.2 연동
- [x] FastAPI 서버 구현
- [x] 기본 학과 추천 기능

### Phase 2: RAG 구현 (예정)
- [ ] Vector DB (Qdrant) 연동
- [ ] 학과 정보 임베딩
- [ ] 검색 기반 추천 개선

### Phase 3: LLMOps (예정)
- [ ] MLflow 실험 추적
- [ ] 프롬프트 버전 관리
- [ ] 모니터링 (Prometheus + Grafana)

### Phase 4: Kubernetes (예정)
- [ ] vLLM 전환
- [ ] K8s 매니페스트 작성
- [ ] HPA 오토스케일링

## 개발

### 로컬 개발 (Docker 없이)
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Ollama 로컬 설치 필요 (https://ollama.ai)
ollama pull llama3.2:3b

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

### 메모리 부족
- Docker Desktop 메모리 할당 증가 (최소 8GB)
- 더 작은 모델 사용: `llama3.2:1b`

### 모델 다운로드 느림
- 첫 실행 시 3-4GB 모델 다운로드로 시간 소요
- `/pull-model` 엔드포인트로 수동 다운로드

## 라이선스

MIT License