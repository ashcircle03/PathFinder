# PathFinder Kubernetes 배포 가이드 (Minikube)

이 가이드는 PathFinder를 Minikube 환경에서 Kubernetes로 배포하는 방법을 안내합니다.

## 📋 사전 요구사항

### 필수 도구
- **Minikube** (v1.30 이상)
- **kubectl** (클러스터 버전과 호환)
- **Docker** (Minikube driver로 사용)
- **NVIDIA GPU** (Ollama 실행용, 선택사항)

### 설치 확인
```bash
minikube version
kubectl version --client
docker --version
```

---

## 🚀 1. Minikube 클러스터 시작

### GPU 지원 없이 시작 (CPU만 사용)
```bash
minikube start --driver=docker --memory=12288 --cpus=4
```

### GPU 지원과 함께 시작 (권장)
```bash
# NVIDIA Container Toolkit이 설치되어 있어야 합니다
minikube start --driver=docker --gpus all --memory=12288 --cpus=4

# GPU Device Plugin 설치
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

**메모리 설정 참고:**
- 최소 권장: 12GB (12288MB)
- Docker Desktop 메모리가 부족한 경우: 8GB (8192MB)
- Docker Desktop 설정에서 메모리 할당을 늘릴 수 있습니다 (Settings → Resources → Memory)

### 클러스터 상태 확인
```bash
minikube status
kubectl cluster-info
```

---

## 🐳 2. Docker 이미지 빌드

Minikube 내부 Docker 데몬을 사용하여 이미지를 빌드합니다.

```bash
# Minikube Docker 환경 설정
eval $(minikube docker-env)

# 프로젝트 루트 디렉토리로 이동
cd /path/to/PathFinder

# API 이미지 빌드
docker build -t pathfinder-api:latest .

# Frontend 이미지 빌드
docker build -t pathfinder-frontend:latest ./frontend

# 이미지 확인
docker images | grep pathfinder
```

**중요**: `eval $(minikube docker-env)` 명령은 현재 터미널 세션에만 적용됩니다. 새 터미널에서는 다시 실행해야 합니다.

---

## 📦 3. Kubernetes 리소스 배포

### 방법 A: 배포 스크립트 사용 (권장)

```bash
cd k8s

# 스크립트 실행 권한 부여 (Linux/Mac)
chmod +x deploy.sh cleanup.sh

# 배포 실행
./deploy.sh
```

### 방법 B: 수동 배포

```bash
cd k8s

# 1. Namespace 생성
kubectl apply -f 00-namespace.yaml

# 2. ConfigMap 생성
kubectl apply -f 01-configmap.yaml

# 3. PVC 생성
kubectl apply -f 02-pvc-ollama.yaml
kubectl apply -f 03-pvc-qdrant.yaml

# 4. Ollama 배포
kubectl apply -f 04-deployment-ollama.yaml
kubectl apply -f 05-service-ollama.yaml

# Ollama 준비 대기
kubectl wait --for=condition=ready pod -l component=ollama -n pathfinder --timeout=300s

# 5. Qdrant 배포
kubectl apply -f 06-deployment-qdrant.yaml
kubectl apply -f 07-service-qdrant.yaml

# Qdrant 준비 대기
kubectl wait --for=condition=ready pod -l component=qdrant -n pathfinder --timeout=300s

# 6. API 배포
kubectl apply -f 08-deployment-api.yaml
kubectl apply -f 09-service-api.yaml

# API 준비 대기
kubectl wait --for=condition=ready pod -l component=api -n pathfinder --timeout=300s

# 7. Frontend 배포
kubectl apply -f 10-deployment-frontend.yaml
kubectl apply -f 11-service-frontend.yaml

# Frontend 준비 대기
kubectl wait --for=condition=ready pod -l component=frontend -n pathfinder --timeout=300s
```

---

## 🎯 4. 초기 설정

### 4.1 Ollama 모델 다운로드

```bash
# Ollama Pod 이름 확인
kubectl get pods -n pathfinder -l component=ollama

# Ollama Pod에 접속하여 모델 다운로드
kubectl exec -it deployment/ollama -n pathfinder -- ollama pull exaone3.5:7.8b

# 모델 확인
kubectl exec -it deployment/ollama -n pathfinder -- ollama list
```

### 4.2 Vector DB 초기화

```bash
# API Pod에서 초기화 스크립트 실행
kubectl exec -it deployment/api -n pathfinder -- python src/initialize_db.py
```

---

## 🌍 5. 애플리케이션 접속

### Frontend 접속

```bash
# 방법 1: Minikube service 명령 사용
minikube service frontend-service -n pathfinder

# 방법 2: URL만 확인
minikube service frontend-service -n pathfinder --url

# 방법 3: NodePort로 직접 접속
# Minikube IP 확인
minikube ip
# 브라우저에서 http://<minikube-ip>:30080 접속
```

### API 직접 접속 (포트 포워딩)

```bash
# API를 localhost:8000으로 포워딩
kubectl port-forward -n pathfinder svc/api-service 8000:8000

# 별도 터미널에서 테스트
curl http://localhost:8000/health
```

---

## 📊 6. 모니터링 및 관리

### Pod 상태 확인

```bash
# 모든 Pod 확인
kubectl get pods -n pathfinder

# 실시간 감시
kubectl get pods -n pathfinder -w

# 상세 정보
kubectl describe pod <pod-name> -n pathfinder
```

### 로그 확인

```bash
# 특정 Pod 로그
kubectl logs -f deployment/api -n pathfinder

# 모든 API Pod 로그
kubectl logs -f -l component=api -n pathfinder

# 이전 컨테이너 로그 (재시작된 경우)
kubectl logs --previous <pod-name> -n pathfinder
```

### 리소스 사용량 확인

```bash
# 모든 리소스 확인
kubectl get all -n pathfinder

# PVC 상태 확인
kubectl get pvc -n pathfinder

# 노드 리소스 확인
kubectl top nodes
kubectl top pods -n pathfinder
```

### 대시보드 사용

```bash
# Kubernetes 대시보드 실행
minikube dashboard
```

---

## 🔧 7. 문제 해결

### Ollama가 시작되지 않는 경우

```bash
# Ollama Pod 이벤트 확인
kubectl describe pod -l component=ollama -n pathfinder

# GPU 리소스 확인
kubectl describe node

# GPU 없이 실행하려면 deployment-ollama.yaml에서 GPU 요청 제거:
# resources.requests.nvidia.com/gpu: "1" 삭제
```

### PVC가 Pending 상태인 경우

```bash
# PVC 상태 확인
kubectl get pvc -n pathfinder
kubectl describe pvc ollama-pvc -n pathfinder

# StorageClass 확인
kubectl get storageclass

# Minikube 기본 StorageClass 사용 확인
kubectl get sc standard
```

### 이미지 Pull 오류

```bash
# Minikube Docker 환경 확인
eval $(minikube docker-env)
docker images | grep pathfinder

# 이미지가 없으면 다시 빌드
docker build -t pathfinder-api:latest .
docker build -t pathfinder-frontend:latest ./frontend
```

### Service에 접속할 수 없는 경우

```bash
# Service 상태 확인
kubectl get svc -n pathfinder

# Endpoints 확인
kubectl get endpoints -n pathfinder

# Minikube 터널 사용 (LoadBalancer 타입인 경우)
minikube tunnel
```

---

## 🧹 8. 리소스 정리

### 스크립트 사용

```bash
cd k8s
./cleanup.sh
```

### 수동 정리

```bash
# 모든 리소스 삭제 (Namespace 제외)
kubectl delete -f 11-service-frontend.yaml
kubectl delete -f 10-deployment-frontend.yaml
kubectl delete -f 09-service-api.yaml
kubectl delete -f 08-deployment-api.yaml
kubectl delete -f 07-service-qdrant.yaml
kubectl delete -f 06-deployment-qdrant.yaml
kubectl delete -f 05-service-ollama.yaml
kubectl delete -f 04-deployment-ollama.yaml
kubectl delete -f 03-pvc-qdrant.yaml
kubectl delete -f 02-pvc-ollama.yaml
kubectl delete -f 01-configmap.yaml

# Namespace 삭제 (모든 리소스 포함)
kubectl delete -f 00-namespace.yaml
```

### Minikube 클러스터 정리

```bash
# 클러스터 중지
minikube stop

# 클러스터 삭제 (모든 데이터 삭제)
minikube delete
```

---

## 📚 9. 아키텍처

### 리소스 구조

```
pathfinder (Namespace)
├── ConfigMap: pathfinder-config
├── PVC: ollama-pvc (20Gi)
├── PVC: qdrant-pvc (5Gi)
├── Deployment: ollama (1 replica)
│   └── Service: ollama-service (ClusterIP:11434)
├── Deployment: qdrant (1 replica)
│   └── Service: qdrant-service (ClusterIP:6333,6334)
├── Deployment: api (2 replicas)
│   └── Service: api-service (ClusterIP:8000)
└── Deployment: frontend (2 replicas)
    └── Service: frontend-service (NodePort:30080)
```

### 네트워크 통신

```
외부 사용자
    ↓
frontend-service (NodePort:30080)
    ↓
frontend Pod (Nginx)
    ↓ /api/* → proxy
api-service (ClusterIP:8000)
    ↓
api Pod (FastAPI)
    ↓
    ├─→ ollama-service:11434 (LLM)
    └─→ qdrant-service:6333 (Vector DB)
```

---

## 🔄 10. 업데이트 및 롤백

### 이미지 업데이트

```bash
# 1. Minikube Docker 환경 설정
eval $(minikube docker-env)

# 2. 새 이미지 빌드
docker build -t pathfinder-api:v2 .

# 3. Deployment 이미지 업데이트
kubectl set image deployment/api api=pathfinder-api:v2 -n pathfinder

# 4. 롤아웃 상태 확인
kubectl rollout status deployment/api -n pathfinder
```

### 롤백

```bash
# 이전 버전으로 롤백
kubectl rollout undo deployment/api -n pathfinder

# 특정 리비전으로 롤백
kubectl rollout history deployment/api -n pathfinder
kubectl rollout undo deployment/api --to-revision=2 -n pathfinder
```

---

## 📝 11. 추가 팁

### 빠른 Pod 재시작

```bash
# Pod 재시작 (롤링 업데이트)
kubectl rollout restart deployment/api -n pathfinder
```

### ConfigMap 업데이트

```bash
# ConfigMap 수정 후 적용
kubectl apply -f 01-configmap.yaml

# Pod 재시작하여 새 설정 반영
kubectl rollout restart deployment/api -n pathfinder
```

### 리소스 스케일링

```bash
# API Pod 개수 조정
kubectl scale deployment/api --replicas=3 -n pathfinder

# 확인
kubectl get deployment api -n pathfinder
```

---

## 🆘 도움말

문제가 발생하면 다음을 확인하세요:

1. **Pod 로그**: `kubectl logs -f <pod-name> -n pathfinder`
2. **Pod 이벤트**: `kubectl describe pod <pod-name> -n pathfinder`
3. **리소스 상태**: `kubectl get all -n pathfinder`
4. **Minikube 상태**: `minikube status`

더 많은 정보는 [Kubernetes 공식 문서](https://kubernetes.io/docs/home/)를 참고하세요.
