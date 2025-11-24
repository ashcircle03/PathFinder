#!/bin/bash

# PathFinder Kubernetes 배포 스크립트 (Minikube)
set -e

echo "🚀 PathFinder Kubernetes 배포 시작..."

# 1. Namespace 생성
echo "📦 Namespace 생성 중..."
kubectl apply -f 00-namespace.yaml

# 2. ConfigMap 생성
echo "⚙️  ConfigMap 생성 중..."
kubectl apply -f 01-configmap.yaml

# 3. PVC 생성
echo "💾 Persistent Volume Claims 생성 중..."
kubectl apply -f 02-pvc-ollama.yaml
kubectl apply -f 03-pvc-qdrant.yaml

# 4. Ollama 배포
echo "🤖 Ollama 배포 중..."
kubectl apply -f 04-deployment-ollama.yaml
kubectl apply -f 05-service-ollama.yaml

# Ollama가 준비될 때까지 대기
echo "⏳ Ollama 준비 대기 중..."
kubectl wait --for=condition=ready pod -l component=ollama -n pathfinder --timeout=300s

# 5. Qdrant 배포
echo "🗄️  Qdrant 배포 중..."
kubectl apply -f 06-deployment-qdrant.yaml
kubectl apply -f 07-service-qdrant.yaml

# Qdrant가 준비될 때까지 대기
echo "⏳ Qdrant 준비 대기 중..."
kubectl wait --for=condition=ready pod -l component=qdrant -n pathfinder --timeout=300s

# 6. API 배포
echo "🔧 API 서버 배포 중..."
kubectl apply -f 08-deployment-api.yaml
kubectl apply -f 09-service-api.yaml

# API가 준비될 때까지 대기
echo "⏳ API 서버 준비 대기 중..."
kubectl wait --for=condition=ready pod -l component=api -n pathfinder --timeout=300s

# 7. Frontend 배포
echo "🌐 Frontend 배포 중..."
kubectl apply -f 10-deployment-frontend.yaml
kubectl apply -f 11-service-frontend.yaml

# Frontend가 준비될 때까지 대기
echo "⏳ Frontend 준비 대기 중..."
kubectl wait --for=condition=ready pod -l component=frontend -n pathfinder --timeout=300s

echo ""
echo "✅ 배포 완료!"
echo ""
echo "📊 리소스 상태 확인:"
kubectl get all -n pathfinder

echo ""
echo "🌍 Frontend 접속 URL:"
minikube service frontend-service -n pathfinder --url

echo ""
echo "💡 유용한 명령어:"
echo "  - Pod 로그 확인: kubectl logs -f <pod-name> -n pathfinder"
echo "  - Pod 상태 확인: kubectl get pods -n pathfinder"
echo "  - 서비스 확인: kubectl get svc -n pathfinder"
echo "  - Frontend 열기: minikube service frontend-service -n pathfinder"
echo ""
echo "🎯 다음 단계:"
echo "  1. Ollama 모델 다운로드:"
echo "     kubectl exec -it deployment/ollama -n pathfinder -- ollama pull exaone3.5:7.8b"
echo "  2. Vector DB 초기화:"
echo "     kubectl exec -it deployment/api -n pathfinder -- python src/initialize_db.py"
echo ""
