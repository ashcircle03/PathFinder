#!/bin/bash

# PathFinder Kubernetes 리소스 정리 스크립트
set -e

echo "🧹 PathFinder Kubernetes 리소스 정리 중..."

# 모든 리소스 삭제 (역순)
echo "🗑️  Frontend 삭제 중..."
kubectl delete -f 11-service-frontend.yaml --ignore-not-found=true
kubectl delete -f 10-deployment-frontend.yaml --ignore-not-found=true

echo "🗑️  API 서버 삭제 중..."
kubectl delete -f 09-service-api.yaml --ignore-not-found=true
kubectl delete -f 08-deployment-api.yaml --ignore-not-found=true

echo "🗑️  Qdrant 삭제 중..."
kubectl delete -f 07-service-qdrant.yaml --ignore-not-found=true
kubectl delete -f 06-deployment-qdrant.yaml --ignore-not-found=true

echo "🗑️  Ollama 삭제 중..."
kubectl delete -f 05-service-ollama.yaml --ignore-not-found=true
kubectl delete -f 04-deployment-ollama.yaml --ignore-not-found=true

echo "🗑️  PVC 삭제 중..."
kubectl delete -f 03-pvc-qdrant.yaml --ignore-not-found=true
kubectl delete -f 02-pvc-ollama.yaml --ignore-not-found=true

echo "🗑️  ConfigMap 삭제 중..."
kubectl delete -f 01-configmap.yaml --ignore-not-found=true

# Namespace는 선택적으로 삭제
read -p "⚠️  Namespace도 삭제하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🗑️  Namespace 삭제 중..."
    kubectl delete -f 00-namespace.yaml --ignore-not-found=true
    echo "✅ Namespace를 포함한 모든 리소스가 삭제되었습니다."
else
    echo "✅ Namespace를 제외한 모든 리소스가 삭제되었습니다."
    echo "💡 남은 리소스 확인: kubectl get all -n pathfinder"
fi

echo ""
echo "🧹 정리 완료!"
