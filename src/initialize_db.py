"""
LangChain 기반 Qdrant Vector DB 초기화 스크립트
"""
import json
import os
from typing import List
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def load_majors_data() -> List[dict]:
    """학과 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), "../data/majors.json")

    with open(data_path, "r", encoding="utf-8") as f:
        majors = json.load(f)

    print(f"✅ 학과 데이터 로드 완료: {len(majors)}개")
    return majors


def create_documents(majors: List[dict]) -> List[Document]:
    """학과 데이터를 LangChain Document로 변환"""
    documents = []

    for major in majors:
        # page_content: 검색 대상이 되는 주요 텍스트
        content = f"{major['name']} - {major['description']}"

        # metadata: 추가 정보 (필터링, 반환 시 사용)
        metadata = {
            "id": major["id"],
            "name": major["name"],
            "category": major["category"],
            "keywords": major["keywords"],
            "career_paths": major["career_paths"],
            "related_subjects": major["related_subjects"],
            "skills_required": major["skills_required"]
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"✅ Document 변환 완료: {len(documents)}개")
    return documents


def initialize_qdrant():
    """Qdrant Vector Store 초기화"""
    # 환경 변수
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    collection_name = "majors"

    print("🚀 Qdrant Vector DB 초기화 시작...")

    # 1. 학과 데이터 로드
    majors = load_majors_data()

    # 2. LangChain Document로 변환
    documents = create_documents(majors)

    # 3. 임베딩 모델 초기화 (한국어 특화)
    print("📦 임베딩 모델 로딩 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Qdrant 클라이언트 초기화
    print(f"🔗 Qdrant 연결: {qdrant_host}")
    qdrant_client = QdrantClient(url=qdrant_host)

    # 5. 기존 컬렉션 삭제 (있다면)
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"🗑️  기존 컬렉션 '{collection_name}' 삭제")
    except Exception:
        print(f"ℹ️  기존 컬렉션 없음")

    # 6. 새 컬렉션 생성 및 데이터 저장
    print("💾 벡터 DB에 데이터 저장 중...")

    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_host,
        collection_name=collection_name,
        force_recreate=True  # 컬렉션 강제 재생성
    )

    # 7. 저장 확인
    collection_info = qdrant_client.get_collection(collection_name)

    # vectors가 dict인 경우와 객체인 경우 모두 처리
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        # 단일 벡터 설정 (dict)
        vector_size = list(vectors_config.values())[0].size
        vector_distance = list(vectors_config.values())[0].distance
    else:
        # VectorParams 객체
        vector_size = vectors_config.size
        vector_distance = vectors_config.distance

    print("=" * 60)
    print("✅ Qdrant Vector DB 초기화 완료!")
    print(f"   - 컬렉션: {collection_name}")
    print(f"   - 벡터 수: {collection_info.points_count}")
    print(f"   - 벡터 차원: {vector_size}")
    print(f"   - 거리 메트릭: {vector_distance}")
    print("=" * 60)

    return {
        "collection_name": collection_name,
        "vectors_count": collection_info.points_count,
        "vector_dim": vector_size,
        "embedding_model": "jhgan/ko-sroberta-multitask"
    }


if __name__ == "__main__":
    # 직접 실행 시 초기화
    initialize_qdrant()
