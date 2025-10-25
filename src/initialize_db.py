"""
Vector DB 초기화 스크립트
학과 정보를 임베딩하여 Qdrant에 저장합니다.
"""
import json
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def load_majors_data(data_path: str = "/app/data/majors.json"):
    """학과 데이터를 로드합니다."""
    if not os.path.exists(data_path):
        # Docker 환경이 아닌 경우 상대 경로 시도
        data_path = Path(__file__).parent.parent / "data" / "majors.json"

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_major_text(major: dict) -> str:
    """학과 정보를 텍스트로 변환합니다."""
    text_parts = [
        f"학과명: {major['name']}",
        f"분야: {major['category']}",
        f"설명: {major['description']}",
        f"관련 키워드: {', '.join(major['keywords'])}",
        f"진로: {', '.join(major['career_paths'])}",
        f"관련 교과: {', '.join(major['related_subjects'])}",
        f"필요 역량: {', '.join(major['skills_required'])}"
    ]
    return "\n".join(text_parts)


def initialize_qdrant():
    """Qdrant 벡터 DB를 초기화하고 학과 데이터를 임베딩하여 저장합니다."""
    # Qdrant 클라이언트 초기화
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    client = QdrantClient(url=qdrant_host)

    # 임베딩 모델 로드 (한국어 지원 모델)
    print("임베딩 모델 로드 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 컬렉션 이름
    collection_name = "majors"

    # 기존 컬렉션이 있다면 삭제
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"기존 컬렉션 '{collection_name}' 삭제됨")
    except Exception:
        pass

    # 새 컬렉션 생성
    print(f"컬렉션 '{collection_name}' 생성 중...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # 학과 데이터 로드
    print("학과 데이터 로드 중...")
    majors = load_majors_data()

    # 각 학과에 대해 임베딩 생성 및 저장
    print("학과 정보 임베딩 및 저장 중...")
    points = []

    for major in majors:
        # 학과 정보를 텍스트로 변환
        major_text = create_major_text(major)

        # 임베딩 생성
        embedding = model.encode(major_text).tolist()

        # 포인트 생성
        point = PointStruct(
            id=major['id'],
            vector=embedding,
            payload={
                "name": major['name'],
                "category": major['category'],
                "description": major['description'],
                "keywords": major['keywords'],
                "career_paths": major['career_paths'],
                "related_subjects": major['related_subjects'],
                "skills_required": major['skills_required'],
                "full_text": major_text
            }
        )
        points.append(point)

    # 배치로 저장
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"✅ {len(points)}개 학과 정보가 벡터 DB에 저장되었습니다!")

    # 저장된 데이터 확인
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"컬렉션 정보: {collection_info}")

    return client, model


if __name__ == "__main__":
    print("=== Qdrant Vector DB 초기화 시작 ===")
    try:
        initialize_qdrant()
        print("=== 초기화 완료 ===")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        raise
