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

    print(f"[OK] 학과 데이터 로드 완료: {len(majors)}개")
    return majors


def load_university_data() -> List[dict]:
    """대학 학과 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), "../data/university_departments.json")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            departments = json.load(f)
        print(f"[OK] 대학 학과 데이터 로드 완료: {len(departments)}개")
        return departments
    except FileNotFoundError:
        print(f"[WARN] 대학 학과 데이터 파일 없음: {data_path}")
        return []


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
            "skills_required": major["skills_required"],
            "source": "basic_majors"
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"[OK] 기본 학과 Document 변환 완료: {len(documents)}개")
    return documents


def create_university_documents(departments: List[dict]) -> List[Document]:
    """대학 학과 데이터를 LangChain Document로 변환"""
    documents = []

    for dept in departments:
        # page_content: 검색 대상이 되는 주요 텍스트 (더 풍부한 정보)
        content_parts = [
            f"{dept['university']} {dept['name']}",
            dept['description'],
            f"키워드: {', '.join(dept['keywords'])}",
        ]

        # curriculum 정보가 있으면 추가
        if dept.get('curriculum'):
            content_parts.append(f"커리큘럼: {' / '.join(dept['curriculum'][:2])}")

        # 진로 정보 추가
        if dept.get('career_prospects'):
            content_parts.append(f"진로: {', '.join(dept['career_prospects'])}")

        content = " | ".join(content_parts)

        # metadata: 상세 정보
        metadata = {
            "name": dept["name"],
            "university": dept["university"],
            "category": dept["category"],
            "keywords": dept["keywords"],
            "career_prospects": dept.get("career_prospects", []),
            "curriculum": dept.get("curriculum", []),
            "admission_quota": dept.get("admission_info", {}).get("quota", ""),
            "website": dept.get("admission_info", {}).get("website", ""),
            "source": "university_data"
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"[OK] 대학 학과 Document 변환 완료: {len(documents)}개")
    return documents


def initialize_qdrant():
    """Qdrant Vector Store 초기화"""
    # 환경 변수
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    collection_name = "majors"

    print("[START] Qdrant Vector DB 초기화 시작...")

    # 1. 기본 학과 데이터 로드
    majors = load_majors_data()

    # 2. 대학 학과 데이터 로드
    university_departments = load_university_data()

    # 3. LangChain Document로 변환
    documents = create_documents(majors)

    # 4. 대학 학과 Document 추가
    if university_departments:
        university_docs = create_university_documents(university_departments)
        documents.extend(university_docs)
        print(f"[OK] 총 Document 수: {len(documents)}개 (기본 {len(majors)}개 + 대학 {len(university_docs)}개)")

    # 5. 임베딩 모델 초기화 (한국어 특화)
    print("[LOADING] 임베딩 모델 로딩 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 6. Qdrant 클라이언트 초기화
    print(f"[CONNECT] Qdrant 연결: {qdrant_host}")
    qdrant_client = QdrantClient(url=qdrant_host)

    # 7. 기존 컬렉션 삭제 (있다면)
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"[DELETE] 기존 컬렉션 '{collection_name}' 삭제")
    except Exception:
        print(f"[INFO] 기존 컬렉션 없음")

    # 8. 새 컬렉션 생성 및 데이터 저장
    print("[SAVE] 벡터 DB에 데이터 저장 중...")

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
    print("[DONE] Qdrant Vector DB 초기화 완료!")
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
