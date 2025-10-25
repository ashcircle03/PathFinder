"""
RAG (Retrieval-Augmented Generation) 기반 학과 추천 시스템
"""
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama


class MajorRecommendationRAG:
    """RAG 기반 학과 추천 시스템"""

    def __init__(self):
        # Qdrant 클라이언트 초기화
        qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        self.qdrant_client = QdrantClient(url=qdrant_host)
        self.collection_name = "majors"

        # 임베딩 모델 로드 (한국어 지원)
        print("임베딩 모델 로드 중...")
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        # Ollama 클라이언트 초기화
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.llm_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        print("RAG 시스템 초기화 완료!")

    def search_similar_majors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        사용자의 관심사와 유사한 학과를 검색합니다.

        Args:
            query: 사용자의 관심사 문자열
            top_k: 반환할 상위 결과 개수

        Returns:
            유사한 학과 정보 리스트
        """
        # 쿼리를 임베딩으로 변환
        query_embedding = self.embedding_model.encode(query).tolist()

        # Qdrant에서 유사한 학과 검색
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # 결과 포맷팅
        results = []
        for hit in search_results:
            results.append({
                "score": hit.score,
                "major_name": hit.payload["name"],
                "category": hit.payload["category"],
                "description": hit.payload["description"],
                "keywords": hit.payload["keywords"],
                "career_paths": hit.payload["career_paths"],
                "related_subjects": hit.payload["related_subjects"],
                "skills_required": hit.payload["skills_required"]
            })

        return results

    def generate_recommendation(self, interests: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        검색된 학과 정보를 바탕으로 LLM을 사용하여 추천을 생성합니다.

        Args:
            interests: 사용자의 관심사
            search_results: 검색된 학과 정보 리스트

        Returns:
            추천 학과와 이유를 포함한 딕셔너리
        """
        # 검색된 학과 정보를 컨텍스트로 구성
        context = "다음은 학생의 관심사와 관련된 학과 정보입니다:\n\n"

        for idx, result in enumerate(search_results, 1):
            context += f"{idx}. {result['major_name']} (관련도: {result['score']:.2f})\n"
            context += f"   분야: {result['category']}\n"
            context += f"   설명: {result['description']}\n"
            context += f"   관련 키워드: {', '.join(result['keywords'])}\n"
            context += f"   진로: {', '.join(result['career_paths'][:3])}\n\n"

        # LLM 프롬프트 생성
        prompt = f"""{context}

학생의 관심사: {interests}

위의 학과 정보를 참고하여 학생에게 가장 적합한 학과 3-5개를 추천하고, 각 학과가 학생의 관심사와 어떻게 연결되는지 구체적으로 설명해주세요.

다음 형식으로 답변해주세요:
1. 추천 학과: [학과명1, 학과명2, 학과명3, ...]
2. 추천 이유: [각 학과를 추천하는 구체적인 이유]

추천 학과:"""

        try:
            # Ollama를 통해 LLM 호출
            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': '당신은 진로 상담 전문가입니다. 학생의 관심사를 분석하여 적합한 대학 학과를 추천해주세요.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )

            llm_response = response['message']['content']

            # 추천 학과 추출
            recommended_majors = [result['major_name'] for result in search_results[:5]]

            return {
                "recommended_majors": recommended_majors,
                "reasoning": llm_response,
                "retrieved_context": search_results
            }

        except Exception as e:
            # LLM 호출 실패 시 검색 결과만 반환
            return {
                "recommended_majors": [result['major_name'] for result in search_results[:5]],
                "reasoning": f"검색된 학과를 기반으로 추천합니다. (LLM 오류: {str(e)})",
                "retrieved_context": search_results
            }

    def recommend_majors(self, interests: str, top_k: int = 5) -> Dict[str, Any]:
        """
        사용자의 관심사를 기반으로 학과를 추천합니다.

        Args:
            interests: 사용자의 관심사
            top_k: 검색할 학과 개수

        Returns:
            추천 결과 딕셔너리
        """
        # 1. 벡터 검색으로 유사한 학과 찾기
        search_results = self.search_similar_majors(interests, top_k=top_k)

        # 2. LLM을 사용하여 추천 생성
        recommendation = self.generate_recommendation(interests, search_results)

        return recommendation

    def health_check(self) -> Dict[str, Any]:
        """RAG 시스템의 상태를 확인합니다."""
        try:
            # Qdrant 연결 확인
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            # 임베딩 모델 확인
            test_embedding = self.embedding_model.encode("테스트")

            return {
                "status": "healthy",
                "qdrant_status": "connected",
                "collection_name": self.collection_name,
                "vectors_count": collection_info.points_count,
                "embedding_model": "jhgan/ko-sroberta-multitask",
                "embedding_dim": len(test_embedding)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
