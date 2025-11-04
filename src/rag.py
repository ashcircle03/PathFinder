"""
LangChain 기반 RAG (Retrieval-Augmented Generation) 학과 추천 시스템
"""
import os
from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class MajorRecommendation(BaseModel):
    """학과 추천 결과 모델"""
    recommended_majors: List[str] = Field(description="추천 학과 목록 (3-5개)")
    reasoning: str = Field(description="추천 이유 및 각 학과에 대한 설명")


class MajorRecommendationRAG:
    """LangChain 기반 학과 추천 RAG 시스템"""

    def __init__(self):
        # 환경 변수
        qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm_model = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")
        self.collection_name = "majors"

        print("🚀 LangChain RAG 시스템 초기화 중...")

        # 1. 임베딩 모델 초기화 (한국어 특화)
        print("📦 임베딩 모델 로딩...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. Qdrant Vector Store 초기화
        print("🗄️ Qdrant Vector Store 연결...")
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=qdrant_host,
        )

        # 3. LLM 초기화 (Ollama)
        print(f"🤖 LLM 초기화: {self.llm_model}")
        self.llm = OllamaLLM(
            model=self.llm_model,
            base_url=ollama_host,
            temperature=0.7,
            # 한국어 응답 개선을 위한 설정
            system="You are a Korean university counselor. Always respond in pure Korean (Hangul only). Never use Chinese characters (Hanja), Japanese, or English except for proper nouns.",
        )

        # 4. Output Parser 설정
        self.output_parser = PydanticOutputParser(pydantic_object=MajorRecommendation)

        # 5. 프롬프트 템플릿 설정
        self._setup_prompts()

        # 6. Retriever 설정
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 상위 5개 학과 검색
        )

        print("✅ LangChain RAG 시스템 초기화 완료!")

    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""

        # RAG 프롬프트 템플릿
        self.rag_template = """당신은 한국의 대학 진학 상담 전문가입니다.
제공된 학과 정보를 바탕으로, 학생에게 가장 적합한 학과를 추천해주세요.

[검색된 학과 정보]
{context}

[학생 관심사]
{question}

위 검색된 학과 정보를 바탕으로:
1. 학생의 관심사와 가장 잘 맞는 학과 3-5개를 추천해주세요
2. 각 학과를 추천하는 구체적인 이유를 설명해주세요
3. 각 학과의 특징과 진로 전망을 간략히 소개해주세요

중요 규칙:
- 반드시 검색된 학과 정보 내에서만 추천하세요
- 존재하지 않는 학과를 만들어내지 마세요
- 순수 한글로만 답변하세요 (한자 사용 금지)
- 중국어나 일본어를 사용하지 마세요
- 모든 전문 용어도 한글로 표기하세요

{format_instructions}

답변:"""

        self.prompt = PromptTemplate(
            template=self.rag_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )

    def search_similar_majors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        사용자의 관심사와 유사한 학과를 검색합니다.

        Args:
            query: 사용자의 관심사 문자열
            top_k: 반환할 상위 결과 개수

        Returns:
            유사한 학과 정보 리스트
        """
        # LangChain retriever를 통한 검색
        docs = self.retriever.get_relevant_documents(query)[:top_k]

        results = []
        for doc in docs:
            results.append({
                "score": doc.metadata.get("_score", 0.0),
                "major_name": doc.metadata.get("name", ""),
                "category": doc.metadata.get("category", ""),
                "description": doc.page_content,
                "keywords": doc.metadata.get("keywords", []),
                "career_paths": doc.metadata.get("career_paths", []),
                "related_subjects": doc.metadata.get("related_subjects", []),
                "skills_required": doc.metadata.get("skills_required", [])
            })

        return results

    def generate_recommendation(
        self,
        interests: str,
        search_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        검색된 학과 정보를 바탕으로 LLM을 사용하여 추천을 생성합니다.

        Args:
            interests: 사용자의 관심사
            search_results: 검색된 학과 정보 리스트 (None이면 자동 검색)

        Returns:
            추천 학과와 이유를 포함한 딕셔너리
        """
        # 검색 결과가 없으면 자동 검색
        if search_results is None:
            search_results = self.search_similar_majors(interests)

        # 컨텍스트 구성
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            context_parts.append(
                f"{idx}. {result['major_name']} ({result['category']})\n"
                f"   설명: {result['description']}\n"
                f"   키워드: {', '.join(result['keywords'][:5])}\n"
                f"   진로: {', '.join(result['career_paths'][:3])}"
            )

        context = "\n\n".join(context_parts)

        try:
            # LLM을 통한 추천 생성
            prompt_text = self.prompt.format(context=context, question=interests)
            response = self.llm.invoke(prompt_text)

            # Output Parser를 통한 파싱 시도
            try:
                parsed_response = self.output_parser.parse(response)
                return {
                    "recommended_majors": parsed_response.recommended_majors,
                    "reasoning": parsed_response.reasoning,
                    "retrieved_context": search_results
                }
            except Exception as parse_error:
                # 파싱 실패 시 fallback: 검색된 학과명 사용
                print(f"⚠️ Output 파싱 실패, fallback 사용: {parse_error}")
                return {
                    "recommended_majors": [r['major_name'] for r in search_results[:5]],
                    "reasoning": response,
                    "retrieved_context": search_results
                }

        except Exception as e:
            # LLM 호출 실패 시 검색 결과만 반환
            print(f"❌ LLM 생성 실패: {e}")
            return {
                "recommended_majors": [r['major_name'] for r in search_results[:5]],
                "reasoning": f"검색된 학과를 기반으로 추천합니다. (LLM 오류: {str(e)})",
                "retrieved_context": search_results
            }

    def recommend_majors(self, interests: str, top_k: int = 5) -> Dict[str, Any]:
        """
        사용자의 관심사를 기반으로 학과를 추천합니다. (통합 메서드)

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
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(url=self.vectorstore._client._host)
            collection_info = qdrant_client.get_collection(self.collection_name)

            # 임베딩 모델 테스트
            test_embedding = self.embeddings.embed_query("테스트")

            # LLM 테스트
            test_response = self.llm.invoke("안녕하세요")

            return {
                "status": "healthy",
                "vectorstore": "connected",
                "collection_name": self.collection_name,
                "vectors_count": collection_info.points_count,
                "embedding_model": "jhgan/ko-sroberta-multitask",
                "embedding_dim": len(test_embedding),
                "llm_model": self.llm_model,
                "llm_status": "ok" if test_response else "error"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 전역 RAG 시스템 인스턴스 (싱글톤)
_rag_system = None


def get_rag_system() -> MajorRecommendationRAG:
    """싱글톤 RAG 시스템 가져오기"""
    global _rag_system
    if _rag_system is None:
        _rag_system = MajorRecommendationRAG()
    return _rag_system
