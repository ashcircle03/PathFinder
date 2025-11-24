"""
LangChain 기반 RAG (Retrieval-Augmented Generation) 학과 추천 시스템
"""
import os
import json
from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
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

        # 6. Advanced Retriever 설정 (BM25 + Vector + Multi-Query + Compression)
        print("🔍 Advanced Retriever 설정 중...")
        self._setup_advanced_retriever()

        print("✅ LangChain RAG 시스템 초기화 완료!")

    def _setup_advanced_retriever(self):
        """고급 검색 시스템 설정: BM25 + Vector Ensemble + Multi-Query + Compression"""

        # 모든 문서 로드 (BM25용)
        print("  📚 문서 로드 중 (BM25용)...")
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "majors.json")
        with open(data_path, "r", encoding="utf-8") as f:
            majors_data = json.load(f)

        # Document 객체로 변환
        documents = []
        for major in majors_data:
            # 키워드와 진로도 텍스트에 포함하여 BM25 성능 향상
            text_content = f"{major['name']} {major['description']} "
            text_content += " ".join(major.get('keywords', []))
            text_content += " " + " ".join(major.get('career_paths', []))

            doc = Document(
                page_content=text_content,
                metadata={
                    "name": major['name'],
                    "category": major.get('category', ''),
                    "keywords": major.get('keywords', []),
                    "career_paths": major.get('career_paths', []),
                    "related_subjects": major.get('related_subjects', []),
                    "skills_required": major.get('skills_required', [])
                }
            )
            documents.append(doc)

        # 1. Vector Retriever (Qdrant)
        print("  🔢 Vector Retriever 설정...")
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # 더 많이 검색 (ensemble에서 필터링됨)
        )

        # 2. BM25 Retriever (Keyword-based)
        print("  🔤 BM25 Retriever 설정...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10

        # 3. Ensemble Retriever (Vector 60% + BM25 40%)
        print("  🎯 Ensemble Retriever 구성...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]  # 의미 검색 60%, 키워드 검색 40%
        )

        # 4. Multi-Query Retriever (쿼리 확장)
        print("  🔄 Multi-Query Retriever 설정...")

        # Multi-Query용 프롬프트
        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""당신은 대학 학과 검색을 돕는 AI 어시스턴트입니다.
학생의 관심사를 바탕으로 다양한 각도에서 3가지 검색 쿼리를 생성하세요.

원본 질문: {question}

다음 관점에서 쿼리를 작성하세요:
1. 직접적인 키워드 중심 쿼리
2. 관련 진로/직업 중심 쿼리
3. 필요한 역량/과목 중심 쿼리

검색 쿼리 (한 줄에 하나씩, 번호 없이):"""
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=self.llm,
            prompt=multi_query_prompt
        )

        # 5. Contextual Compression (노이즈 제거)
        print("  🗜️ Compression Retriever 설정...")
        compressor = LLMChainExtractor.from_llm(self.llm)

        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever
        )

        print("  ✅ Advanced Retriever 구성 완료!")
        print("     - Vector Search (Qdrant)")
        print("     - BM25 Keyword Search")
        print("     - Ensemble (60% Vector + 40% BM25)")
        print("     - Multi-Query Expansion")
        print("     - Contextual Compression")

    def _setup_prompts(self):
        """프롬프트 템플릿 설정 (Few-Shot 예제 포함)"""

        # Few-Shot 예제들 - 올바른 JSON 형식을 보여주는 예제
        examples = [
            {
                "context": """1. 컴퓨터공학과 (공학)
   설명: 소프트웨어와 하드웨어의 이론과 응용을 연구하는 학과입니다.
   키워드: 프로그래밍, 알고리즘, 소프트웨어 개발, 데이터구조, 네트워크
   진로: 소프트웨어 엔지니어, 시스템 엔지니어, 데이터 과학자

2. 인공지능학과 (공학)
   설명: 인공지능, 머신러닝, 딥러닝 기술을 연구하고 개발하는 학과입니다.
   키워드: AI, 머신러닝, 딥러닝, 자연어처리, 컴퓨터비전
   진로: AI 연구원, 머신러닝 엔지니어, 데이터 사이언티스트""",
                "question": "프로그래밍과 AI에 관심이 많아요. 수학도 좋아합니다.",
                "output": """{
  "recommended_majors": ["인공지능학과", "컴퓨터공학과", "데이터사이언스학과"],
  "reasoning": "학생의 프로그래밍과 AI 관심사를 고려했을 때 인공지능학과가 가장 적합합니다. 머신러닝과 딥러닝을 심도있게 배울 수 있으며, 수학 실력을 활용할 수 있습니다. 컴퓨터공학과는 프로그래밍 기초를 탄탄히 다질 수 있으며, 데이터사이언스학과는 수학과 프로그래밍을 결합하여 데이터 분석 역량을 기를 수 있습니다."
}"""
            },
            {
                "context": """1. 심리학과 (인문사회)
   설명: 인간의 마음과 행동을 과학적으로 연구하는 학과입니다.
   키워드: 상담, 심리치료, 인간행동, 심리검사, 발달심리
   진로: 상담심리사, 임상심리사, 조직심리 전문가

2. 사회복지학과 (인문사회)
   설명: 사회적 약자를 돕고 복지 정책을 연구하는 학과입니다.
   키워드: 복지정책, 사회서비스, 지역사회, 노인복지, 아동복지
   진로: 사회복지사, 복지정책 전문가, NGO 활동가""",
                "question": "사람을 돕는 일에 관심이 있고, 상담에 흥미가 있습니다.",
                "output": """{
  "recommended_majors": ["심리학과", "사회복지학과", "상담학과"],
  "reasoning": "사람을 돕고 상담에 관심이 있다면 심리학과에서 인간의 마음을 과학적으로 탐구하며 상담 전문가로 성장할 수 있습니다. 사회복지학과는 실질적인 도움을 제공하는 방법을 배우며 현장에서 바로 적용할 수 있는 실무 능력을 기를 수 있습니다. 상담학과는 전문 상담사로서의 역량을 집중적으로 개발할 수 있습니다."
}"""
            },
            {
                "context": """1. 경영학과 (상경)
   설명: 기업 경영과 관리에 필요한 이론과 실무를 배우는 학과입니다.
   키워드: 경영전략, 마케팅, 재무관리, 인사관리, 회계
   진로: 경영 컨설턴트, 마케팅 전문가, 재무 분석가

2. 경제학과 (상경)
   설명: 경제 현상과 시장 원리를 분석하고 연구하는 학과입니다.
   키워드: 거시경제, 미시경제, 금융, 시장분석, 경제정책
   진로: 이코노미스트, 금융 애널리스트, 경제 연구원""",
                "question": "리더십이 있고, 창업에 관심이 많습니다. 수익 모델 개발이 흥미로워요.",
                "output": """{
  "recommended_majors": ["경영학과", "경제학과", "창업학과"],
  "reasoning": "리더십과 창업에 관심이 있다면 경영학과에서 기업 운영과 전략을 체계적으로 배울 수 있습니다. 경영학은 마케팅, 재무, 인사 등 창업에 필요한 모든 기능을 다루며 실무 중심 교육을 받을 수 있습니다. 경제학과는 시장 원리와 수익 모델을 분석하는 능력을 기를 수 있으며, 창업학과는 실제 창업 과정을 직접 경험하며 배울 수 있습니다."
}"""
            }
        ]

        # 예제 템플릿
        example_template = """[검색된 학과 정보]
{context}

[학생 관심사]
{question}

답변:
{output}"""

        example_prompt = PromptTemplate(
            input_variables=["context", "question", "output"],
            template=example_template
        )

        # Few-Shot 프롬프트 구성
        prefix = """당신은 한국의 대학 진학 상담 전문가입니다.
제공된 학과 정보를 바탕으로, 학생에게 가장 적합한 학과를 추천해주세요.

다음은 올바른 추천 형식의 예제들입니다:

"""

        suffix = """이제 다음 학생의 경우에 대해 위와 동일한 형식으로 추천해주세요:

[검색된 학과 정보]
{context}

[학생 관심사]
{question}

중요 규칙:
- 반드시 검색된 학과 정보 내에서만 추천하세요
- 존재하지 않는 학과를 만들어내지 마세요
- 순수 한글로만 답변하세요 (한자 사용 금지)
- 정확히 JSON 형식으로 답변하세요

{format_instructions}

답변:"""

        self.prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
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
