"""
LangChain 기반 RAG (Retrieval-Augmented Generation) 학과 추천 시스템
"""
import os
import json
from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import EnsembleRetriever
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

        # Ensemble Retriever를 최종 retriever로 사용
        # (Multi-Query, Compression은 LLM 의존성이 높아 불안정하므로 제거)
        self.retriever = ensemble_retriever

        print("  ✅ Retriever 구성 완료!")
        print("     - Vector Search (Qdrant)")
        print("     - BM25 Keyword Search")
        print("     - Ensemble (60% Vector + 40% BM25)")

    def _setup_prompts(self):
        """프롬프트 템플릿 설정 (단순화된 버전)"""

        # 단순한 프롬프트 템플릿 사용
        template = """당신은 한국의 대학 진학 상담 전문가입니다.
학생의 관심사를 분석하여 가장 적합한 학과를 추천해주세요.

[검색된 학과 정보]
{context}

[학생 정보]
{question}

위 학과 정보를 바탕으로 학생에게 3-5개의 학과를 추천하고, 각 학과를 왜 추천하는지 상세히 설명해주세요.

반드시 아래 JSON 형식으로만 답변하세요. 다른 텍스트는 포함하지 마세요:
{{{{
  "recommended_majors": ["학과1", "학과2", "학과3"],
  "reasoning": "추천 이유를 여기에 작성하세요. 각 학과가 학생에게 적합한 이유를 설명합니다."
}}}}

답변:"""

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def search_similar_majors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        사용자의 관심사와 유사한 학과를 검색합니다.
        중복 학과를 제거하고 다양한 학과를 반환합니다.

        Args:
            query: 사용자의 관심사 문자열
            top_k: 반환할 상위 결과 개수

        Returns:
            유사한 학과 정보 리스트 (중복 제거됨)
        """
        # LangChain retriever를 통한 검색 (invoke 사용)
        print(f"🔍 검색 쿼리: {query}")
        # 더 많이 검색해서 중복 제거 후 top_k 반환
        docs = self.retriever.invoke(query)[:top_k * 3]
        print(f"📄 검색 결과: {len(docs)}개 문서 (중복 제거 전)")

        results = []
        seen_majors = set()  # 중복 체크용
        
        for doc in docs:
            major_name = doc.metadata.get("name", "")
            university = doc.metadata.get("university", "")  # 대학명
            
            # 대학명이 있으면 "대학 + 학과" 형식으로, 없으면 학과명만
            if university:
                display_name = f"{university} {major_name}"
            else:
                display_name = major_name
            
            # 같은 학과명은 한 번만 포함 (다양성 확보)
            if major_name in seen_majors:
                continue
            seen_majors.add(major_name)
            
            results.append({
                "score": doc.metadata.get("_score", 0.0),
                "major_name": major_name,
                "university": university,
                "display_name": display_name,
                "category": doc.metadata.get("category", ""),
                "description": doc.page_content,
                "keywords": doc.metadata.get("keywords", []),
                "career_paths": doc.metadata.get("career_paths", []) or doc.metadata.get("career_prospects", []),
                "related_subjects": doc.metadata.get("related_subjects", []),
                "skills_required": doc.metadata.get("skills_required", [])
            })
            
            # 충분한 다양한 학과를 찾으면 중단
            if len(results) >= top_k:
                break
        
        print(f"✅ 최종 결과: {len(results)}개 학과 (중복 제거 후)")
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

        # 컨텍스트 구성 (대학명 포함)
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            display = result.get('display_name', result['major_name'])
            career_paths = result.get('career_paths', [])
            career_str = ', '.join(career_paths[:3]) if career_paths else '다양한 진로 가능'
            keywords = result.get('keywords', [])
            keywords_str = ', '.join(keywords[:5]) if keywords else ''
            
            context_parts.append(
                f"{idx}. {display} ({result['category']})\n"
                f"   설명: {result['description'][:200]}...\n"
                f"   키워드: {keywords_str}\n"
                f"   진로: {career_str}"
            )

        context = "\n\n".join(context_parts)

        try:
            # LLM을 통한 추천 생성
            prompt_text = self.prompt.format(context=context, question=interests)
            print(f"📝 프롬프트 길이: {len(prompt_text)} 글자")
            response = self.llm.invoke(prompt_text)
            print(f"🤖 LLM 응답 길이: {len(response)} 글자")
            print(f"🤖 LLM 응답 미리보기: {response[:500]}...")

            # Output Parser를 통한 파싱 시도
            try:
                parsed_response = self.output_parser.parse(response)
                return {
                    "recommended_majors": parsed_response.recommended_majors,
                    "reasoning": parsed_response.reasoning,
                    "retrieved_context": search_results
                }
            except Exception as parse_error:
                print(f"⚠️ Pydantic 파싱 실패: {parse_error}")
                
                # JSON 직접 파싱 시도
                import re
                json_match = re.search(r'\{[^{}]*"recommended_majors"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    try:
                        import json
                        parsed_json = json.loads(json_match.group())
                        return {
                            "recommended_majors": parsed_json.get("recommended_majors", []),
                            "reasoning": parsed_json.get("reasoning", response),
                            "retrieved_context": search_results
                        }
                    except json.JSONDecodeError:
                        pass
                
                # 최종 fallback: 검색된 학과명 사용
                print(f"⚠️ JSON 파싱도 실패, fallback 사용")
                return {
                    "recommended_majors": [r['major_name'] for r in search_results[:5]],
                    "reasoning": response if response else "검색된 학과를 기반으로 추천합니다.",
                    "retrieved_context": search_results
                }

        except Exception as e:
            # LLM 호출 실패 시 검색 결과만 반환
            import traceback
            print(f"❌ LLM 생성 실패: {type(e).__name__}: {e}")
            print(f"❌ 상세 에러:\n{traceback.format_exc()}")
            return {
                "recommended_majors": [r['major_name'] for r in search_results[:5]],
                "reasoning": f"검색된 학과를 기반으로 추천합니다.",
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
            # 임베딩 모델 테스트
            test_embedding = self.embeddings.embed_query("테스트")
            
            # Retriever 테스트 (간단한 검색)
            test_docs = self.retriever.invoke("컴퓨터")
            
            return {
                "status": "healthy",
                "embedding_model": "jhgan/ko-sroberta-multitask",
                "embedding_dim": len(test_embedding),
                "llm_model": self.llm_model,
                "retriever_status": "ok" if test_docs else "no_results",
                "collection_name": self.collection_name
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
