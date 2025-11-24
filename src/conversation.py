"""
LangChain 기반 대화형 진로 상담 시스템
"""
import os
from typing import Dict, Any, List, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class StudentProfile(BaseModel):
    """학생 프로필 구조화 모델"""
    interests: List[str] = Field(
        description="학생의 관심사 및 활동 (예: 프로그래밍, 게임, 독서)",
        default_factory=list
    )
    favorite_subjects: List[str] = Field(
        description="좋아하는 과목 (예: 수학, 과학, 영어)",
        default_factory=list
    )
    strengths: List[str] = Field(
        description="강점 및 성격 특성 (예: 리더십, 창의력, 분석력)",
        default_factory=list
    )
    career_goals: List[str] = Field(
        description="진로 목표 및 하고 싶은 일",
        default_factory=list
    )
    confidence_score: float = Field(
        description="정보 충분도 (0.0 ~ 1.0, 0.7 이상이면 추천 가능)",
        default=0.0,
        ge=0.0,
        le=1.0
    )


class ConversationResponse(BaseModel):
    """대화 응답 모델"""
    response: str = Field(description="상담사의 응답")
    next_question: Optional[str] = Field(description="다음 질문 (선택사항)", default=None)
    is_ready_to_recommend: bool = Field(
        description="학과 추천을 할 수 있는 충분한 정보가 수집되었는지 여부",
        default=False
    )


class CareerCounselorConversation:
    """대화형 진로 상담 시스템"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm_model = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")

        print(f"[ConversationChain] 세션 {session_id} 초기화 중...")

        # LLM 초기화
        self.llm = OllamaLLM(
            model=self.llm_model,
            base_url=ollama_host,
            temperature=0.7,
            system="You are a friendly Korean university counselor. Always respond in pure Korean (Hangul only)."
        )

        # Memory 설정 (대화 이력 저장 + 자동 요약)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="response",
            max_token_limit=500  # 최근 500토큰 유지, 나머지는 요약
        )
        print(f"  ✅ ConversationSummaryBufferMemory 설정 (max_tokens: 500)")

        # 프롬프트 템플릿 설정
        self._setup_prompts()

        # Conversation Chain 생성
        self.conversation_chain = self._create_conversation_chain()

        # 상태 관리
        self.collected_info = {
            "interests": [],
            "subjects": [],
            "personality": [],
            "goals": [],
            "conversation_count": 0
        }

        # Pydantic Parser 설정
        self.profile_parser = PydanticOutputParser(pydantic_object=StudentProfile)

        print(f"[OK] 대화형 상담 시스템 초기화 완료 (세션: {session_id})")

    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        self.system_message = """당신은 20년 경력의 진로 상담 전문가입니다.

고등학생과 대화하며 다음 정보를 자연스럽게 수집하세요:
1. 관심사 (좋아하는 활동, 취미, 흥미있는 분야)
2. 잘하는 과목 (수학, 과학, 국어, 영어, 사회 등)
3. 성격 및 강점 (리더십, 창의력, 분석력 등)
4. 진로 목표 (하고 싶은 일, 되고 싶은 직업)

대화 스타일:
- 친근하고 격려하는 톤으로 대화하세요
- 한 번에 한 가지 질문만 하세요
- 학생의 답변에 공감하고 긍정적으로 반응하세요
- 학생이 짧게 답하거나 부정적이면, 질문 방식을 바꿔 다른 각도로 접근하세요
- 학생이 "없어", "모르겠어", "다른 건 없어" 등으로 답하면 구체적인 예시를 들어가며 새로운 주제로 전환하세요
- 같은 질문을 반복하지 마세요
- 3-5회 대화 후 4가지 카테고리 중 최소 3개 정보가 모이면 "학과 추천을 시작하겠습니다"라고 말하세요

감정 감지 및 대응:
- 학생이 짜증내거나 피곤해하면: 대화 속도를 조절하고 부담을 줄이세요
- 학생이 소극적이면: 선택형 질문이나 구체적 예시를 제공하세요
- 학생이 적극적이면: 더 깊이있는 질문으로 확장하세요

중요한 규칙:
- 반드시 순수 한국어로만 답변하세요 (한자, 영어, 일본어 사용 금지)
- 학생을 존중하고 격려하는 태도를 유지하세요
- 짧고 간결하게 답변하세요 (2-3문장)

{chat_history}

학생: {input}
상담사:"""

        self.prompt_template = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=self.system_message
        )

    def _create_conversation_chain(self):
        """Conversation Chain 생성"""
        # Simple ConversationChain (프롬프트 + LLM + Memory)
        chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=True,  # 디버깅용
            output_key="response"
        )
        return chain

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        사용자 메시지를 받아 대화를 진행합니다.

        Args:
            user_message: 사용자의 입력 메시지

        Returns:
            응답 딕셔너리 (response, is_ready_to_recommend 등)
        """
        self.collected_info["conversation_count"] += 1

        # LLM을 통한 대화 진행
        try:
            response = self.conversation_chain.predict(input=user_message)

            # 초기 대화(1-2회)는 프로필 추출 생략 (성능 최적화)
            if self.collected_info["conversation_count"] >= 2:
                # 구조화된 프로필 추출 (Pydantic 기반)
                profile = self._extract_profile_structured()

                # 충분한 정보가 수집되었는지 판단 (LLM 기반 confidence score)
                is_ready = self._check_readiness_with_llm(profile)
                confidence_score = profile.confidence_score
            else:
                # 초기 대화: 프로필 추출 생략
                is_ready = False
                confidence_score = 0.0
                print(f"[INFO] 초기 대화 ({self.collected_info['conversation_count']}회) - 프로필 추출 생략")

            return {
                "response": response,
                "is_ready_to_recommend": is_ready,
                "collected_info": self.collected_info,
                "conversation_count": self.collected_info["conversation_count"],
                "confidence_score": confidence_score
            }

        except Exception as e:
            print(f"[ERROR] 대화 처리 실패: {e}")
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "is_ready_to_recommend": False,
                "error": str(e)
            }

    def _extract_profile_structured(self) -> StudentProfile:
        """LLM을 사용하여 대화에서 구조화된 프로필 추출"""
        try:
            # 대화 이력 가져오기
            memory_variables = self.memory.load_memory_variables({})
            chat_history = str(memory_variables.get("chat_history", ""))

            if not chat_history or chat_history == "[]":
                # 대화가 없으면 빈 프로필 반환
                return StudentProfile()

            # 프로필 추출 프롬프트
            extraction_prompt = f"""다음 대화 내용을 분석하여 학생의 프로필을 추출하세요.

대화 내용:
{chat_history}

위 대화에서 다음 정보를 추출하세요:
1. 관심사 (좋아하는 활동, 취미)
2. 좋아하는 과목
3. 강점 및 성격 특성
4. 진로 목표

추출할 수 있는 정보만 포함하고, 없는 정보는 빈 리스트로 두세요.
또한 충분한 정보가 수집되었는지 판단하여 confidence_score를 설정하세요 (0.0~1.0).
- 0.7 이상: 학과 추천 가능 (4가지 카테고리 중 3개 이상 정보 있음)
- 0.5~0.7: 조금 더 정보 필요
- 0.5 미만: 많은 정보 필요

{self.profile_parser.get_format_instructions()}

**중요: 반드시 순수 JSON만 출력하세요. 주석, 설명, 코드 블록 마커(```json, ```)는 절대 포함하지 마세요.**

추출된 프로필:"""

            response = self.llm.invoke(extraction_prompt)
            profile = self.profile_parser.parse(response)

            # collected_info 업데이트
            self.collected_info["interests"] = profile.interests
            self.collected_info["subjects"] = profile.favorite_subjects
            self.collected_info["personality"] = profile.strengths
            self.collected_info["goals"] = profile.career_goals

            return profile

        except Exception as e:
            print(f"[WARN] 프로필 추출 실패, fallback 사용: {e}")
            # 실패 시 낮은 confidence score 반환
            return StudentProfile(
                interests=self.collected_info.get("interests", []),
                favorite_subjects=self.collected_info.get("subjects", []),
                confidence_score=0.3
            )

    def _check_readiness_with_llm(self, profile: StudentProfile) -> bool:
        """LLM confidence score를 기반으로 추천 준비 여부 판단"""
        # 1. 최소 대화 횟수 확인
        if self.collected_info["conversation_count"] < 3:
            return False

        # 2. 정보 완성도 체크 (4가지 카테고리 중 최소 3개 필요)
        categories_filled = 0
        if len(profile.interests) > 0:
            categories_filled += 1
        if len(profile.favorite_subjects) > 0:
            categories_filled += 1
        if len(profile.strengths) > 0:
            categories_filled += 1
        if len(profile.career_goals) > 0:
            categories_filled += 1

        # 3. LLM이 판단한 confidence score 확인 (0.7 이상 + 3개 카테고리 필수)
        if profile.confidence_score >= 0.7 and categories_filled >= 3:
            print(f"[INFO] 충분한 정보 수집됨 (confidence: {profile.confidence_score:.2f}, categories: {categories_filled}/4)")
            return True

        # 4. 대화가 7회 이상이고 최소 2개 카테고리 정보 있으면 추천
        if self.collected_info["conversation_count"] >= 7 and categories_filled >= 2:
            print(f"[INFO] 대화 7회 이상, 최소 정보 확보 → 추천 진행 (confidence: {profile.confidence_score:.2f}, categories: {categories_filled}/4)")
            return True

        print(f"[INFO] 더 많은 정보 필요 (confidence: {profile.confidence_score:.2f}, categories: {categories_filled}/4, count: {self.collected_info['conversation_count']})")
        return False

    def get_collected_interests(self) -> str:
        """수집된 관심사를 문자열로 반환 (RAG 검색용) - 구조화된 프로필 활용"""
        # Pydantic 구조화 추출 활용 (노이즈 자동 제거)
        profile = self._extract_profile_structured()

        interests_list = []

        # 1. 구조화된 관심사 (LLM이 의미 있는 키워드만 추출)
        interests_list.extend(profile.interests[:5])  # 최대 5개

        # 2. 좋아하는 과목
        interests_list.extend(profile.favorite_subjects)

        # 3. 강점 및 성격 (선택적, 최대 3개)
        interests_list.extend(profile.strengths[:3])

        # 4. 진로 목표
        interests_list.extend(profile.career_goals[:2])  # 최대 2개

        # 중복 제거 (순서 유지)
        unique_interests = list(dict.fromkeys(interests_list))

        print(f"[INFO] 추출된 관심사: {unique_interests[:10]}")
        return ", ".join(unique_interests[:10])

    def reset_session(self):
        """세션 초기화"""
        self.memory.clear()
        self.collected_info = {
            "interests": [],
            "subjects": [],
            "personality": [],
            "goals": [],
            "conversation_count": 0
        }
        print(f"[RESET] 세션 {self.session_id} 초기화 완료")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 히스토리를 반환"""
        chat_history = self.memory.load_memory_variables({})

        if not chat_history or "chat_history" not in chat_history:
            return []

        messages = chat_history["chat_history"]
        history = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return history


# 전역 세션 저장소 (간단한 메모리 기반)
_conversation_sessions: Dict[str, CareerCounselorConversation] = {}


def get_conversation_session(session_id: str) -> CareerCounselorConversation:
    """세션 ID로 대화 세션 가져오기 (없으면 새로 생성)"""
    global _conversation_sessions

    if session_id not in _conversation_sessions:
        _conversation_sessions[session_id] = CareerCounselorConversation(session_id)

    return _conversation_sessions[session_id]


def delete_conversation_session(session_id: str) -> bool:
    """세션 삭제"""
    global _conversation_sessions

    if session_id in _conversation_sessions:
        del _conversation_sessions[session_id]
        print(f"[DELETE] 세션 {session_id} 삭제 완료")
        return True

    return False
