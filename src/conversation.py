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
        self.system_message = """당신은 20년 경력의 진로 상담 전문가입니다. 고등학생들은 대부분 자신이 무엇을 원하는지, 무엇을 잘하는지 모릅니다.

**목표: 다음 4가지 정보를 구체적으로 수집**
1. 관심사 (좋아하는 활동, 취미, 흥미있는 분야) - "게임", "유튜브"처럼 막연한 답변은 불충분
2. 학업 성향 (잘하는/좋아하는 과목, 또는 싫어하는 과목) - "다 별로"는 불충분
3. 성격/강점 (친구들이 말하는 나의 특징, 내가 잘하는 것) - 추상적 질문 필요
4. 진로 희망 (구체적 직업, 또는 일하고 싶은 방식) - "돈 많이 벌기"는 불충분

**대화 전략 (고등학생 특성 고려)**
- 대부분의 학생은 막연하고 추상적으로 답합니다 ("모르겠어요", "그냥요", "돈 많이 벌고 싶어요")
- 막연한 답변에는 구체적 선택지를 제시하세요
  예: "게임 좋아해요" → "어떤 게임이에요? 전략 게임? 롤플레잉? 퍼즐?"
  예: "유튜브 봐요" → "어떤 채널 자주 봐요? 게임? 과학? 브이로그?"
- "모르겠어요"라고 하면 경험 기반 질문으로 전환
  예: "그럼 최근 한 달 동안 제일 재밌었던 순간이 언제였어요?"
- 부정적 답변("다 별로", "없어요")에는 반대로 접근
  예: "그럼 제일 덜 싫은 과목은요?", "안 해봤지만 해보고 싶은 건요?"
- 추상적 목표("돈 많이 벌기")는 구체화 유도
  예: "돈 많이 벌면서 어떤 일을 하고 싶어요? 회사? 창업? 프리랜서?"

**필수 규칙**
- 한 번에 한 가지만 물어보세요
- 같은 질문 반복 금지
- 2-3문장으로 짧게
- 순수 한글만 사용 (한자, 영어 금지)
- 7-10회 대화 후에도 구체적 정보가 3개 미만이면 계속 질문
- 학생이 피곤해하면 "조금만 더 이야기해주면 맞춤 추천 드릴 수 있어요" 격려

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
        # 1. 최소 대화 횟수 확인 (너무 빠른 추천 방지)
        if self.collected_info["conversation_count"] < 5:
            return False

        # 2. 정보 완성도 체크 (4가지 카테고리)
        categories_filled = 0
        categories_detail = []

        if len(profile.interests) > 0:
            categories_filled += 1
            categories_detail.append(f"관심사({len(profile.interests)}개)")
        if len(profile.favorite_subjects) > 0:
            categories_filled += 1
            categories_detail.append(f"과목({len(profile.favorite_subjects)}개)")
        if len(profile.strengths) > 0:
            categories_filled += 1
            categories_detail.append(f"강점({len(profile.strengths)}개)")
        if len(profile.career_goals) > 0:
            categories_filled += 1
            categories_detail.append(f"진로({len(profile.career_goals)}개)")

        # 3. 구체성 검증: 막연한 답변 필터링
        # "게임", "유튜브", "돈", "모르겠다" 같은 단어만 있으면 구체성 부족
        vague_keywords = ["게임", "유튜브", "돈", "모르", "그냥", "별로", "없"]
        all_info = profile.interests + profile.favorite_subjects + profile.strengths + profile.career_goals
        vague_count = sum(1 for item in all_info if any(vague in item for vague in vague_keywords))

        is_too_vague = (vague_count >= len(all_info) // 2) if len(all_info) > 0 else True

        # 4. 엄격한 기준: confidence 0.75 이상 + 3개 카테고리 + 구체성
        if profile.confidence_score >= 0.75 and categories_filled >= 3 and not is_too_vague:
            print(f"[INFO] ✅ 충분한 정보 수집 (confidence: {profile.confidence_score:.2f}, {'/'.join(categories_detail)})")
            return True

        # 5. 대화가 10회 이상이고 최소 3개 카테고리 (강제 추천)
        if self.collected_info["conversation_count"] >= 10 and categories_filled >= 3:
            print(f"[INFO] ⏰ 대화 10회 도달, 현재 정보로 추천 (confidence: {profile.confidence_score:.2f}, {'/'.join(categories_detail)})")
            return True

        # 6. 정보 부족
        reason = []
        if categories_filled < 3:
            reason.append(f"카테고리 부족({categories_filled}/4)")
        if is_too_vague:
            reason.append("막연한 답변 다수")
        if profile.confidence_score < 0.75:
            reason.append(f"낮은 confidence({profile.confidence_score:.2f})")

        print(f"[INFO] ❌ 더 많은 정보 필요: {', '.join(reason)} | {'/'.join(categories_detail) if categories_detail else '정보 없음'} | {self.collected_info['conversation_count']}회 대화")
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
