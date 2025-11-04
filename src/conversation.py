"""
LangChain 기반 대화형 진로 상담 시스템
"""
import os
from typing import Dict, Any, List, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


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

        # Memory 설정 (대화 이력 저장)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="response"
        )

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
- 3-5회 대화 후 충분한 정보가 모이면 "학과 추천을 시작하겠습니다"라고 말하세요

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

            # 관심사 키워드 추출 (간단한 휴리스틱)
            self._extract_keywords(user_message, response)

            # 충분한 정보가 수집되었는지 판단
            is_ready = self._check_if_ready_to_recommend(response)

            return {
                "response": response,
                "is_ready_to_recommend": is_ready,
                "collected_info": self.collected_info,
                "conversation_count": self.collected_info["conversation_count"]
            }

        except Exception as e:
            print(f"[ERROR] 대화 처리 실패: {e}")
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "is_ready_to_recommend": False,
                "error": str(e)
            }

    def _extract_keywords(self, user_message: str, ai_response: str):
        """사용자 메시지와 AI 응답에서 키워드 추출"""
        # 관심사 관련 키워드
        interest_keywords = ["좋아", "관심", "흥미", "재미", "취미", "즐기"]
        # 과목 관련 키워드
        subject_keywords = ["수학", "과학", "영어", "국어", "사회", "역사", "물리", "화학", "생물"]
        # 성격 관련 키워드
        personality_keywords = ["성격", "리더", "창의", "분석", "꼼꼼", "외향", "내향"]

        message_lower = user_message.lower()

        # 간단한 키워드 매칭 (실제로는 NLP 라이브러리 사용 권장)
        if any(keyword in message_lower for keyword in interest_keywords):
            words = user_message.split()
            self.collected_info["interests"].extend([w for w in words if len(w) > 1])

        for subject in subject_keywords:
            if subject in message_lower:
                if subject not in self.collected_info["subjects"]:
                    self.collected_info["subjects"].append(subject)

    def _check_if_ready_to_recommend(self, response: str) -> bool:
        """학과 추천을 할 준비가 되었는지 판단"""
        # 1. 대화 횟수가 3회 이상인지
        if self.collected_info["conversation_count"] < 3:
            return False

        # 2. 응답에 "추천" 키워드가 있는지
        recommend_keywords = ["추천", "학과를 알려", "소개", "제안"]
        if any(keyword in response for keyword in recommend_keywords):
            return True

        # 3. 충분한 정보가 수집되었는지
        has_enough_info = (
            len(self.collected_info["interests"]) > 0 or
            len(self.collected_info["subjects"]) > 0
        )

        if self.collected_info["conversation_count"] >= 5 and has_enough_info:
            return True

        return False

    def get_collected_interests(self) -> str:
        """수집된 관심사를 문자열로 반환 (RAG 검색용)"""
        interests_list = []

        if self.collected_info["interests"]:
            interests_list.extend(self.collected_info["interests"][:5])

        if self.collected_info["subjects"]:
            interests_list.extend(self.collected_info["subjects"])

        # 대화 히스토리에서 추가 추출
        chat_history = self.memory.load_memory_variables({})
        if chat_history and "chat_history" in chat_history:
            # 간단하게 마지막 몇 개 메시지만 추출
            messages = chat_history["chat_history"]
            for msg in messages[-3:]:
                if isinstance(msg, HumanMessage):
                    words = msg.content.split()
                    interests_list.extend([w for w in words if len(w) > 2])

        # 중복 제거 및 문자열 변환
        unique_interests = list(set(interests_list))
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
