"""
학교생활기록부 PDF 파싱 및 학생 프로필 추출

고등학생의 학교생활기록부 PDF를 분석하여 학과 추천에 필요한 정보를 자동으로 추출합니다.
OCR 기능으로 이미지 기반 PDF도 처리 가능합니다.
"""
import os
import re
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import PyPDF2
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# LLM 기반 분석은 규칙 기반으로 대체되어 현재 사용하지 않음


class SchoolRecordProfile(BaseModel):
    """학교생활기록부에서 추출한 학생 프로필"""

    academic_strengths: List[str] = Field(
        description="학생이 우수한 성적을 받거나 흥미를 보인 과목들 (예: 수학, 과학, 영어 등)"
    )

    extracurricular_activities: List[str] = Field(
        description="동아리, 자율활동, 봉사활동 등에서 나타난 주요 활동 및 관심사"
    )

    personality_traits: List[str] = Field(
        description="행동특성, 교사 관찰에서 나타난 성격적 특징 (예: 리더십, 협동심, 탐구심 등)"
    )

    career_interests: List[str] = Field(
        description="진로활동, 진로탐색에서 나타난 관심 분야 (예: 공학, 의학, 예술 등)"
    )

    teacher_observations: List[str] = Field(
        description="교과 세부능력특기사항 및 종합의견에서 교사가 관찰한 학습 태도 및 특징"
    )

    recommended_interests_summary: str = Field(
        description="위 정보를 종합하여 학생에게 적합할 것으로 보이는 학과/전공 방향 (200자 이내)"
    )


class SchoolRecordParser:
    """학교생활기록부 PDF 파서"""

    def __init__(self):
        """초기화"""
        pass

    def _clean_text(self, text: str) -> str:
        """
        OCR 텍스트 정제
        
        - 불필요한 줄바꿈 제거
        - 특수문자 정리
        - 연속 공백 제거
        """
        # 줄바꿈을 공백으로 변환 (단, 문단 구분은 유지)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # 연속 줄바꿈은 하나로
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 연속 공백 제거
        text = re.sub(r' {2,}', ' ', text)
        # OCR 오류로 자주 나타나는 패턴 정리
        text = re.sub(r'[^\w\s가-힣.,!?():\-]', '', text)
        return text.strip()

    def _extract_activities_improved(self, text: str) -> List[str]:
        """
        개선된 비교과 활동 추출
        
        - 동아리 활동
        - 봉사활동
        - 대회/경진대회 참가
        - 프로젝트/연구 활동
        """
        activities = []
        
        # 동아리 이름 패턴 (예: "수학동아리", "과학탐구반")
        club_patterns = [
            r'([가-힣]+(?:동아리|부|반|클럽))',
            r'([가-힣]+(?:탐구반|연구반|활동반))',
        ]
        for pattern in club_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:5]:
                if len(match) >= 3 and match not in activities:
                    activities.append(match)
        
        # 대회 참가 패턴
        competition_patterns = [
            r'([가-힣]+(?:대회|경진대회|올림피아드|경시대회))',
            r'([가-힣]+(?:공모전|발표대회))',
        ]
        for pattern in competition_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:
                if len(match) >= 4 and match not in activities:
                    activities.append(match)
        
        # 특별 활동 키워드
        special_activities = ['학생회', '반장', '부반장', '회장', '부회장', '임원']
        for activity in special_activities:
            if activity in text and activity not in activities:
                activities.append(activity)
        
        return activities[:10]  # 최대 10개

    def extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        """
        PDF에서 텍스트 추출 (OCR 지원)

        먼저 PyPDF2로 텍스트 추출을 시도하고,
        텍스트가 충분하지 않으면 OCR을 사용합니다.
        """
        try:
            # 1단계: PyPDF2로 텍스트 추출 시도
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            text = text.strip()

            # 충분한 텍스트가 추출되었으면 반환
            if len(text) > 100:
                print(f"✅ PyPDF2로 텍스트 추출 성공 ({len(text)} 글자)")
                return text

            # 2단계: 텍스트가 부족하면 OCR 시도
            print("⚠️ 텍스트가 부족합니다. OCR을 시도합니다...")
            return self.extract_text_with_ocr(pdf_file)

        except Exception as e:
            # PyPDF2 실패시 OCR로 폴백
            print(f"⚠️ PyPDF2 실패: {str(e)}. OCR을 시도합니다...")
            return self.extract_text_with_ocr(pdf_file)

    def extract_text_with_ocr(self, pdf_file: BytesIO) -> str:
        """
        OCR을 사용하여 PDF에서 텍스트 추출

        이미지 기반 PDF (스캔본)에서 텍스트를 추출합니다.
        """
        try:
            # PDF의 BytesIO 포인터를 처음으로 되돌림
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()

            # PDF를 이미지로 변환 (최대 10페이지까지)
            print("📄 PDF를 이미지로 변환 중...")
            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=10)

            print(f"🖼️ {len(images)}페이지 이미지 변환 완료")

            # 각 페이지에서 OCR 수행
            text = ""
            for i, image in enumerate(images, 1):
                print(f"  📝 페이지 {i} OCR 처리 중...")

                # Tesseract OCR 실행 (한글 + 영어)
                page_text = pytesseract.image_to_string(
                    image,
                    lang='kor+eng',  # 한글과 영어 인식
                    config='--psm 6'  # Page segmentation mode: Assume a single uniform block of text
                )

                if page_text:
                    text += page_text + "\n\n"

            text = text.strip()

            if len(text) > 100:
                print(f"✅ OCR 완료! 총 {len(text)} 글자 추출")
                return text
            else:
                raise ValueError(f"OCR로도 충분한 텍스트를 추출하지 못했습니다 (추출: {len(text)} 글자)")

        except Exception as e:
            raise ValueError(f"OCR 텍스트 추출 실패: {str(e)}")

    def analyze_school_record(self, pdf_content: bytes) -> SchoolRecordProfile:
        """
        학교생활기록부 PDF를 분석하여 학생 프로필 추출

        Args:
            pdf_content: PDF 파일의 바이트 데이터

        Returns:
            SchoolRecordProfile: 추출된 학생 프로필
        """
        # PDF에서 텍스트 추출
        pdf_file = BytesIO(pdf_content)
        record_text = self.extract_text_from_pdf(pdf_file)

        if not record_text or len(record_text) < 100:
            raise ValueError("PDF에서 충분한 텍스트를 추출할 수 없습니다. 파일이 올바른지 확인하세요.")

        # 텍스트 정제
        record_text = self._clean_text(record_text)

        # 규칙 기반 프로필 추출
        try:
            print("🔍 규칙 기반 텍스트 분석 시작...")

            # 1. 학업 우수 과목 추출 (성적이 좋은 과목들)
            academic_strengths = []
            subjects = ['국어', '수학', '영어', '과학', '사회', '역사', '물리', '화학', '생명과학', '지구과학', '경제', '정치']
            for subject in subjects:
                # "과목명" + "우수" or "뛰어남" or "높은" or "A" or "1등급" 패턴
                pattern = f'{subject}.*?(?:우수|뛰어남|높은|관심|흥미|A|1등급|2등급)'
                if re.search(pattern, record_text):
                    academic_strengths.append(subject)

            # 2. 비교과 활동 추출 (개선된 메서드 사용)
            extracurricular_activities = self._extract_activities_improved(record_text)
            
            # 추가 활동 키워드 기반 추출
            activity_keywords = ['봉사', '리더십', '독서', '체험', '캠프', '프로젝트', '토론', '발표']
            for keyword in activity_keywords:
                matches = re.findall(f'([가-힣a-zA-Z0-9\\s]+{keyword}[가-힣a-zA-Z0-9\\s]*)', record_text)
                for match in matches[:2]:  # 최대 2개
                    activity = match.strip()
                    if len(activity) < 50 and activity not in extracurricular_activities:
                        extracurricular_activities.append(activity)

            # 3. 성격 특성 추출
            personality_traits = []
            trait_keywords = ['리더십', '협동심', '탐구심', '창의성', '성실', '적극적', '배려', '책임감', '열정', '꼼꼼']
            for keyword in trait_keywords:
                if keyword in record_text:
                    personality_traits.append(keyword)

            # 4. 진로 관심사 추출
            career_interests = []
            career_keywords = ['공학', '의학', '교육', '경영', '경제', '법학', '예술', '디자인', '컴퓨터', 'IT', '연구', '과학', '인문', '사회']
            for keyword in career_keywords:
                if keyword in record_text:
                    career_interests.append(keyword)

            # 5. 교사 관찰 내용 추출 (세부능력특기사항에서)
            teacher_observations = []
            # "적극적", "노력", "참여" 등의 패턴 찾기
            obs_patterns = [
                r'적극적[으로]*\s*[가-힣\s]{5,30}',
                r'노력[하는]*\s*[가-힣\s]{5,30}',
                r'관심[을이]*\s*[가-힣\s]{5,30}'
            ]
            for pattern in obs_patterns:
                matches = re.findall(pattern, record_text)
                for match in matches[:2]:
                    obs = match.strip()
                    if obs not in teacher_observations:
                        teacher_observations.append(obs)

            # 6. 종합 요약 생성
            summary_parts = []
            if academic_strengths:
                summary_parts.append(f"{', '.join(academic_strengths[:3])} 과목에 강점")
            if career_interests:
                summary_parts.append(f"{', '.join(career_interests[:2])} 분야에 관심")
            if personality_traits:
                summary_parts.append(f"{', '.join(personality_traits[:2])} 특성 보유")

            recommended_interests_summary = ". ".join(summary_parts) if summary_parts else "다양한 분야에 관심"

            print(f"✅ 추출 완료:")
            print(f"  - 학업 강점: {academic_strengths}")
            print(f"  - 비교과: {len(extracurricular_activities)}개")
            print(f"  - 성격: {personality_traits}")
            print(f"  - 진로: {career_interests}")

            # Pydantic 모델 생성
            profile = SchoolRecordProfile(
                academic_strengths=academic_strengths[:5],  # 최대 5개
                extracurricular_activities=extracurricular_activities[:10],  # 최대 10개
                personality_traits=personality_traits[:5],
                career_interests=career_interests[:5],
                teacher_observations=teacher_observations[:5],
                recommended_interests_summary=recommended_interests_summary
            )

            return profile

        except Exception as e:
            print(f"❌ 프로필 추출 실패: {str(e)}")
            raise ValueError(f"학생 프로필 추출 실패: {str(e)}")

    def profile_to_interests_text(self, profile: SchoolRecordProfile) -> str:
        """
        추출된 프로필을 RAG 시스템에 입력할 수 있는 텍스트로 변환

        Args:
            profile: 추출된 학생 프로필

        Returns:
            str: RAG 시스템에 전달할 관심사 텍스트
        """
        interests_parts = []

        if profile.academic_strengths:
            interests_parts.append(f"잘하는 과목: {', '.join(profile.academic_strengths)}")

        if profile.extracurricular_activities:
            interests_parts.append(f"주요 활동: {', '.join(profile.extracurricular_activities[:5])}")

        if profile.career_interests:
            interests_parts.append(f"진로 관심사: {', '.join(profile.career_interests)}")

        if profile.personality_traits:
            interests_parts.append(f"성격 및 강점: {', '.join(profile.personality_traits[:3])}")

        return ". ".join(interests_parts)


# 싱글톤 인스턴스
_parser_instance = None


def get_parser() -> SchoolRecordParser:
    """학교생활기록부 파서 싱글톤 인스턴스 반환"""
    global _parser_instance

    if _parser_instance is None:
        _parser_instance = SchoolRecordParser()

    return _parser_instance
