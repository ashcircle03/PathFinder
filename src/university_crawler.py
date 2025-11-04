"""
대학 입학처 홈페이지 크롤러
실제 대학 홈페이지에서 학과 정보를 수집합니다.
"""
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime


class UniversityCrawler:
    """대학 입학처 정보 크롤러"""

    def __init__(self):
        self.universities = {
            "서울대학교": {
                "url": "https://admission.snu.ac.kr",
                "departments_url": "https://www.snu.ac.kr/academics/undergraduate",
            },
            "연세대학교": {
                "url": "https://admission.yonsei.ac.kr",
                "departments_url": "https://www.yonsei.ac.kr/sc/support/departments.jsp",
            },
            "고려대학교": {
                "url": "https://admission.korea.ac.kr",
                "departments_url": "https://www.korea.ac.kr/mbshome/mbs/university/subview.do?id=university_010401000000",
            },
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.crawled_data = []

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """비동기로 페이지를 가져옵니다."""
        try:
            async with session.get(url, headers=self.headers, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"⚠️ {url} 접근 실패: HTTP {response.status}")
                    return None
        except Exception as e:
            print(f"❌ {url} 크롤링 오류: {e}")
            return None

    def parse_department_info(self, html: str, university_name: str) -> List[Dict[str, Any]]:
        """HTML에서 학과 정보를 파싱합니다."""
        soup = BeautifulSoup(html, "html.parser")
        departments = []

        # 일반적인 학과 정보 추출 패턴
        # 실제 HTML 구조에 맞춰 수정 필요
        dept_elements = soup.find_all(["div", "li", "tr"], class_=re.compile(r"(dept|department|major|college)"))

        for element in dept_elements:
            dept_info = self._extract_department_details(element, university_name)
            if dept_info:
                departments.append(dept_info)

        return departments

    def _extract_department_details(self, element, university_name: str) -> Optional[Dict[str, Any]]:
        """개별 학과 요소에서 상세 정보를 추출합니다."""
        try:
            # 학과명 추출
            name = element.find(["h3", "h4", "strong", "b"])
            if not name:
                return None

            dept_name = name.get_text(strip=True)

            # 설명 추출
            description = element.find(["p", "div"], class_=re.compile(r"(desc|description|intro)"))
            desc_text = description.get_text(strip=True) if description else ""

            # 기본 정보 구성
            return {
                "university": university_name,
                "name": dept_name,
                "description": desc_text,
                "url": element.find("a")["href"] if element.find("a") else "",
                "crawled_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"⚠️ 학과 정보 추출 실패: {e}")
            return None

    async def crawl_university(self, university_name: str, urls: Dict[str, str]) -> List[Dict[str, Any]]:
        """특정 대학의 학과 정보를 크롤링합니다."""
        print(f"🔍 {university_name} 크롤링 시작...")

        async with aiohttp.ClientSession() as session:
            # 학과 목록 페이지 가져오기
            html = await self.fetch_page(session, urls["departments_url"])

            if html:
                departments = self.parse_department_info(html, university_name)
                print(f"✅ {university_name}: {len(departments)}개 학과 발견")
                return departments
            else:
                print(f"❌ {university_name} 크롤링 실패")
                return []

    async def crawl_all_universities(self) -> List[Dict[str, Any]]:
        """모든 대학의 학과 정보를 병렬로 크롤링합니다."""
        print("🚀 대학 입학처 정보 크롤링 시작...")

        tasks = [
            self.crawl_university(name, urls)
            for name, urls in self.universities.items()
        ]

        results = await asyncio.gather(*tasks)

        # 결과 병합
        all_departments = []
        for dept_list in results:
            all_departments.extend(dept_list)

        self.crawled_data = all_departments
        print(f"✅ 총 {len(all_departments)}개 학과 정보 수집 완료")

        return all_departments

    def save_to_json(self, filepath: str = "data/university_departments.json"):
        """크롤링한 데이터를 JSON 파일로 저장합니다."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.crawled_data, f, ensure_ascii=False, indent=2)
            print(f"💾 데이터 저장 완료: {filepath}")
        except Exception as e:
            print(f"[ERROR] 데이터 저장 실패: {e}")

    async def crawl_specific_department_details(
        self, department_url: str, university_name: str
    ) -> Dict[str, Any]:
        """특정 학과의 상세 페이지를 크롤링합니다."""
        async with aiohttp.ClientSession() as session:
            html = await self.fetch_page(session, department_url)

            if not html:
                return {}

            soup = BeautifulSoup(html, "html.parser")

            details = {
                "university": university_name,
                "curriculum": [],
                "admission_info": {},
                "career_prospects": [],
            }

            # 커리큘럼 정보 추출
            curriculum_section = soup.find(["div", "section"], class_=re.compile(r"(curriculum|course)"))
            if curriculum_section:
                courses = curriculum_section.find_all(["li", "div"])
                details["curriculum"] = [course.get_text(strip=True) for course in courses[:20]]

            # 입학 정보 추출
            admission_section = soup.find(["div", "section"], class_=re.compile(r"(admission|entrance)"))
            if admission_section:
                details["admission_info"] = {
                    "content": admission_section.get_text(strip=True)[:500]
                }

            # 진로 정보 추출
            career_section = soup.find(["div", "section"], class_=re.compile(r"(career|job|future)"))
            if career_section:
                careers = career_section.find_all(["li", "p"])
                details["career_prospects"] = [career.get_text(strip=True) for career in careers[:10]]

            return details


class MockUniversityCrawler:
    """
    실제 크롤링 대신 샘플 데이터를 생성하는 Mock 크롤러
    (대학 웹사이트 부하 방지 및 개발/테스트용)
    """

    def __init__(self):
        self.universities = [
            "서울대학교", "연세대학교", "고려대학교", "카이스트", "포스텍",
            "성균관대학교", "한양대학교", "중앙대학교", "경희대학교", "이화여자대학교"
        ]

    async def crawl_all_universities(self) -> List[Dict[str, Any]]:
        """샘플 대학 학과 정보를 생성합니다."""
        print("[Mock Crawler] 샘플 데이터 생성 중...")

        sample_departments = []

        # 기존 majors.json의 학과들을 각 대학에 매핑
        base_majors = [
            {
                "name": "컴퓨터공학과",
                "category": "공학",
                "description_template": "{univ} 컴퓨터공학과는 소프트웨어와 하드웨어를 아우르는 종합적인 컴퓨터 과학 교육을 제공합니다. AI, 빅데이터, 사이버보안 등 최신 기술을 배우며, 산학협력 프로그램을 통해 실무 경험을 쌓을 수 있습니다.",
                "keywords": ["프로그래밍", "AI", "빅데이터", "소프트웨어", "알고리즘"],
                "admission_quota": "60-100명",
                "graduation_requirements": "전공 60학점 이상, 졸업논문 또는 캡스톤 프로젝트",
            },
            {
                "name": "경영학과",
                "category": "상경",
                "description_template": "{univ} 경영학과는 경영 전반의 이론과 실무를 겸비한 글로벌 리더를 양성합니다. 마케팅, 재무, 인사조직, 경영전략 등 다양한 분야를 학습하며, 국내외 기업과의 인턴십 기회를 제공합니다.",
                "keywords": ["경영", "마케팅", "재무", "전략", "리더십"],
                "admission_quota": "80-120명",
                "graduation_requirements": "전공 54학점 이상, 경영학 인증 프로그램 이수",
            },
            {
                "name": "인공지능학과",
                "category": "공학",
                "description_template": "{univ} 인공지능학과는 미래 AI 기술을 선도할 전문 인력을 양성합니다. 머신러닝, 딥러닝, 자연어처리, 컴퓨터비전 등을 집중적으로 학습하며, 최신 GPU 서버와 연구 환경을 제공합니다.",
                "keywords": ["AI", "머신러닝", "딥러닝", "데이터과학", "빅데이터"],
                "admission_quota": "40-60명",
                "graduation_requirements": "전공 66학점 이상, AI 프로젝트 필수",
            },
            {
                "name": "심리학과",
                "category": "인문사회",
                "description_template": "{univ} 심리학과는 인간의 마음과 행동을 과학적으로 탐구합니다. 임상심리, 상담심리, 산업심리 등 다양한 세부 전공을 제공하며, 심리 실험실과 상담 센터에서 실습 경험을 쌓을 수 있습니다.",
                "keywords": ["심리", "상담", "인간행동", "정신건강", "뇌과학"],
                "admission_quota": "50-70명",
                "graduation_requirements": "전공 48학점 이상, 심리학 실험 실습 이수",
            },
            {
                "name": "전자전기공학과",
                "category": "공학",
                "description_template": "{univ} 전자전기공학과는 반도체, 통신, 로봇, 에너지 분야의 핵심 기술을 다룹니다. 첨단 실험 장비를 활용한 실습과 대기업 연구소와의 산학협력을 통해 실무형 인재를 양성합니다.",
                "keywords": ["전자", "반도체", "통신", "회로", "임베디드"],
                "admission_quota": "70-90명",
                "graduation_requirements": "전공 60학점 이상, 종합설계 프로젝트",
            },
        ]

        for univ in self.universities:
            for major in base_majors:
                dept_info = {
                    "university": univ,
                    "name": major["name"],
                    "category": major["category"],
                    "description": major["description_template"].format(univ=univ),
                    "keywords": major["keywords"],
                    "admission_info": {
                        "quota": major["admission_quota"],
                        "requirements": major["graduation_requirements"],
                        "website": f"https://{univ.replace('대학교', '')}.ac.kr/dept/{major['name'].replace('과', '')}",
                    },
                    "curriculum": [
                        "1학년: 전공기초 및 교양",
                        "2학년: 전공필수 과목",
                        "3학년: 전공심화 및 선택",
                        "4학년: 캡스톤 프로젝트 및 졸업연구",
                    ],
                    "career_prospects": [
                        f"{major['category']} 관련 대기업 취업",
                        "연구소 및 국책기관",
                        "대학원 진학",
                        "창업",
                    ],
                    "crawled_at": datetime.now().isoformat(),
                    "source": "mock_data",
                }
                sample_departments.append(dept_info)

        print(f"[OK] {len(sample_departments)}개 샘플 학과 정보 생성 완료")
        return sample_departments

    def save_to_json(self, filepath: str = "data/university_departments.json"):
        """생성한 샘플 데이터를 JSON 파일로 저장합니다."""
        asyncio.run(self._save_async(filepath))

    async def _save_async(self, filepath: str):
        data = await self.crawl_all_universities()
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] 샘플 데이터 저장 완료: {filepath}")
        except Exception as e:
            print(f"[ERROR] 데이터 저장 실패: {e}")


async def main():
    """테스트용 메인 함수"""
    # Mock 크롤러 사용 (개발/테스트용)
    crawler = MockUniversityCrawler()
    departments = await crawler.crawl_all_universities()

    # JSON 저장
    try:
        with open("data/university_departments.json", "w", encoding="utf-8") as f:
            json.dump(departments, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] 데이터 저장 완료: data/university_departments.json")
    except Exception as e:
        print(f"[ERROR] 데이터 저장 실패: {e}")

    # 통계 출력
    universities = set(d["university"] for d in departments)
    print(f"\n[STATS] 크롤링 통계:")
    print(f"   - 대학 수: {len(universities)}")
    print(f"   - 학과 수: {len(departments)}")
    print(f"   - 대학 목록: {', '.join(universities)}")


if __name__ == "__main__":
    asyncio.run(main())
