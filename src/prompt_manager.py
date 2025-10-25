"""
프롬프트 버전 관리 시스템
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os


class PromptManager:
    """프롬프트 버전 관리 클래스"""

    def __init__(self, prompts_file: str = None):
        if prompts_file is None:
            # 기본 경로
            base_path = Path(__file__).parent.parent
            prompts_file = base_path / "prompts" / "prompts.yaml"

        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Any]:
        """YAML 파일에서 프롬프트 로드"""
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")

        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_type: str, version: str = None) -> Dict[str, Any]:
        """
        특정 프롬프트 가져오기

        Args:
            prompt_type: 프롬프트 타입 (예: 'basic_recommendation', 'rag_recommendation')
            version: 버전 (지정하지 않으면 current 버전 사용)

        Returns:
            프롬프트 딕셔너리 (system, user_template, temperature 등)
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt_config = self.prompts[prompt_type]

        # 버전이 지정되지 않으면 current 버전 사용
        if version is None:
            version = prompt_config.get('current', 'v1')

        if version not in prompt_config:
            raise ValueError(f"Unknown version '{version}' for prompt type '{prompt_type}'")

        return prompt_config[version]

    def format_user_prompt(self, prompt_type: str, version: str = None, **kwargs) -> str:
        """
        사용자 프롬프트 포맷팅

        Args:
            prompt_type: 프롬프트 타입
            version: 버전
            **kwargs: 템플릿에 들어갈 변수들 (interests, context 등)

        Returns:
            포맷팅된 사용자 프롬프트
        """
        prompt = self.get_prompt(prompt_type, version)
        user_template = prompt['user_template']
        return user_template.format(**kwargs)

    def get_system_message(self, prompt_type: str, version: str = None) -> str:
        """시스템 메시지 가져오기"""
        prompt = self.get_prompt(prompt_type, version)
        return prompt['system']

    def get_llm_params(self, prompt_type: str, version: str = None) -> Dict[str, Any]:
        """LLM 파라미터 가져오기 (temperature, num_predict 등)"""
        prompt = self.get_prompt(prompt_type, version)
        return {
            'temperature': prompt.get('temperature', 0.7),
            'num_predict': prompt.get('num_predict', 512)
        }

    def list_versions(self, prompt_type: str) -> list:
        """특정 프롬프트 타입의 모든 버전 나열"""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt_config = self.prompts[prompt_type]
        return [k for k in prompt_config.keys() if k != 'current']

    def get_current_version(self, prompt_type: str) -> str:
        """현재 사용 중인 버전 가져오기"""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        return self.prompts[prompt_type].get('current', 'v1')


# 전역 프롬프트 매니저 인스턴스
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """싱글톤 프롬프트 매니저 가져오기"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
