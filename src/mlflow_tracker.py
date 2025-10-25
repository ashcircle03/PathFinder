"""
MLflow 실험 추적 시스템
"""
import mlflow
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager


class MLflowTracker:
    """MLflow 실험 추적 클래스"""

    def __init__(self):
        # MLflow 서버 URL 설정
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        # 실험 이름 설정
        experiment_name = "pathfinder-recommendations"
        mlflow.set_experiment(experiment_name)

        self.enabled = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """
        MLflow 실행 시작 (컨텍스트 매니저)

        Usage:
            with tracker.start_run("recommendation"):
                tracker.log_param("model", "qwen2.5:32b")
                tracker.log_metric("score", 0.85)
        """
        if self.enabled:
            with mlflow.start_run(run_name=run_name) as run:
                yield run
        else:
            # MLflow가 비활성화된 경우 아무것도 하지 않음
            yield None

    def log_param(self, key: str, value: Any):
        """파라미터 로깅"""
        if self.enabled:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Failed to log param {key}: {e}")

    def log_params(self, params: Dict[str, Any]):
        """여러 파라미터 한번에 로깅"""
        if self.enabled:
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"Failed to log params: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """메트릭 로깅"""
        if self.enabled:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """여러 메트릭 한번에 로깅"""
        if self.enabled:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"Failed to log metrics: {e}")

    def log_text(self, text: str, artifact_file: str):
        """텍스트 아티팩트 로깅"""
        if self.enabled:
            try:
                mlflow.log_text(text, artifact_file)
            except Exception as e:
                print(f"Failed to log text artifact {artifact_file}: {e}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """딕셔너리 아티팩트 로깅 (JSON)"""
        if self.enabled:
            try:
                mlflow.log_dict(dictionary, artifact_file)
            except Exception as e:
                print(f"Failed to log dict artifact {artifact_file}: {e}")

    def set_tag(self, key: str, value: Any):
        """태그 설정"""
        if self.enabled:
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                print(f"Failed to set tag {key}: {e}")

    def set_tags(self, tags: Dict[str, Any]):
        """여러 태그 한번에 설정"""
        if self.enabled:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                print(f"Failed to set tags: {e}")

    def log_recommendation_run(
        self,
        interests: str,
        model: str,
        prompt_version: str,
        temperature: float,
        num_predict: int,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        recommended_majors: list,
        avg_similarity_score: float,
        num_retrieved: int
    ):
        """
        추천 실행 전체를 로깅하는 헬퍼 함수

        Args:
            interests: 사용자 관심사
            model: LLM 모델 이름
            prompt_version: 프롬프트 버전
            temperature: Temperature 파라미터
            num_predict: 생성 토큰 수
            retrieval_time: 검색 시간
            generation_time: 생성 시간
            total_time: 전체 시간
            recommended_majors: 추천된 학과 리스트
            avg_similarity_score: 평균 유사도 점수
            num_retrieved: 검색된 문서 수
        """
        if not self.enabled:
            return

        try:
            with self.start_run("recommendation"):
                # 파라미터 로깅
                self.log_params({
                    "model": model,
                    "prompt_version": prompt_version,
                    "temperature": temperature,
                    "num_predict": num_predict,
                })

                # 메트릭 로깅
                self.log_metrics({
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "avg_similarity_score": avg_similarity_score,
                    "num_retrieved": num_retrieved,
                    "num_recommended": len(recommended_majors)
                })

                # 입력/출력 로깅
                self.log_text(interests, "input/interests.txt")
                self.log_dict({
                    "recommended_majors": recommended_majors
                }, "output/recommendations.json")

                # 태그 설정
                self.set_tags({
                    "experiment_type": "recommendation",
                    "rag_enabled": "true"
                })

        except Exception as e:
            print(f"Failed to log recommendation run: {e}")


# 전역 MLflow 트래커 인스턴스
_mlflow_tracker = None


def get_mlflow_tracker() -> MLflowTracker:
    """싱글톤 MLflow 트래커 가져오기"""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker()
    return _mlflow_tracker
