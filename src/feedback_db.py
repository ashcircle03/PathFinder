"""
피드백 데이터베이스 관리
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class FeedbackDB:
    """피드백 데이터베이스 클래스"""

    def __init__(self):
        self.connection_params = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'pathfinder'),
            'user': os.getenv('POSTGRES_USER', 'pathfinder'),
            'password': os.getenv('POSTGRES_PASSWORD', 'pathfinder_password')
        }
        self.enabled = os.getenv('FEEDBACK_DB_ENABLED', 'true').lower() == 'true'

        if self.enabled:
            self._create_tables()

    def _get_connection(self):
        """데이터베이스 연결 가져오기"""
        return psycopg2.connect(**self.connection_params)

    def _create_tables(self):
        """필요한 테이블 생성"""
        if not self.enabled:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            recommendation_id VARCHAR(255) NOT NULL,
            interests TEXT NOT NULL,
            recommended_majors JSONB NOT NULL,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            is_helpful BOOLEAN NOT NULL,
            selected_majors JSONB,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_recommendation_id ON feedback(recommendation_id);
        CREATE INDEX IF NOT EXISTS idx_created_at ON feedback(created_at);
        CREATE INDEX IF NOT EXISTS idx_rating ON feedback(rating);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                conn.commit()
        except Exception as e:
            print(f"Failed to create tables: {e}")

    def save_feedback(
        self,
        recommendation_id: str,
        interests: str,
        recommended_majors: List[str],
        rating: int,
        is_helpful: bool,
        selected_majors: Optional[List[str]] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        피드백 저장

        Args:
            recommendation_id: 추천 ID
            interests: 사용자 관심사
            recommended_majors: 추천된 학과 리스트
            rating: 평점 (1-5)
            is_helpful: 도움이 되었는지 여부
            selected_majors: 사용자가 선택한 학과
            comment: 사용자 코멘트

        Returns:
            성공 여부
        """
        if not self.enabled:
            return False

        insert_sql = """
        INSERT INTO feedback
        (recommendation_id, interests, recommended_majors, rating, is_helpful, selected_majors, comment)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_sql,
                        (
                            recommendation_id,
                            interests,
                            json.dumps(recommended_majors, ensure_ascii=False),
                            rating,
                            is_helpful,
                            json.dumps(selected_majors, ensure_ascii=False) if selected_majors else None,
                            comment
                        )
                    )
                conn.commit()
            return True
        except Exception as e:
            print(f"Failed to save feedback: {e}")
            return False

    def get_average_rating(self, days: int = 30) -> Optional[float]:
        """최근 N일간 평균 평점 조회"""
        if not self.enabled:
            return None

        query_sql = """
        SELECT AVG(rating)::float as avg_rating
        FROM feedback
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query_sql, (days,))
                    result = cur.fetchone()
                    return result['avg_rating'] if result else None
        except Exception as e:
            print(f"Failed to get average rating: {e}")
            return None

    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """피드백 통계 조회"""
        if not self.enabled:
            return {}

        stats_sql = """
        SELECT
            COUNT(*) as total_count,
            AVG(rating)::float as avg_rating,
            SUM(CASE WHEN is_helpful THEN 1 ELSE 0 END) as helpful_count,
            SUM(CASE WHEN NOT is_helpful THEN 1 ELSE 0 END) as not_helpful_count
        FROM feedback
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(stats_sql, (days,))
                    result = cur.fetchone()
                    return dict(result) if result else {}
        except Exception as e:
            print(f"Failed to get feedback stats: {e}")
            return {}

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 피드백 조회"""
        if not self.enabled:
            return []

        query_sql = """
        SELECT
            id, recommendation_id, interests, recommended_majors,
            rating, is_helpful, selected_majors, comment, created_at
        FROM feedback
        ORDER BY created_at DESC
        LIMIT %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query_sql, (limit,))
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            print(f"Failed to get recent feedback: {e}")
            return []


# 전역 피드백 DB 인스턴스
_feedback_db = None


def get_feedback_db() -> FeedbackDB:
    """싱글톤 피드백 DB 가져오기"""
    global _feedback_db
    if _feedback_db is None:
        try:
            _feedback_db = FeedbackDB()
        except Exception as e:
            print(f"Failed to initialize FeedbackDB: {e}")
            # 더미 클래스 반환 (DB 없이도 동작하도록)
            class DummyFeedbackDB:
                enabled = False
                def save_feedback(self, *args, **kwargs): return False
                def get_average_rating(self, *args, **kwargs): return None
                def get_feedback_stats(self, *args, **kwargs): return {}
                def get_recent_feedback(self, *args, **kwargs): return []
            _feedback_db = DummyFeedbackDB()
    return _feedback_db
