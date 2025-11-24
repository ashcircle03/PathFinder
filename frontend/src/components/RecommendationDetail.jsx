import { useState } from 'react'
import './RecommendationDetail.css'

function RecommendationDetail({ result, onBack }) {
  const [expandedMajor, setExpandedMajor] = useState(null)

  if (!result) return null

  const toggleMajor = (index) => {
    setExpandedMajor(expandedMajor === index ? null : index)
  }

  return (
    <div className="recommendation-detail">
      <div className="detail-header">
        <button onClick={onBack} className="back-button">
          ← 돌아가기
        </button>
        <h2>🎯 상세 추천 결과</h2>
      </div>

      <div className="detail-content">
        <section className="summary-section">
          <h3>📊 추천 요약</h3>
          <div className="summary-card">
            <div className="summary-item">
              <span className="summary-label">추천 학과 수</span>
              <span className="summary-value">{result.recommended_majors.length}개</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">분석 기반</span>
              <span className="summary-value">RAG + LLM</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">검색된 정보</span>
              <span className="summary-value">{result.retrieved_context?.length || 0}개 학과</span>
            </div>
          </div>
        </section>

        <section className="reasoning-section">
          <h3>💡 종합 추천 이유</h3>
          <div className="reasoning-card">
            <p>{result.reasoning}</p>
          </div>
        </section>

        <section className="majors-section">
          <h3>🎓 추천 학과 상세</h3>
          <div className="majors-detailed-list">
            {result.recommended_majors.map((major, index) => {
              const contextInfo = result.retrieved_context?.find(ctx => ctx.name === major)
              const isExpanded = expandedMajor === index

              return (
                <div key={index} className={`major-detail-card ${isExpanded ? 'expanded' : ''}`}>
                  <div className="major-header" onClick={() => toggleMajor(index)}>
                    <div className="major-title">
                      <span className="major-number">{index + 1}</span>
                      <h4>{major}</h4>
                    </div>
                    <button className="expand-icon">
                      {isExpanded ? '−' : '+'}
                    </button>
                  </div>

                  {isExpanded && contextInfo && (
                    <div className="major-details">
                      {contextInfo.description && (
                        <div className="detail-section">
                          <h5>📝 학과 소개</h5>
                          <p>{contextInfo.description}</p>
                        </div>
                      )}

                      {contextInfo.keywords && contextInfo.keywords.length > 0 && (
                        <div className="detail-section">
                          <h5>🔑 핵심 키워드</h5>
                          <div className="keywords-list">
                            {contextInfo.keywords.map((keyword, i) => (
                              <span key={i} className="keyword-badge">{keyword}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {contextInfo.career_paths && contextInfo.career_paths.length > 0 && (
                        <div className="detail-section">
                          <h5>💼 진로 경로</h5>
                          <ul className="career-list">
                            {contextInfo.career_paths.map((career, i) => (
                              <li key={i}>{career}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {contextInfo.related_subjects && contextInfo.related_subjects.length > 0 && (
                        <div className="detail-section">
                          <h5>📚 관련 과목</h5>
                          <div className="subjects-list">
                            {contextInfo.related_subjects.map((subject, i) => (
                              <span key={i} className="subject-tag">{subject}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {contextInfo.required_skills && contextInfo.required_skills.length > 0 && (
                        <div className="detail-section">
                          <h5>⚡ 필요 역량</h5>
                          <div className="skills-grid">
                            {contextInfo.required_skills.map((skill, i) => (
                              <div key={i} className="skill-item">
                                <span className="skill-icon">✓</span>
                                <span>{skill}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {contextInfo.university && (
                        <div className="detail-section">
                          <h5>🏫 대학 정보</h5>
                          <div className="university-info">
                            <p><strong>대학명:</strong> {contextInfo.university}</p>
                            {contextInfo.department_url && (
                              <a
                                href={contextInfo.department_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="department-link"
                              >
                                학과 홈페이지 방문 →
                              </a>
                            )}
                          </div>
                        </div>
                      )}

                      {contextInfo.score !== undefined && (
                        <div className="detail-section">
                          <h5>🎯 매칭 점수</h5>
                          <div className="score-bar">
                            <div
                              className="score-fill"
                              style={{ width: `${contextInfo.score * 100}%` }}
                            />
                          </div>
                          <p className="score-text">
                            {(contextInfo.score * 100).toFixed(1)}% 일치
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </section>

        {result.retrieved_context && result.retrieved_context.length > 0 && (
          <section className="all-context-section">
            <h3>🔍 전체 검색 결과</h3>
            <p className="section-description">
              Vector DB에서 검색된 모든 학과 정보입니다. AI가 이 정보를 바탕으로 추천을 생성했습니다.
            </p>
            <div className="all-context-grid">
              {result.retrieved_context.map((ctx, index) => (
                <div key={index} className="context-mini-card">
                  <div className="context-mini-header">
                    <h5>{ctx.name}</h5>
                    {ctx.score !== undefined && (
                      <span className="mini-score">{(ctx.score * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  <p className="context-mini-description">
                    {ctx.description?.substring(0, 100)}
                    {ctx.description?.length > 100 ? '...' : ''}
                  </p>
                  {ctx.category && (
                    <span className="category-badge">{ctx.category}</span>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="detail-actions">
          <button onClick={onBack} className="primary-action-button">
            새로운 추천 받기
          </button>
        </div>
      </div>
    </div>
  )
}

export default RecommendationDetail
