import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [interests, setInterests] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!interests.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post('/api/recommend', {
        interests: interests.trim()
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || '추천 생성에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="card">
        <div className="header">
          <h1>🎓 PathFinder</h1>
          <p className="subtitle">AI 기반 대학 학과 추천 서비스</p>
        </div>

        <form onSubmit={handleSubmit} className="form">
          <label htmlFor="interests">
            관심 분야를 입력해주세요
          </label>
          <textarea
            id="interests"
            value={interests}
            onChange={(e) => setInterests(e.target.value)}
            placeholder="예: 프로그래밍, 게임 개발, AI, 수학"
            rows="4"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !interests.trim()}>
            {loading ? '추천 생성 중...' : '학과 추천 받기'}
          </button>
        </form>

        {error && (
          <div className="error">
            <p>⚠️ {error}</p>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>AI가 분석 중입니다...</p>
          </div>
        )}

        {result && (
          <div className="result">
            <h2>✨ 추천 학과</h2>
            <div className="majors-list">
              {result.recommended_majors.map((major, index) => (
                <div key={index} className="major-card">
                  <span className="major-number">{index + 1}</span>
                  <span className="major-name">{major}</span>
                </div>
              ))}
            </div>

            <div className="reasoning">
              <h3>📝 추천 이유</h3>
              <p>{result.reasoning}</p>
            </div>

            {result.retrieved_context && (
              <details className="context">
                <summary>🔍 검색된 학과 정보 보기</summary>
                <div className="context-list">
                  {result.retrieved_context.map((ctx, index) => (
                    <div key={index} className="context-item">
                      <h4>{ctx.name}</h4>
                      <p className="description">{ctx.description}</p>
                      <div className="keywords">
                        {ctx.keywords?.slice(0, 5).map((keyword, i) => (
                          <span key={i} className="keyword">{keyword}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}

            <button
              onClick={() => {
                setResult(null)
                setInterests('')
              }}
              className="reset-button"
            >
              다시 추천받기
            </button>
          </div>
        )}
      </div>

      <footer>
        <p>Powered by LangChain, EXAONE-3.5, Qdrant</p>
      </footer>
    </div>
  )
}

export default App
