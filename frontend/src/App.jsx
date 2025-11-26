import { useState } from 'react'
import axios from 'axios'
import ChatInterface from './components/ChatInterface'
import RecommendationDetail from './components/RecommendationDetail'
import PDFAnalysis from './components/PDFAnalysis'
import './App.css'

function App() {
  const [mode, setMode] = useState('select') // 'select', 'quick', 'chat', 'pdf', 'detail'
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

  const handleViewDetail = () => {
    setMode('detail')
  }

  const handleBackFromDetail = () => {
    setResult(null)
    setInterests('')
    setMode('quick')
  }

  const handleBackToSelect = () => {
    setMode('select')
    setResult(null)
    setInterests('')
  }

  // 모드 선택 화면
  if (mode === 'select') {
    return (
      <div className="container">
        <div className="card">
          <div className="header">
            <h1>🎓 PathFinder</h1>
            <p className="subtitle">AI 기반 대학 학과 추천 서비스</p>
          </div>

          <div className="mode-selection">
            <h2>추천 방식을 선택해주세요</h2>
            <div className="mode-cards">
              <div className="mode-card" onClick={() => setMode('pdf')}>
                <div className="mode-icon">📄</div>
                <h3>생활기록부 분석</h3>
                <p>학교생활기록부 PDF를 업로드하여 자동 분석</p>
                <span className="mode-time">2-3분 소요</span>
              </div>

              <div className="mode-card" onClick={() => setMode('quick')}>
                <div className="mode-icon">⚡</div>
                <h3>빠른 추천</h3>
                <p>관심사를 입력하면 즉시 학과를 추천받습니다</p>
                <span className="mode-time">1-2분 소요</span>
              </div>

              <div className="mode-card" onClick={() => setMode('chat')}>
                <div className="mode-icon">💬</div>
                <h3>대화형 상담</h3>
                <p>AI 상담사와 대화하며 맞춤형 추천을 받습니다</p>
                <span className="mode-time">3-5분 소요</span>
              </div>
            </div>
          </div>

          <footer>
            <p>Powered by LangChain, EXAONE-3.5, Qdrant</p>
          </footer>
        </div>
      </div>
    )
  }

  // PDF 분석 화면
  if (mode === 'pdf') {
    return (
      <div className="container">
        <button onClick={handleBackToSelect} className="back-to-select">
          ← 돌아가기
        </button>
        <PDFAnalysis />
        <footer>
          <p>Powered by LangChain, EXAONE-3.5, Qdrant</p>
        </footer>
      </div>
    )
  }

  // 대화형 상담 화면
  if (mode === 'chat') {
    return (
      <div className="container">
        <button onClick={handleBackToSelect} className="back-to-select">
          ← 돌아가기
        </button>
        <ChatInterface />
        <footer>
          <p>Powered by LangChain, EXAONE-3.5, Qdrant</p>
        </footer>
      </div>
    )
  }

  // 상세 결과 화면
  if (mode === 'detail') {
    return (
      <div className="container">
        <RecommendationDetail result={result} onBack={handleBackFromDetail} />
        <footer>
          <p>Powered by LangChain, EXAONE-3.5, Qdrant</p>
        </footer>
      </div>
    )
  }

  // 빠른 추천 화면 (기존)
  return (
    <div className="container">
      <button onClick={handleBackToSelect} className="back-to-select">
        ← 돌아가기
      </button>
      <div className="card">
        <div className="header">
          <h1>🎓 PathFinder</h1>
          <p className="subtitle">빠른 학과 추천</p>
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

            <div className="result-actions">
              <button onClick={handleViewDetail} className="detail-button">
                상세 결과 보기
              </button>
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
