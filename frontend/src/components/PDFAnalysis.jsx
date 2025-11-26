import { useState } from 'react'
import axios from 'axios'
import './PDFAnalysis.css'

function PDFAnalysis() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [analysisPhase, setAnalysisPhase] = useState('') // 'uploading', 'analyzing', 'recommending'

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('PDF 파일만 업로드 가능합니다.')
      setFile(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)
    setAnalysisPhase('uploading')

    const formData = new FormData()
    formData.append('file', file)

    try {
      setAnalysisPhase('analyzing')
      const response = await axios.post('/api/analyze-and-recommend', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 300000 // 5분 타임아웃 (OCR 시간 고려)
      })

      setResult(response.data)
      setAnalysisPhase('')
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || '분석에 실패했습니다.'
      setError(errorMessage)
      setAnalysisPhase('')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setAnalysisPhase('')
  }

  return (
    <div className="pdf-analysis">
      <div className="pdf-header">
        <h2>📄 학교생활기록부 분석</h2>
        <p className="pdf-subtitle">PDF를 업로드하면 자동으로 프로필을 분석하고 학과를 추천합니다</p>
      </div>

      {!result && (
        <form onSubmit={handleSubmit} className="pdf-upload-form">
          <div className="file-upload-area">
            <input
              type="file"
              id="pdf-file"
              accept=".pdf"
              onChange={handleFileChange}
              disabled={loading}
              className="file-input"
            />
            <label htmlFor="pdf-file" className="file-label">
              <div className="upload-icon">📁</div>
              <div className="upload-text">
                {file ? (
                  <>
                    <strong>{file.name}</strong>
                    <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                  </>
                ) : (
                  <>
                    <strong>PDF 파일을 선택하세요</strong>
                    <span>또는 파일을 드래그 앤 드롭하세요</span>
                  </>
                )}
              </div>
            </label>
          </div>

          {file && !loading && (
            <button type="submit" className="analyze-button">
              분석 및 추천 시작
            </button>
          )}
        </form>
      )}

      {error && (
        <div className="pdf-error">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <strong>오류가 발생했습니다</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="analysis-progress">
          <div className="progress-spinner"></div>
          <div className="progress-text">
            {analysisPhase === 'uploading' && '📤 파일 업로드 중...'}
            {analysisPhase === 'analyzing' && (
              <>
                <div>🔍 학교생활기록부 분석 중...</div>
                <div className="progress-sub">OCR 처리 및 AI 분석이 진행됩니다 (1-2분 소요)</div>
              </>
            )}
            {analysisPhase === 'recommending' && '🎓 학과 추천 생성 중...'}
          </div>
        </div>
      )}

      {result && (
        <div className="pdf-result">
          <div className="analysis-summary">
            <h3>✅ 프로필 분석 완료</h3>

            {result.academic_strengths?.length > 0 && (
              <div className="profile-section">
                <h4>📚 학업 강점</h4>
                <div className="tags">
                  {result.academic_strengths.map((item, i) => (
                    <span key={i} className="tag tag-blue">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {result.extracurricular_activities?.length > 0 && (
              <div className="profile-section">
                <h4>🎯 주요 활동</h4>
                <div className="tags">
                  {result.extracurricular_activities.slice(0, 5).map((item, i) => (
                    <span key={i} className="tag tag-green">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {result.career_interests?.length > 0 && (
              <div className="profile-section">
                <h4>💼 진로 관심사</h4>
                <div className="tags">
                  {result.career_interests.map((item, i) => (
                    <span key={i} className="tag tag-purple">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {result.personality_traits?.length > 0 && (
              <div className="profile-section">
                <h4>⭐ 성격 및 강점</h4>
                <div className="tags">
                  {result.personality_traits.map((item, i) => (
                    <span key={i} className="tag tag-orange">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {result.recommended_interests_summary && (
              <div className="profile-section summary">
                <h4>📝 종합 분석</h4>
                <p>{result.recommended_interests_summary}</p>
              </div>
            )}
          </div>

          <div className="recommendation-section">
            <h3>🎓 추천 학과</h3>
            <div className="majors-grid">
              {result.recommended_majors?.map((major, index) => (
                <div key={index} className="major-card-pdf">
                  <span className="major-rank">{index + 1}</span>
                  <span className="major-name">{major}</span>
                </div>
              ))}
            </div>

            <div className="reasoning-section">
              <h4>💡 추천 이유</h4>
              <p>{result.reasoning}</p>
            </div>

            {result.retrieved_context && (
              <details className="context-section">
                <summary>🔍 참고한 학과 정보 ({result.retrieved_context.length}개)</summary>
                <div className="context-list">
                  {result.retrieved_context.map((ctx, index) => (
                    <div key={index} className="context-item">
                      <h5>{ctx.name}</h5>
                      <p>{ctx.description}</p>
                      {ctx.keywords && (
                        <div className="context-keywords">
                          {ctx.keywords.slice(0, 5).map((kw, i) => (
                            <span key={i} className="keyword">{kw}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>

          <button onClick={handleReset} className="reset-button-pdf">
            다시 분석하기
          </button>
        </div>
      )}
    </div>
  )
}

export default PDFAnalysis
