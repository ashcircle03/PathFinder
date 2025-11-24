import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './ChatInterface.css'

function ChatInterface() {
  const [sessionId] = useState(() => `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // 초기 환영 메시지
    setMessages([
      {
        role: 'assistant',
        content: '안녕하세요! PathFinder 진로 상담사입니다. 대학 학과 추천을 도와드리겠습니다. 먼저 어떤 분야에 관심이 있으신가요?',
        timestamp: new Date()
      }
    ])
  }, [])

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!inputMessage.trim() || loading) return

    const userMessage = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/chat', {
        message: userMessage.content,
        session_id: sessionId
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
        isReadyToRecommend: response.data.is_ready_to_recommend,
        conversationCount: response.data.conversation_count,
        collectedInfo: response.data.collected_info
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      setError(err.response?.data?.detail || '메시지 전송에 실패했습니다.')
      setMessages(prev => [...prev, {
        role: 'system',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.',
        timestamp: new Date()
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleGetRecommendation = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`/api/chat/${sessionId}/recommend`)
      setRecommendation(response.data)

      setMessages(prev => [...prev, {
        role: 'system',
        content: '학과 추천이 완료되었습니다! 아래에서 결과를 확인하세요.',
        timestamp: new Date()
      }])
    } catch (err) {
      setError(err.response?.data?.detail || '추천 생성에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = async () => {
    try {
      await axios.delete(`/api/chat/${sessionId}`)
    } catch (err) {
      console.error('세션 삭제 실패:', err)
    }

    window.location.reload()
  }

  const lastMessage = messages[messages.length - 1]
  const canRecommend = lastMessage?.role === 'assistant' && lastMessage?.isReadyToRecommend

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>💬 대화형 진로 상담</h2>
        <p className="chat-subtitle">자연스러운 대화로 맞춤 학과를 찾아보세요</p>
      </div>

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message message-${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : msg.role === 'assistant' ? '🎓' : 'ℹ️'}
            </div>
            <div className="message-content">
              <div className="message-text">{msg.content}</div>
              <div className="message-time">
                {msg.timestamp.toLocaleTimeString('ko-KR', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </div>
              {msg.collectedInfo && (
                <div className="collected-info">
                  <details>
                    <summary>수집된 정보 보기 ({msg.conversationCount}회 대화)</summary>
                    <div className="info-content">
                      {msg.collectedInfo.interests?.length > 0 && (
                        <div>
                          <strong>관심사:</strong> {msg.collectedInfo.interests.join(', ')}
                        </div>
                      )}
                      {msg.collectedInfo.subjects?.length > 0 && (
                        <div>
                          <strong>좋아하는 과목:</strong> {msg.collectedInfo.subjects.join(', ')}
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message message-assistant">
            <div className="message-avatar">🎓</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div className="chat-error">
          ⚠️ {error}
        </div>
      )}

      {canRecommend && !recommendation && (
        <div className="recommend-prompt">
          <p>충분한 정보가 모였습니다! 학과 추천을 받아보시겠어요?</p>
          <button
            onClick={handleGetRecommendation}
            disabled={loading}
            className="recommend-button"
          >
            학과 추천 받기
          </button>
        </div>
      )}

      {recommendation && (
        <div className="recommendation-result">
          <h3>✨ 추천 학과</h3>
          <div className="majors-grid">
            {recommendation.recommended_majors.map((major, index) => (
              <div key={index} className="major-item">
                <span className="major-rank">{index + 1}</span>
                <span className="major-name">{major}</span>
              </div>
            ))}
          </div>

          <div className="reasoning-box">
            <h4>📝 추천 이유</h4>
            <p>{recommendation.reasoning}</p>
          </div>

          {recommendation.retrieved_context && (
            <details className="context-details">
              <summary>🔍 검색된 학과 정보 ({recommendation.retrieved_context.length}개)</summary>
              <div className="context-grid">
                {recommendation.retrieved_context.map((ctx, index) => (
                  <div key={index} className="context-card">
                    <h5>{ctx.name}</h5>
                    <p className="context-description">{ctx.description}</p>
                    {ctx.keywords && (
                      <div className="context-keywords">
                        {ctx.keywords.slice(0, 5).map((kw, i) => (
                          <span key={i} className="keyword-tag">{kw}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )}

          <button onClick={handleReset} className="reset-chat-button">
            새로운 상담 시작하기
          </button>
        </div>
      )}

      <form onSubmit={handleSendMessage} className="chat-input-form">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="메시지를 입력하세요..."
          disabled={loading || recommendation}
          className="chat-input"
        />
        <button
          type="submit"
          disabled={loading || !inputMessage.trim() || recommendation}
          className="send-button"
        >
          {loading ? '⏳' : '📤'}
        </button>
      </form>
    </div>
  )
}

export default ChatInterface
