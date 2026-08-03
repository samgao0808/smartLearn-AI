import { useState } from "react";
import { askQuestion } from "./api.js";

function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk(event) {
    event.preventDefault();
    const text = message.trim();
    if (!text || loading || !enabled) return;
    setLoading(true);
    setError("");
    onBusy?.(true);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    try {
      const result = await askQuestion(text);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          citations: result.citations || [],
          sources: result.sources || [],
        },
      ]);
      setMessage("");
    } catch (err) {
      setError(err.message || "Chat failed.");
    } finally {
      setLoading(false);
      onBusy?.(false);
    }
  }

  return (
    <section className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <p className="chat-empty">Ask a question about the uploaded PDF.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-label">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="message-content">{msg.content}</div>
            {msg.role === "assistant" && msg.citations.length > 0 && (
              <div className="citations">
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    type="button"
                    className="citation-btn"
                    onClick={() => onJumpToPage?.(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <p className="chat-loading">Asking…</p>}
        {error && <p className="chat-error" role="alert">{error}</p>}
      </div>
      <form onSubmit={handleAsk} className="chat-form">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about the uploaded PDF…"
        />
        <button
          type="submit"
          disabled={!message.trim() || loading || disabled || !enabled}
        >
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>
    </section>
  );
}

export default ChatPanel;
