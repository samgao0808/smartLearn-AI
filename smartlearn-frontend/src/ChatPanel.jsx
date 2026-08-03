import { useState } from "react";
import { askQuestionStream } from "./api.js";

function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateAssistant(id, patch) {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg)),
    );
  }

  async function handleAsk(event) {
    event.preventDefault();
    const text = message.trim();
    if (!text || loading || !enabled) return;
    setLoading(true);
    setError("");
    onBusy?.(true);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setMessage("");
    const assistantId = `a${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        sources: [],
        streaming: true,
      },
    ]);
    try {
      let full = "";
      await askQuestionStream(text, {
        onMeta: (meta) =>
          updateAssistant(assistantId, {
            citations: meta.citations || [],
            sources: meta.sources || [],
          }),
        onDelta: (delta) => {
          full += delta;
          updateAssistant(assistantId, { content: full });
        },
      });
      updateAssistant(assistantId, { streaming: false });
    } catch (err) {
      updateAssistant(assistantId, { streaming: false });
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
        {messages.map((msg) => (
          <div key={msg.id || msg.content || msg.role} className={`message ${msg.role}`}>
            <div className="message-label">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="message-content">
              {msg.content}
              {msg.streaming && <span className="stream-cursor" aria-hidden="true" />}
            </div>
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
        {loading && !messages.some((m) => m.streaming) && (
          <p className="chat-loading">Asking…</p>
        )}
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
