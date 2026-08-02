import { useState } from "react";
import { askQuestion, uploadPDF } from "./api.js";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const busy = status !== "idle";

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;
    try {
      setStatus("uploading");
      setError("");
      const result = await uploadPDF(file);
      setUpload(result);
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setStatus("idle");
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    if (!message.trim()) return;
    try {
      setStatus("asking");
      setError("");
      const result = await askQuestion(message.trim());
      setAnswer(result);
    } catch (err) {
      setError(err.message || "Chat failed.");
    } finally {
      setStatus("idle");
    }
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      {/* ── Upload section ── */}
      <form onSubmit={handleUpload}>
        <label htmlFor="pdf-file">PDF file</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={!file || busy}>
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </form>

      {upload && (
        <p>
          Uploaded: {upload.filename} ({upload.pages} pages, {upload.characters}{" "}
          characters)
        </p>
      )}

      {/* ── Chat section ── */}
      {upload && (
        <form onSubmit={handleAsk}>
          <label htmlFor="message">Message</label>
          <textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <button
            type="submit"
            disabled={!message.trim() || busy}
          >
            {status === "asking" ? "Asking…" : "Ask"}
          </button>
        </form>
      )}

      {/* ── Shared feedback ── */}
      {error && <p role="alert">{error}</p>}

      {answer && (
        <section>
          <p>{answer.answer}</p>
          <div>
            {answer.citations.map((page) => (
              <span key={page}>Page {page}</span>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

export default App;
