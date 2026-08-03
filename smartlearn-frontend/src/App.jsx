import { useRef, useState } from "react";
import { uploadPDF } from "./api.js";
import ChatPanel from "./ChatPanel.jsx";
import PdfPreview from "./PdfPreview.jsx";

function App() {
  const [upload, setUpload] = useState(null);
  const [uploadKey, setUploadKey] = useState(0);
  const [activePage, setActivePage] = useState(1);
  const [status, setStatus] = useState("idle");
  const [chatBusy, setChatBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const busy = status !== "idle" || chatBusy;

  async function uploadNow(file) {
    try {
      setStatus("uploading");
      setError("");
      const result = await uploadPDF(file);
      setUpload(result);
      setActivePage(1);
      setUploadKey((key) => key + 1); // remount ChatPanel so old messages clear
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setStatus("idle");
    }
  }

  function acceptFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are accepted.");
      return;
    }
    setError("");
    void uploadNow(file);
  }

  function handleFileInput(event) {
    acceptFile(event.target.files?.[0]);
    event.target.value = ""; // allow re-selecting the same file
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragOver(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function handleJumpToPage(page) {
    setActivePage(page);
  }

  return (
    <main>
      <header className="app-header">
        <div className="app-logo">📚</div>
        <div>
          <h1>SmartLearn Lite</h1>
          <p className="app-subtitle">RAG answers with page citations — drop a PDF to start</p>
        </div>
      </header>

      <div
        className={`drop-zone${dragOver ? " drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input ref={inputRef} type="file" accept=".pdf" hidden onChange={handleFileInput} />
        {status === "uploading" ? (
          <div className="drop-zone-inner">
            <span className="drop-icon spinning">⏳</span>
            <p className="file-name">Processing PDF…</p>
            <p className="file-hint">Chunking, embedding and building the index</p>
          </div>
        ) : upload ? (
          <div className="drop-zone-inner">
            <span className="drop-icon">✅</span>
            <p className="file-name">{upload.filename}</p>
            <p className="file-meta">
              {upload.pages} pages · {upload.characters.toLocaleString()} characters
            </p>
            <p className="file-hint">Drop a new PDF here to replace it</p>
          </div>
        ) : (
          <div className="drop-zone-inner">
            <span className="drop-icon">📄</span>
            <p className="file-name">Drop a PDF here, or click to browse</p>
            <p className="file-hint">PDF only · page-cited answers</p>
          </div>
        )}
      </div>

      {error && (
        <p className="app-error" role="alert">{error}</p>
      )}

      {upload ? (
        <div className="workspace">
          <PdfPreview
            upload={upload}
            activePage={activePage}
            previewKey={uploadKey}
          />
          <ChatPanel
            key={uploadKey}
            enabled={!!upload}
            onBusy={setChatBusy}
            disabled={busy}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      ) : (
        <p className="app-empty">Upload a PDF to start.</p>
      )}
    </main>
  );
}

export default App;
