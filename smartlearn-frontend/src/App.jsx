import { useState } from "react";
import { uploadPDF } from "./api.js";
import ChatPanel from "./ChatPanel.jsx";
import PdfPreview from "./PdfPreview.jsx";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [uploadKey, setUploadKey] = useState(0);
  const [activePage, setActivePage] = useState(1);
  const [status, setStatus] = useState("idle");
  const [chatBusy, setChatBusy] = useState(false);
  const [error, setError] = useState("");

  const busy = status !== "idle" || chatBusy;

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;
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

  function handleJumpToPage(page) {
    setActivePage(page);
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      <form onSubmit={handleUpload} className="upload-form">
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

      {error && <p className="app-error" role="alert">{error}</p>}

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
