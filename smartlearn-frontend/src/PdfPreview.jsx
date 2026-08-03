import { API, CHAT_ID } from "./api.js";

export function getDocumentFileURL(page = 1) {
  return `${API}/documents/${encodeURIComponent(CHAT_ID)}/file#page=${page}`;
}

function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return <div className="pdf-preview placeholder">No PDF loaded yet.</div>;
  }
  return (
    <section className="pdf-preview">
      <header className="pdf-preview-header">
        <span>{upload.filename}</span>
        <span>Page {activePage}</span>
      </header>
      <iframe
        key={`${previewKey}-${activePage}`}
        title="PDF preview"
        src={getDocumentFileURL(activePage)}
        className="pdf-frame"
      />
    </section>
  );
}

export default PdfPreview;
