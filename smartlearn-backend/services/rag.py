import json
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text."""
    text = text.replace("\x00", "")
    text = re.sub(r"­", "", text)  # soft hyphens
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", text)  # noisy line breaks
    return text.strip()


def extract_pages_for_rag(pdf_path) -> list[dict]:
    """Read a PDF page by page, preserving original page numbers."""
    reader = PdfReader(str(pdf_path))
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


def save_json(obj, path, indent: int = 2) -> Path:
    """Save a Python object to a UTF-8 JSON file, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
    return path


def load_json(path):
    """Read a saved JSON artifact back into Python."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def preview_records(records: list[dict], columns: list[str], rows: int = 5):
    """Show a small notebook table for the chosen columns."""
    import pandas as pd

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable = [column for column in columns if column in frame.columns]
    return frame[usable].head(rows)


def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split one oversized block into smaller pieces, preferring natural boundaries."""
    if len(text) <= chunk_size:
        return [text]
    pieces = []
    while len(text) > chunk_size:
        cut = text.rfind(" ", 0, chunk_size + 1)
        if cut <= 0:
            cut = chunk_size
        pieces.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text.strip():
        pieces.append(text.strip())
    return pieces


def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
    """Convert paragraph-level records into chunks, preserving page order."""
    chunks = []
    next_id = 1
    for record in records:
        page = record["page"]
        paragraphs = [p.strip() for p in record["text"].split("\n\n") if p.strip()]
        current = ""
        current_page = page
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > chunk_size:
                for piece in slice_long_text(current, chunk_size):
                    chunks.append(
                        {
                            "chunk_id": next_id,
                            "page": current_page,
                            "chunk_mode": "paragraph",
                            "text": piece,
                        }
                    )
                    next_id += 1
                current = ""
            current = paragraph
        if current:
            for piece in slice_long_text(current, chunk_size):
                chunks.append(
                    {
                        "chunk_id": next_id,
                        "page": current_page,
                        "chunk_mode": "paragraph",
                        "text": piece,
                    }
                )
                next_id += 1
    return chunks


def chunk_by_characters(
    records: list[dict], chunk_size: int, overlap: int = 0
) -> list[dict]:
    """Create plain fixed-size sliding-window chunks with optional overlap."""
    chunks = []
    next_id = 1
    step = chunk_size - overlap
    for record in records:
        text = record["text"]
        page = record["page"]
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    {
                        "chunk_id": next_id,
                        "page": page,
                        "chunk_mode": "character_overlap" if overlap else "character",
                        "text": piece,
                    }
                )
                next_id += 1
            if step <= 0:
                break
            start += step
    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Select the requested chunking strategy and return a uniform chunk schema."""
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)
    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)
    if chunk_mode == "character_overlap":
        return chunk_by_characters(records, chunk_size, overlap=overlap)
    raise ValueError(
        f"Unknown chunk_mode={chunk_mode!r}. "
        "Choose 'paragraph', 'character', or 'character_overlap'."
    )


def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix for saved artifacts."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model_name).replace(".", "_")


def resolve_model_source(model_name: str, artifact_root=None) -> str:
    """Prefer a local cached model folder when it already exists."""
    if artifact_root is not None:
        candidates = [
            Path(artifact_root) / "hf_models" / "all-MiniLM-L6-v2",
            Path(artifact_root) / "hf_models" / model_name.replace("/", "_"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return model_name


def get_device() -> str:
    """Choose CPU or CUDA for the current machine."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(model_name: str, artifact_root=None):
    """Create or reuse one sentence-transformer model instance."""
    from sentence_transformers import SentenceTransformer

    source = resolve_model_source(model_name, artifact_root)
    return SentenceTransformer(
        source,
        device=get_device(),
        model_kwargs={"use_safetensors": False},
    )


def embed_texts(
    model,
    texts: list[str],
    batch_size: int = 32,
    device: str | None = None,
) -> "np.ndarray":
    """Encode a list of texts into normalized float32 vectors."""
    import numpy as np

    device = device or get_device()
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )


def artifact_paths_for(
    document_id: str,
    pdf_name: str,
    chunk_mode: str,
    model_name: str,
    artifact_root,
) -> dict:
    """Decide where pages, chunks, embeddings, and manifests should be saved."""
    root = Path(artifact_root) / document_id
    safe_model = model_tag(model_name)
    return {
        "raw_pages_path": root / f"{document_id}_pages.json",
        "chunk_path": root / f"{document_id}_{chunk_mode}.json",
        "embedding_path": root / f"{document_id}_{chunk_mode}_{safe_model}.npy",
        "manifest_path": root / f"{document_id}_{chunk_mode}_{safe_model}.manifest.json",
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str,
    model_name: str,
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Build or reuse the full pages -> chunks -> embeddings -> manifest bundle."""
    import numpy as np

    if artifact_root is None:
        raise ValueError("artifact_root is required to persist RAG artifacts")

    paths = artifact_paths_for(
        document_id=document_id,
        pdf_name=pdf_name,
        chunk_mode=chunk_mode,
        model_name=model_name,
        artifact_root=artifact_root,
    )

    signature = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
    }

    if paths["manifest_path"].exists():
        manifest = load_json(paths["manifest_path"])
        if all(manifest.get(k) == v for k, v in signature.items()):
            chunks = load_json(paths["chunk_path"])
            embeddings = np.load(paths["embedding_path"])
            return {"chunks": chunks, "embeddings": embeddings, "manifest": manifest}

    chunks = build_chunks(
        pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    model = load_model(model_name, artifact_root)
    device = get_device()
    embeddings = embed_texts(
        model,
        [c["text"] for c in chunks],
        batch_size=batch_size,
        device=device,
    )

    save_json(pages, paths["raw_pages_path"])
    save_json(chunks, paths["chunk_path"])
    np.save(paths["embedding_path"], embeddings)

    manifest = {
        **signature,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "device": device,
        "chunk_path": str(paths["chunk_path"]),
        "embedding_path": str(paths["embedding_path"]),
        "raw_pages_path": str(paths["raw_pages_path"]),
    }
    save_json(manifest, paths["manifest_path"])

    return {"chunks": chunks, "embeddings": embeddings, "manifest": manifest}
