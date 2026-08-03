import io
import json
import os
import re
from pathlib import Path

import faiss
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text."""
    text = text.replace("\x00", "")
    text = re.sub(r"­", "", text)  # soft hyphens
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", text)  # noisy line breaks
    return text.strip()


def extract_pages_for_rag(pdf_path, page_limit: int | None = None) -> list[dict]:
    """Read a PDF page by page, preserving original page numbers."""
    reader = PdfReader(str(pdf_path))
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_limit is not None and page_number > page_limit:
            break
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Read page records from raw uploaded PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
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


def relative_path_str(path, base) -> str:
    """Return a shorter display path for one artifact relative to a base folder."""
    return os.path.relpath(str(path), str(base))


def build_faiss_index(embeddings) -> "faiss.Index":
    """Build a searchable FAISS inner-product index from normalized vectors."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path) -> None:
    """Write a FAISS index to a binary .faiss file."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path) -> "faiss.Index":
    """Load a saved FAISS index back into memory."""
    return faiss.read_index(str(index_path))


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path=None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Build or load a reusable FAISS index bundle for one PDF."""
    if artifact_root is None:
        raise ValueError("artifact_root is required to persist RAG artifacts")

    if pages is None:
        if pdf_path is None:
            raise ValueError("Either pages or pdf_path must be provided")
        pages = extract_pages_for_rag(pdf_path)

    bundle = ensure_artifacts(
        document_id=document_id,
        pdf_name=pdf_name,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    tag = model_tag(model_name)
    index_dir = (
        Path(artifact_root)
        / document_id
        / f"{chunk_mode}_c{chunk_size}_o{overlap}_{tag}"
    )
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "index.meta.json"

    signature = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_chunks": len(bundle["chunks"]),
        "embedding_dim": bundle["embeddings"].shape[1],
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
    }

    if index_path.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if all(meta.get(k) == v for k, v in signature.items()):
            return {
                "chunks": bundle["chunks"],
                "embeddings": bundle["embeddings"],
                "manifest": bundle["manifest"],
                "index": load_faiss_index(index_path),
                "paths": {
                    "index": index_path,
                    "chunks": Path(bundle["manifest"]["chunk_path"]),
                    "embeddings": Path(bundle["manifest"]["embedding_path"]),
                },
            }

    index = build_faiss_index(bundle["embeddings"])
    save_faiss_index(index, index_path)
    save_json(signature, meta_path)

    return {
        "chunks": bundle["chunks"],
        "embeddings": bundle["embeddings"],
        "manifest": bundle["manifest"],
        "index": index,
        "paths": {
            "index": index_path,
            "chunks": Path(bundle["manifest"]["chunk_path"]),
            "embeddings": Path(bundle["manifest"]["embedding_path"]),
        },
    }


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Return a server-style document record with pages, chunks, and index paths."""
    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "chunk_size": len(bundle["chunks"]),
        "num_chunks": len(bundle["chunks"]),
        "embedding_dim": bundle["embeddings"].shape[1],
        "model_name": model_name,
        "model_source": resolve_model_source(model_name, artifact_root),
        "history": [],
        "artifacts": {
            "index": bundle["paths"]["index"],
            "chunks": bundle["paths"]["chunks"],
            "embeddings": bundle["paths"]["embeddings"],
        },
    }


def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: bytes | None = None,
    pages: list[dict] | None = None,
    upload_root=None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Build one upload-time Day 3 record for documents[chat_id]."""
    if pages is None:
        if pdf_bytes is None:
            raise ValueError("Either pdf_bytes or pages must be provided")
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)

    document = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    saved_pdf_path = None
    if upload_root is not None and pdf_bytes is not None:
        upload_root = Path(upload_root)
        upload_root.mkdir(parents=True, exist_ok=True)
        saved_pdf_path = upload_root / f"{chat_id}.pdf"
        with open(saved_pdf_path, "wb") as fh:
            fh.write(pdf_bytes)

    file_path = str(saved_pdf_path) if saved_pdf_path is not None else filename
    document["chat_id"] = chat_id
    document["file_path"] = file_path
    document["saved_pdf_path"] = file_path
    document["rag"] = {
        "document_id": chat_id,
        "index_path": str(document["artifacts"]["index"]),
        "model_name": model_name,
    }
    return document


def build_upload_response(document: dict) -> dict:
    """Return the visible Day 2 upload JSON from a richer Day 3 record."""
    pages = document.get("pages", [])
    characters = sum(len(p["text"]) for p in pages)
    return {
        "status": "ok",
        "filename": document.get("filename", "document.pdf"),
        "pages": len(pages),
        "characters": characters,
    }


def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
    """Return numeric PDF page citations from an answer or its hits."""
    numbers = [int(n) for n in re.findall(r"\d+", answer)]
    if hits:
        valid_pages = {int(h["page"]) for h in hits}
        numbers = [n for n in numbers if n in valid_pages]
    return sorted(set(numbers))


def build_sources(hits: list[dict]) -> list[dict]:
    """Return frontend-friendly source objects with page, chunk id, score, and preview."""
    return [
        {
            "page": int(h["page"]),
            "chunk_id": h["chunk_id"],
            "score": round(float(h["score"]), 4),
            "preview": h["text"][:200],
        }
        for h in hits
    ]


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
    hits: list[dict] | None = None,
) -> dict:
    """Retrieve evidence and answer with the LLM when a key exists, else local."""
    if hits is None:
        hits = search_document(
            question,
            document,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

    answer = None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            from services.llm import answer_from_pages

            evidence_pages = [{"page": h["page"], "text": h["text"]} for h in hits]
            answer = answer_from_pages(evidence_pages, question)
        except Exception:
            answer = None

    if answer is None:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)
    return {"answer": answer, "citations": citations, "sources": sources}


def append_history(document: dict, question: str, result: dict) -> list[dict]:
    """Append one Q&A turn to the stored document's in-memory history."""
    history = document.get("history", [])
    history.append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        }
    )
    document["history"] = history
    return history


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Build one grounded prompt from recent history and retrieved evidence."""
    blocks = []
    if history:
        blocks.append("Conversation so far:")
        for turn in history[-3:]:
            blocks.append(f'User: {turn.get("question", "")}')
            blocks.append(f'Assistant: {turn.get("answer", "")}')
        blocks.append("")
    blocks.append("Relevant document excerpts:")
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] (Page {hit['page']}) {hit['text'][:500]}")
    blocks.append("")
    blocks.append(f"Question: {question}")
    blocks.append("Answer concisely from the excerpts above. Cite facts with [Page N].")
    return "\n".join(blocks)


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Answer one question and append the completed turn to in-memory history.

    Retrieval-on-demand (C2): a short follow-up reuses the previous turn's
    hits instead of running a fresh FAISS search.
    """
    hits, reused = retrieve_or_reuse(
        question,
        document,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )
    document["last_hits"] = hits
    result = answer_document(
        document,
        question,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
        hits=hits,
    )
    result["reused"] = reused
    result["history"] = append_history(document, question, result)
    return result


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Route entry point: fresh retrieval + answer + in-memory history update."""
    return answer_document_turn(
        document,
        message,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )


def answer_document_stream(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
):
    """Generator yielding SSE-friendly event dicts for a streamed answer.

    Event shapes:
        {"type": "meta", "citations": [...], "sources": [...], "reused": bool}
        {"type": "delta", "text": "..."}
        {"type": "done"}

    Citations/sources come straight from the hit pages (not parsed from the
    answer text) so the UI can render source buttons before the first token.
    Falls back to the local sentence answer if the LLM stream fails.
    """
    hits, reused = retrieve_or_reuse(
        message,
        document,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )
    document["last_hits"] = hits
    citations = sorted({int(h["page"]) for h in hits})
    sources = build_sources(hits)
    yield {
        "type": "meta",
        "citations": citations,
        "sources": sources,
        "reused": reused,
    }

    full_answer = ""
    try:
        from services.llm import stream_answer_from_pages

        evidence_pages = [{"page": h["page"], "text": h["text"]} for h in hits]
        for delta in stream_answer_from_pages(evidence_pages, message):
            full_answer += delta
            yield {"type": "delta", "text": delta}
    except Exception:
        fallback = best_sentence_answer(message, hits)
        full_answer = fallback
        yield {"type": "delta", "text": fallback}

    append_history(
        document,
        message,
        {"answer": full_answer, "citations": citations},
    )
    yield {"type": "done"}


def normalize_for_match(text: str) -> str:
    """Normalize text for simple string-based scoring."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return whether any acceptable answer appears after normalization."""
    normalized = normalize_for_match(text)
    return any(normalize_for_match(answer) in normalized for answer in answers)


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
):
    """Answer each question and summarize retrieval / answer hit per row."""
    import pandas as pd

    rows = []
    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        answers = item["answers"]
        document = documents_by_name[pdf_name]

        hits = search_document(
            question,
            document,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )
        pages = sorted({int(h["page"]) for h in hits})
        local_answer = best_sentence_answer(question, hits)

        page_text = " ".join(h["text"] for h in hits)
        retrieval_hit = contains_any_answer(page_text, answers)
        answer_hit = contains_any_answer(local_answer, answers)

        rows.append(
            {
                "pdf_name": pdf_name,
                "question": question,
                "pages": pages,
                "local_answer": local_answer,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
            }
        )
    return pd.DataFrame(rows)


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "at", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "which", "what", "how", "why", "who",
    "do", "does", "did", "it", "its", "as", "from", "we", "you", "they",
}


def keyword_set(text: str) -> set:
    """Extract lightweight lexical tokens (stopwords removed) for reranking."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


_FOLLOWUP_WORDS = {
    # English — unambiguous deictic/content cues ("more", "it", "second" ...)
    "more", "details", "detail", "elaborate", "explain", "continue",
    "second", "third", "next", "mentioned", "above", "previous", "earlier",
    "it", "this", "that", "they", "these", "those",
    # Chinese
    "继续", "更多", "详细", "细节", "第二个", "上面", "刚才", "前面", "然后", "再",
    "它", "这个", "这些", "那些", "那个",
}

# Interrogatives/connectors that only mean "follow-up" when the message
# carries NO content words at all ("why?", "and then?").
_BARE_FOLLOWUPS = {
    "why", "what", "which", "how", "who", "where", "when", "so", "then",
    "and", "or", "but",
    "为什么", "怎么",
}

# Everything that can signal a reference to prior context.
_REFERENCE_WORDS = (
    _FOLLOWUP_WORDS
    | _BARE_FOLLOWUPS
    | {"and", "or", "but", "so", "then", "too", "also"}
)


def should_reuse_last_hits(message: str, document: dict) -> bool:
    """Decide whether a follow-up can skip fresh retrieval (C2).

    True when there is prior history, the previous hits are still around,
    and the new message either contains an unambiguous deictic cue
    ("more details", "it", "second" ...) or carries no new content words
    at all ("why?", "and then?").
    """
    history = document.get("history") or []
    if not history:
        return False
    last_hits = document.get("last_hits")
    if not last_hits:
        return False
    text = message.lower()
    ascii_words = set(re.findall(r"[a-z0-9]+", text))
    q_tokens = ascii_words - _STOPWORDS
    cn_deictic = [word for word in _FOLLOWUP_WORDS if not word.isascii()]
    cn_reference = [word for word in _REFERENCE_WORDS if not word.isascii()]
    has_cn_deictic = any(word in text for word in cn_deictic)
    if q_tokens:
        # Message has content words: reuse when every content word is itself
        # a reference word ("and then?"), or any deictic cue is present
        # ("give me more details").
        if all(token in _REFERENCE_WORDS for token in q_tokens):
            return True
        return any(word in _FOLLOWUP_WORDS for word in ascii_words) or has_cn_deictic
    if ascii_words:
        # Pure reference like "why?" / "and then?" — every word must be one.
        return all(word in _REFERENCE_WORDS for word in ascii_words)
    # Pure Chinese phrase: reuse when any reference word appears anywhere
    # (covers "为什么？" from _BARE_FOLLOWUPS and "更多细节" from _FOLLOWUP_WORDS).
    return any(word in text for word in cn_reference)


def retrieve_or_reuse(
    message: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
) -> tuple[list[dict], bool]:
    """Retrieve fresh hits, or reuse the last turn's hits for a follow-up."""
    if should_reuse_last_hits(message, document):
        return list(document["last_hits"])[:top_k], True
    hits = search_document(
        message,
        document,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )
    return hits, False


def rerank_hybrid(
    question: str,
    candidates: list[dict],
    top_k: int = 3,
    dense_weight: float = 0.6,
) -> list[dict]:
    """Blend dense scores with lexical keyword overlap, then return top-k.

    Lexical overlap rewards exact content-word matches, which dense embeddings
    alone can miss for specific terms (e.g. "BM25", "HotpotQA"). The returned
    hits keep their original dense ``score`` field for ``build_sources``.
    """
    if not candidates:
        return []
    dense = [c["score"] for c in candidates]
    d_min, d_max = min(dense), max(dense)

    def norm(value: float) -> float:
        if d_max - d_min < 1e-9:
            return 1.0
        return (value - d_min) / (d_max - d_min)

    q_tokens = keyword_set(question)
    scored = []
    for candidate in candidates:
        c_tokens = keyword_set(candidate["text"])
        overlap = len(q_tokens & c_tokens)
        # Require >= 2 content-word matches: a single token (often a stray
        # conference/journal name on a references page) is pure noise.
        lexical = overlap / max(1, len(q_tokens)) if overlap >= 2 else 0.0
        combined = dense_weight * norm(candidate["score"]) + (
            1 - dense_weight
        ) * lexical
        scored.append((combined, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:top_k]]


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Retrieve a wide candidate pool, then hybrid-rerank down to top-k."""
    import numpy as np

    model_source = bundle.get("model_source") or bundle["manifest"]["model_name"]
    model = load_model(model_source)
    query = embed_texts(
        model,
        [question],
        batch_size=batch_size,
    )
    scores, ids = bundle["index"].search(query, candidate_pool)
    flat_ids = ids[0]
    flat_scores = scores[0]
    chunks = bundle["chunks"]
    candidates = []
    for idx, score in zip(flat_ids, flat_scores):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        candidates.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            }
        )
        if len(candidates) >= candidate_pool:
            break
    return rerank_hybrid(question, candidates, top_k=top_k)


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """Load the saved FAISS index and retrieve top-k hits."""
    bundle = {
        "chunks": document["chunks"],
        "manifest": {
            "model_name": document["model_name"],
        },
        "model_source": document.get("model_source") or document["model_name"],
        "index": load_faiss_index(document["artifacts"]["index"]),
    }
    return search_bundle(
        question,
        bundle,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=history,
    )


def split_sentences(text: str) -> list[str]:
    """Split retrieved chunk text into candidate answer sentences."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return one short local answer sentence with a page tag when possible."""
    if not hits:
        return "No relevant information found in the document."

    q_tokens = keyword_set(question)
    best_sentence = ""
    best_page = None
    best_key = (-1, -1.0)  # (token overlap, hit score): overlap first, then rank
    for hit in hits:
        hit_score = float(hit.get("score", 0.0))
        for sentence in split_sentences(hit["text"]):
            s_tokens = keyword_set(sentence)
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            key = (overlap, hit_score)
            if key > best_key:
                best_key = key
                best_sentence = sentence
                best_page = hit["page"]
    if best_sentence:
        if best_page is not None:
            return f"{best_sentence} [Page {best_page}]"
        return best_sentence
    return hits[0]["text"][:200] + f" [Page {hits[0]['page']}]"
