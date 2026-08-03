import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the PDF, say that the document does not provide enough information. "
    "Never invent a page number."
)


def answer_from_pages(pages: list[dict], message: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PDF text:\n{document_text}\n\nmessage: {message}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def stream_answer_from_pages(pages: list[dict], message: str):
    """Yield answer text deltas for `message` grounded in `pages`.

    Streams from OpenRouter (stream=True). Raises RuntimeError if the API
    key is missing; yields nothing if the stream errors mid-way.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    stream = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        temperature=0.0,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PDF text:\n{document_text}\n\nmessage: {message}",
            },
        ],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def rewrite_query(message: str, max_tokens: int = 80) -> str:
    """Paraphrase a vague question into a keyword search query for the retriever.

    Best-effort: returns the original message on any failure so retrieval always
    has something to embed.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return message

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Paraphrase the user question into ONE keyword search query "
                        "that a PDF text retriever can match. Include the key terms and "
                        "likely answer keywords. Do NOT invent facts or names. "
                        "Return only the query sentence, no explanations."
                    ),
                },
                {"role": "user", "content": message},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return text or message
    except Exception:
        return message
