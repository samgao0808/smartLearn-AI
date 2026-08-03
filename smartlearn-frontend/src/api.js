export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const CHAT_ID = "day2-demo";

async function readJSON(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return data;
}

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(
    `${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`,
    { method: "POST", body: formData },
  );
  return readJSON(response);
}

export async function askQuestion(message) {
  const response = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: message.trim(), chat_id: CHAT_ID }),
  });
  return readJSON(response);
}

function parseSSEBlock(block) {
  let type = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (data.length === 0) return null;
  return { type, data: data.join("\n") };
}

/**
 * Ask a question and stream the answer over SSE.
 * Calls onMeta({citations, sources}) first, then onDelta(text) per token,
 * resolving when the server sends "done".
 */
export async function askQuestionStream(message, { onMeta, onDelta } = {}) {
  const response = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: message.trim(), chat_id: CHAT_ID }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  if (!response.body) {
    throw new Error("Streaming not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      const event = parseSSEBlock(block);
      if (!event) continue;
      if (event.type === "meta") onMeta?.(JSON.parse(event.data));
      else if (event.type === "delta") onDelta?.(JSON.parse(event.data).text);
      else if (event.type === "error") {
        throw new Error(JSON.parse(event.data).detail || "Chat failed.");
      } else if (event.type === "done") {
        return;
      }
    }
  }
}
