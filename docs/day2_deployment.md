# Day 2 Deployment

## URLs
- Frontend: https://smartlearn-ai-ten.vercel.app/
- Backend health: https://smartlearn-ai-production-113f.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-113f.up.railway.app/docs

## Source
- Repository: samgao0808/smartLearn-AI
- Deployed branch / merge target: main
- Merged commit: eee1530
- Pull Request: https://github.com/samgao0808/smartLearn-AI/pull/1

## Root Directories
- Railway: smartlearn-backend (railway.toml)
- Vercel: smartlearn-frontend

## Environment variable names
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS, PORT (auto)
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass
- Upload: pass
- Known /chat + citations: pass
- Unknown question: pass
- Public UI full flow: pass

## Known limitations
- Railway restart clears in-memory uploaded/chat state; re-upload is expected.
- Free OpenRouter model may occasionally return empty responses; retry resolves.
- PDF max 30 pages, no OCR for scanned documents.
- No authentication, no database persistence, single chat_id.
