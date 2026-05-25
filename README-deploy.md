# Deployment guide — Vercel frontend + Python backend

This file explains how to deploy the `web/` frontend to Vercel and the Python API (`server.py`) to a Python-friendly host (Render, Railway, Fly). Recommended: host backend separately and point the frontend to it via `VITE_API_URL`.

Prerequisites
- A GitHub repo with this project pushed
- Node.js and npm for the frontend
- A Python environment for testing the backend

Quick commit
```bash
git add .
git commit -m "prepare deployment: requirements, VITE_API_URL, server PORT/host"
git push origin main
```

Backend (Render example)
1. Add `requirements.txt` (already included).
2. Create a new Web Service on Render and connect your repo.
3. Build / Install: Render will run `pip install -r requirements.txt` (or add that command).
4. Start command: `python server.py` (Render provides `PORT`).
5. Render exposes a public URL like `https://your-backend.onrender.com` — copy this for the frontend.

Railway / Fly notes
- Railway: create a new project, link repo, and set the start command `python server.py`.
- Fly: prefer a small Dockerfile; set `CMD ["python","server.py"]` and deploy.

Server config
- `server.py` now reads `PORT` and binds to `0.0.0.0` by default. No further changes required on most hosts.
- CORS: responses include `Access-Control-Allow-Origin: *`.

Frontend (Vercel)
1. In Vercel, import your GitHub repo and set the Root Directory to `web`.
2. Vite is auto-detected. Build command: `npm run build`. Output directory: `dist`.
3. Set environment variable `VITE_API_URL` to your backend URL, e.g. `https://your-backend.onrender.com` (no trailing slash).
4. Deploy.

Local testing
- Backend locally:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```
- Frontend locally (inside `web/`):
```bash
cd web
npm install
export VITE_API_URL=http://localhost:8000
npm run dev
```

Alternative: single-host on Vercel (advanced)
- You can port `server.py` to a serverless function in `web/api/` but beware size/time limits and binary dependencies. I recommend the two-host approach for reliability.

Troubleshooting
- 502 / network errors: ensure `VITE_API_URL` points to a reachable backend and that the backend is running.
- Timeouts: analysis is CPU/IO-bound; use a host with enough resources or reduce batch size.

If you want, I can (A) deploy the backend to Render for you (requires repo access) or (B) port the API to a Vercel serverless function — tell me which.
