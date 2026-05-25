---
title: Auto README Studio
emoji: 📝
colorFrom: slate
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# Auto README Studio

This Hugging Face Space runs the React demo and Python analysis API in one Docker container.

## What ships in the Space

- `web/` is built during the Docker image build.
- `server.py` serves both the frontend and the `/api/analyze` endpoint from port `7860`.
- The frontend uses same-origin requests, so no extra API URL setting is needed in production.

## Local development

```bash
source .venv/bin/activate
python server.py
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

## Notes

- The Docker image installs the Python dependencies from `requirements.txt`.
- The frontend build uses the existing `web/package-lock.json` for deterministic installs.
- Example `.py` and `.txt` references in `references/` are included in the image.
