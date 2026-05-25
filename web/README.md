# Auto README Studio

React front-end for the local README generator.

## Run locally

1. Start both services from the repo root with the helper script:

```bash
./dev.sh
```

If you prefer separate terminals, start the Python API server from the project virtualenv:

```bash
source ../.venv/bin/activate && python server.py
```

2. In a second terminal, install the web dependencies and start Vite:

```bash
cd web
npm install
npm run dev
```

The app proxies `/api` requests to `http://127.0.0.1:8000`, so uploads and README generation stay local.
The first BERTScore-backed request may download model weights the first time it runs.

## Deploying the frontend to Vercel / connecting to a Hugging Face Space backend

- Set an environment variable in your Vercel project named `VITE_API_URL` with the value of your Space runtime URL (use the `hf.space` host). Example:

```
VITE_API_URL=https://donut12345-nlp-auto-readme.hf.space
```

- In Vercel: Project → Settings → Environment Variables → Add `VITE_API_URL` (set for `Production`). Then redeploy the Vercel project.

- Locally, create `web/.env` from `web/.env.example` for testing and run `npm run dev`.

- The frontend already reads `import.meta.env.VITE_API_URL` as its API base, and the backend allows cross-origin JSON responses, so the deployed frontend will call `${VITE_API_URL}/api/analyze`.
