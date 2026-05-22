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