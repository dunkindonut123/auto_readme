# Auto README Studio

Auto README Studio is a local demo website for the README generator in this project.
It lets users upload up to 10 Python files, runs the summarization models on each file,
and shows a generated README for every upload. The website also displays the model
winner and lets users copy or download the README output.

## What it does

- Upload multiple `.py` files at once, up to 10 files per batch.
- Compare the local TF-IDF, BM25, and LSA summarizers.
- Generate a README for each file with summary, usage example, and structure details.
- Expand each result card only when you click it.

## Run the website

### Option 1: one command

From the repo root, run:

```bash
./dev.sh
```

This starts the Python API on `http://127.0.0.1:8000` and the React app with Vite.

### Option 2: separate terminals

Terminal 1:

```bash
source ../.venv/bin/activate && python server.py
```

Terminal 2:

```bash
cd web
npm install
npm run dev
```

If Vite reports that port `5173` is busy, it will choose another port such as `5174`.

## How it works

The frontend sends uploaded files to the local API at `/api/analyze`.
The backend writes each file to a temporary workspace, runs the existing README
generation pipeline, and returns JSON with the generated markdown and evaluation data.

## Project structure

- `server.py` starts the local API used by the website.
- `web/` contains the React/Vite frontend.
- `app.py` and `readme_engine.py` contain the existing README generation pipeline.
- `test_corpus/` and `references/` provide sample files for evaluation.

## Notes

- Use the project virtualenv when starting the API.
- The first BERTScore-backed request may download model weights the first time it runs.
