"""Lightweight batch analysis API for the React README demo."""

from __future__ import annotations

import json
import os
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import app as pipeline
import readme_engine as re_eng


MAX_FILES = 10
DEFAULT_TOP_N = 2
DEFAULT_METRIC = "rougeL"


def _send_json(handler: BaseHTTPRequestHandler, payload: dict, status: int = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _safe_filename(filename: str | None, fallback_index: int) -> str:
    candidate = Path(filename or "").name.strip()
    if not candidate:
        candidate = f"example_{fallback_index + 1}.py"
    if not candidate.endswith(".py"):
        raise ValueError(f"Only .py files are supported: {candidate}")
    return candidate


def _parse_uploads(handler: BaseHTTPRequestHandler) -> tuple[list[dict], str, int]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data request")

    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length)
    multipart_message = BytesParser(policy=email_default_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + raw_body
    )

    if not multipart_message.is_multipart():
        raise ValueError("Upload at least one Python file")

    files: list[dict] = []
    metric = DEFAULT_METRIC
    top_n = DEFAULT_TOP_N

    for part in multipart_message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue

        if field_name == "files":
            filename = _safe_filename(part.get_filename(), len(files))
            content = part.get_payload(decode=True) or b""
            files.append({"filename": filename, "content": content})
            continue

        text_value = part.get_content()
        if field_name == "metric" and isinstance(text_value, str):
            metric = text_value.strip() or DEFAULT_METRIC
        elif field_name == "top_n" and isinstance(text_value, str):
            try:
                top_n = max(1, min(int(text_value.strip()), 5))
            except ValueError as exc:
                raise ValueError("top_n must be an integer") from exc

    if not files:
        raise ValueError("Upload at least one Python file")
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} Python files")

    return files, metric, top_n


def _analyse_batch(files: list[dict], metric: str, top_n: int) -> dict:
    with TemporaryDirectory(prefix="auto-readme-") as tmpdir:
        tmpdir_path = Path(tmpdir)

        file_paths: list[Path] = []
        for item in files:
            path = tmpdir_path / item["filename"]
            path.write_bytes(item["content"])
            file_paths.append(path)

        corpus = pipeline.build_local_corpus(tmpdir, verbose=False)
        tfidf_model, bm25_model, lsa_model, baseline_model = pipeline.fit_all_models(
            corpus, verbose=False
        )

        results: list[dict] = []
        analysis_args = SimpleNamespace(metric=metric, top_n=top_n, all_metrics=True)

        for path in file_paths:
            file_args = SimpleNamespace(**vars(analysis_args))
            rouge_scores, bertscore_scores, coverage_scores, winning_model, hypotheses, metadata = pipeline.run_single_file(
                str(path),
                file_args,
                tfidf_model,
                bm25_model,
                lsa_model,
                baseline_model,
                verbose=False,
            )

            readme = re_eng.generate_markdown(
                metadata=metadata,
                filename=path.name,
                summary_sentences=hypotheses[winning_model],
                rouge_scores=rouge_scores,
                bertscore_scores=bertscore_scores,
                coverage_scores=coverage_scores,
                winning_model=winning_model,
            )

            results.append(
                {
                    "filename": path.name,
                    "winning_model": winning_model,
                    "winning_metric": file_args.metric,
                    "summary_sentences": hypotheses[winning_model],
                    "metadata": metadata,
                    "rouge_scores": rouge_scores,
                    "bertscore_scores": bertscore_scores,
                    "coverage_scores": coverage_scores,
                    "readme": readme,
                }
            )

        return {
            "uploaded_count": len(files),
            "metric": metric,
            "top_n": top_n,
            "files": results,
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "AutoReadmeDemo/1.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/api/health"}:
            _send_json(
                self,
                {
                    "status": "ok",
                    "message": "Auto README demo API is running.",
                    "max_files": MAX_FILES,
                },
            )
            return
        _send_json(self, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            _send_json(self, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            files, metric, top_n = _parse_uploads(self)
            payload = _analyse_batch(files, metric, top_n)
            payload["warnings"] = []
            payload["max_files"] = MAX_FILES
            _send_json(self, payload)
        except Exception as exc:  # pragma: no cover - surfaced to the browser
            _send_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def main() -> None:
    host = os.environ.get("AUTO_README_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTO_README_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    print(f"[API] Serving on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[API] Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()