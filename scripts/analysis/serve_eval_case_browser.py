#!/usr/bin/env python3
"""Serve the static eval case browser with a separate generated data directory."""

from __future__ import annotations

import argparse
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def main() -> None:
    args = parse_args()
    site_dir = Path(args.site_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not (site_dir / "index.html").exists():
        raise SystemExit(f"missing index.html in {site_dir}")
    if not (data_dir / "app_data.json").exists():
        raise SystemExit(f"missing app_data.json in {data_dir}; run build_eval_case_browser.py first")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("", "/"):
                self._send_file(site_dir / "index.html")
                return
            if parsed.path.startswith("/data/"):
                rel = unquote(parsed.path.removeprefix("/data/"))
                self._send_file(_safe_join(data_dir, rel))
                return
            rel = unquote(parsed.path.lstrip("/"))
            self._send_file(_safe_join(site_dir, rel))

        def log_message(self, fmt: str, *items: object) -> None:
            if not args.quiet:
                super().log_message(fmt, *items)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in {".json", ".jsonl"}:
                content_type = "application/json; charset=utf-8"
            if path.suffix in {".html", ".css", ".js"}:
                content_type = content_type + "; charset=utf-8"
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[eval-case-browser] url=http://{args.host}:{args.port}")
    print(f"[eval-case-browser] site={site_dir}")
    print(f"[eval-case-browser] data={data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--site-dir", default="docs/evaluation/eval_case_browser")
    parser.add_argument("--data-dir", default="/tmp/shared-storage/OnPolicy/analysis/eval_case_browser")
    return parser.parse_args()


def _safe_join(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if root != candidate and root not in candidate.parents:
        raise PermissionError(f"path escapes root: {rel}")
    return candidate


if __name__ == "__main__":
    main()
