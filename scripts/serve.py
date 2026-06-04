"""
モデル常駐デーモン。query.py から自動起動される。
15分間リクエストがなければ自動終了。
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import query as query_mod

_IDLE_TIMEOUT = 900  # 15分
_last_activity = time.monotonic()
_lock = threading.Lock()


def _touch():
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/ping":
            self._reply(200, b"pong")
        else:
            self._reply(404, b"")

    def do_POST(self):
        _touch()
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        try:
            out = _dispatch(req)
            self._reply(200, json.dumps({"output": out}).encode())
        except Exception as e:
            self._reply(500, json.dumps({"error": str(e)}).encode())

    def _reply(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _dispatch(req: dict) -> str:
    ns = argparse.Namespace(**req["args"])
    try:
        collection = query_mod._check_db()
    except RuntimeError as e:
        return str(e) + "\n"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd = req["command"]
        if cmd == "search":
            query_mod._do_search(ns, collection)
        elif cmd == "expand":
            query_mod._do_expand(ns, collection)
        else:
            raise ValueError(f"unknown command: {cmd}")
    return buf.getvalue()


def _watchdog():
    while True:
        time.sleep(30)
        with _lock:
            if time.monotonic() - _last_activity > _IDLE_TIMEOUT:
                os._exit(0)


if __name__ == "__main__":
    try:
        srv = http.server.HTTPServer(("127.0.0.1", config.SERVER_PORT), _Handler)
    except OSError:
        # ポートが既に使用中 → 別インスタンスが起動済みなので静かに終了
        sys.exit(0)

    # モデルを先にロードしてから serve_forever でリクエスト受付開始
    config.load_model()
    config.get_collection()

    threading.Thread(target=_watchdog, daemon=True).start()
    srv.serve_forever()
