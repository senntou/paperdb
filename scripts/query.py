"""論文ベクトルDBへの検索CLI。

使い方:
    python scripts/query.py search "SelectiveNetの損失関数は？"
    python scripts/query.py search "質問" -k 8 -u -s 200
    python scripts/query.py search "論文一覧" -t
    python scripts/query.py expand "selectivenet::3" -w 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

_SERVER_URL = f"http://127.0.0.1:{config.SERVER_PORT}"


# ── DB チェック ─────────────────────────────────────────────────────────────

def _check_db():
    if not config.DB_DIR.exists():
        raise RuntimeError("DBが未構築です。先に `make ingest` を実行してください。")
    collection = config.get_collection()
    if collection.count() == 0:
        raise RuntimeError("DBが空です。pdf/ にPDFを入れて `make ingest` を実行してください。")
    return collection


# ── ビジネスロジック（サーバー・直接呼び出し共通） ───────────────────────────

def _do_search(args, collection):
    q_emb = config.embed([args.question], kind="query")
    res = collection.query(
        query_embeddings=q_emb,
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not docs:
        print("該当する箇所が見つかりませんでした。")
        return 0

    titles_only = args.titles_only
    snippet_len = 0 if titles_only else args.snippet
    unique = args.unique or titles_only

    if unique:
        seen: dict[str, int] = {}
        filtered = []
        for i, meta in enumerate(metas):
            src = meta.get("source", "")
            if src not in seen:
                seen[src] = i
                filtered.append(i)
        docs  = [docs[i]  for i in filtered]
        metas = [metas[i] for i in filtered]
        dists = [dists[i] for i in filtered]

    print(f"質問: {args.question}\n")
    print(f"=== 上位 {len(docs)} 件 ===\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        score = 1.0 - dist
        source = meta.get("source", "")
        chunk_idx = meta.get("chunk_idx", "?")
        print(f"[{rank}] 類似度 {score:.3f}")
        print(f"    論文タイトル : {meta.get('title')}")
        print(f"    元PDF        : pdf/{source}.pdf  (p.{meta.get('page')})")
        print(f"    チャンクID   : {source}::{chunk_idx}")
        if snippet_len != 0:
            snippet = " ".join(doc.split())
            if snippet_len > 0 and len(snippet) > snippet_len:
                snippet = snippet[:snippet_len] + " …"
            print(f"    抜粋         : {snippet}")
        print()
    return 0


def _do_expand(args, collection):
    try:
        source, idx_str = args.chunk_id.rsplit("::", 1)
        center = int(idx_str)
    except ValueError:
        print(f"エラー: チャンクIDの形式が不正です。 例: selectivenet::3")
        return 1

    window = args.window
    indices = list(range(max(0, center - window), center + window + 1))

    ids_to_fetch = [f"{source}::{i}" for i in indices]
    res = collection.get(
        ids=ids_to_fetch,
        include=["documents", "metadatas"],
    )

    if not res["ids"]:
        print(f"チャンク '{args.chunk_id}' が見つかりませんでした。")
        return 1

    pairs = sorted(
        zip(res["ids"], res["documents"], res["metadatas"]),
        key=lambda x: x[2].get("chunk_idx", 0),
    )

    title = pairs[0][2].get("title", source)
    print(f"論文: {title}  (pdf/{source}.pdf)\n")
    for cid, doc, meta in pairs:
        marker = " ◀" if meta.get("chunk_idx") == center else ""
        print(f"--- [{cid}]  p.{meta.get('page')}{marker} ---")
        print(doc.strip())
        print()
    return 0


# ── サーバークライアント ──────────────────────────────────────────────────────

def _ping() -> bool:
    try:
        urllib.request.urlopen(f"{_SERVER_URL}/ping", timeout=0.5)
        return True
    except Exception:
        return False


def _start_server() -> bool:
    """serve.py をバックグラウンドで起動し、応答が来るまで待つ（最大30秒）。"""
    serve_py = Path(__file__).parent / "serve.py"
    subprocess.Popen(
        [sys.executable, str(serve_py)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):  # 0.5s × 60 = 30s
        time.sleep(0.5)
        if _ping():
            return True
    return False


def _ensure_server() -> bool:
    return _ping() or _start_server()


def _call_server(command: str, args_dict: dict) -> str | None:
    payload = json.dumps({"command": command, "args": args_dict}).encode()
    req = urllib.request.Request(
        f"{_SERVER_URL}/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("output", "")
    except Exception:
        return None


# ── CLIコマンド ───────────────────────────────────────────────────────────────

def cmd_search(args):
    if _ensure_server():
        out = _call_server("search", {
            "question": args.question,
            "top_k": args.top_k,
            "unique": args.unique,
            "snippet": args.snippet,
            "titles_only": args.titles_only,
        })
        if out is not None:
            print(out, end="")
            return 0
    # サーバーが使えない場合は直接実行
    try:
        collection = _check_db()
    except RuntimeError as e:
        print(e)
        return 1
    return _do_search(args, collection)


def cmd_expand(args):
    if _ensure_server():
        out = _call_server("expand", {
            "chunk_id": args.chunk_id,
            "window": args.window,
        })
        if out is not None:
            print(out, end="")
            return 0
    # サーバーが使えない場合は直接実行
    try:
        collection = _check_db()
    except RuntimeError as e:
        print(e)
        return 1
    return _do_expand(args, collection)


def main():
    ap = argparse.ArgumentParser(description="論文ベクトルDB検索")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sp = sub.add_parser("search", help="意味検索")
    sp.add_argument("question", help="質問文（日本語可）")
    sp.add_argument("-k", "--top-k",  type=int, default=5,   help="取得チャンク数（既定5）")
    sp.add_argument("-u", "--unique", action="store_true",   help="1論文1チャンク（最高スコア）に絞る")
    sp.add_argument("-s", "--snippet",type=int, default=300, help="抜粋の最大文字数（0で非表示、既定300）")
    sp.add_argument("-t", "--titles-only", action="store_true", help="タイトル＋スコアのみ表示（-u -s 0 相当）")
    sp.set_defaults(func=cmd_search)

    ep = sub.add_parser("expand", help="指定チャンクの前後を表示")
    ep.add_argument("chunk_id", help="チャンクID（例: selectivenet::3）")
    ep.add_argument("-w", "--window", type=int, default=2, help="前後何チャンクを表示するか（既定2）")
    ep.set_defaults(func=cmd_expand)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
