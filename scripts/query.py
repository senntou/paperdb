"""論文ベクトルDBへの検索CLI。

使い方:
    python scripts/query.py "SelectiveNetの損失関数は？"
    python scripts/query.py "質問" -k 8
"""
from __future__ import annotations

import argparse

import config


def main():
    ap = argparse.ArgumentParser(description="論文ベクトルDB検索")
    ap.add_argument("question", help="質問文（日本語可）")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="取得件数（既定5）")
    args = ap.parse_args()

    if not config.DB_DIR.exists():
        print("DBが未構築です。先に `make ingest` を実行してください。")
        return 1

    collection = config.get_collection()
    if collection.count() == 0:
        print("DBが空です。pdf/ にPDFを入れて `make ingest` を実行してください。")
        return 1

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

    print(f"質問: {args.question}\n")
    print(f"=== 上位 {len(docs)} 件 ===\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        score = 1.0 - dist  # cosine距離 → 類似度
        snippet = " ".join(doc.split())
        if len(snippet) > 500:
            snippet = snippet[:500] + " …"
        print(f"[{rank}] 類似度 {score:.3f}")
        print(f"    論文タイトル : {meta.get('title')}")
        print(f"    元PDF        : pdf/{meta.get('source')}.pdf  (p.{meta.get('page')})")
        print(f"    抜粋         : {snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
