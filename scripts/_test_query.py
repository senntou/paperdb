"""query.py の動作確認スクリプト（モデルDL不要）。
実際の埋め込みモデルの代わりに TF-IDF ベクトルを使い、
ingest → search → expand の一連フローをテストする。
"""
from __future__ import annotations
import sys, subprocess, shutil, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ---- TF-IDF スタブで config.embed を差し替え ----------------------------
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_corpus: list[str] = []
_vect: TfidfVectorizer | None = None
_dim = 128

def _stub_embed(texts, *, kind):
    global _corpus, _vect, _dim
    _corpus.extend(texts)
    vect = TfidfVectorizer(max_features=_dim, sublinear_tf=True)
    mat = vect.fit_transform(_corpus).toarray()
    # 今回のテキストだけ取り出してL2正規化
    result = mat[-len(texts):]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (result / norms).tolist()

import config as _config
_config.embed = _stub_embed
# -------------------------------------------------------------------------

# DB をまっさらにしてからテスト
db_dir = ROOT / "db"
state_dir = ROOT / ".state"
txt_dir = ROOT / "txt"
for d in (db_dir, state_dir, txt_dir):
    if d.exists():
        shutil.rmtree(d)

print("=" * 60)
print("STEP 1: ingest (TF-IDF stub)")
print("=" * 60)
import ingest
ingest.main()

PY = str(ROOT / ".venv/bin/python")

def run(label, *args):
    print("\n" + "─" * 60)
    print(f"TEST: {label}")
    print("─" * 60)
    # query.py を直接インポートして実行（同プロセス内でスタブが有効）
    sys.argv = ["query.py", *args]
    import importlib, query
    importlib.reload(query)
    try:
        query.main()
    except SystemExit:
        pass

print("\n" + "=" * 60)
print("STEP 2: search テスト")
print("=" * 60)

run("通常検索 (default)", "search", "loss function for selective prediction")
run("重複排除 -u", "search", "uncertainty estimation neural network", "-u")
run("タイトル一覧 -t", "search", "attention mechanism transformer", "-t", "-k", "10")
run("スニペット短縮 -s 100", "search", "coverage constraint training", "-s", "100")
run("スニペット非表示 -s 0", "search", "deep learning reject option", "-s", "0")

print("\n" + "=" * 60)
print("STEP 3: expand テスト")
print("=" * 60)

# search して chunk_id を拾う
collection = _config.get_collection()
q_emb = _stub_embed(["attention self-attention positional encoding"], kind="query")
res = collection.query(query_embeddings=q_emb, n_results=1,
                       include=["metadatas"])
meta = res["metadatas"][0][0]
chunk_id = f"{meta['source']}::{meta['chunk_idx']}"
print(f"\n→ expand 対象: {chunk_id}")

run(f"expand -w 1 ({chunk_id})", "expand", chunk_id, "-w", "1")
run(f"expand -w 2 ({chunk_id})", "expand", chunk_id, "-w", "2")

print("\n" + "=" * 60)
print("ALL TESTS DONE")
print("=" * 60)
