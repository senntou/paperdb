"""共通設定・パス・モデルロード。ingest.py / query.py から参照する。"""
from __future__ import annotations

import functools
from pathlib import Path

# --- パス ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "pdf"
TXT_DIR = ROOT / "txt"
DB_DIR = ROOT / "db"
STATE_DIR = ROOT / ".state"
STATE_FILE = STATE_DIR / "manifest.json"

# --- 埋め込みモデル -----------------------------------------------------
# 日英クロスリンガル検索に強い多言語モデル。e5系は passage/query プレフィックス必須。
MODEL_NAME = "intfloat/multilingual-e5-large"
COLLECTION = "papers"
SERVER_PORT = 18964

# --- チャンク設定 -------------------------------------------------------
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150


def _pick_device():
    """使用デバイスを決定。環境変数 PAPERDB_DEVICE で明示指定可（cpu/cuda）。既定は CUDA 自動検出。"""
    import os

    forced = os.environ.get("PAPERDB_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda") or forced.startswith("cuda:"):
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@functools.lru_cache(maxsize=1)
def load_model():
    """SentenceTransformer をロード。初回はモデルDL(~1GB)。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=_pick_device())


def embed(texts, *, kind: str):
    """texts を埋め込む。kind は 'passage'(本文) か 'query'(質問)。"""
    if kind not in ("passage", "query"):
        raise ValueError("kind must be 'passage' or 'query'")
    model = load_model()
    prefixed = [f"{kind}: {t}" for t in texts]
    device = _pick_device()
    batch_size = 128 if device == "cuda" else 32
    embs = model.encode(
        prefixed,
        normalize_embeddings=True,  # cosine 用に正規化
        show_progress_bar=len(prefixed) > batch_size,
        batch_size=batch_size,
    )
    return embs.tolist()


def get_collection():
    """ChromaDB の永続コレクションを返す（埋め込みは自前で渡すので embedder 指定なし）。"""
    import chromadb

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
