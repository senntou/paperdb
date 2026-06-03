"""PDF → テキスト → チャンク → 埋め込み → ChromaDB へ増分取り込み。

使い方:
    python scripts/ingest.py        # pdf/ 内の新規・更新PDFのみ処理
"""
from __future__ import annotations

import hashlib
import json
import sys

import config


# --- ハッシュ管理 -------------------------------------------------------
def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if config.STATE_FILE.exists():
        return json.loads(config.STATE_FILE.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


# --- PDF 抽出 -----------------------------------------------------------
def extract_pages(pdf_path):
    """(ページ番号, テキスト) のリストと推定タイトルを返す。"""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append((i, page.get_text("text")))

    # タイトル推定: PDFメタdata → 1ページ目の最初の非空行 → ファイル名
    title = (doc.metadata or {}).get("title") or ""
    title = title.strip()
    if not title and pages:
        for line in pages[0][1].splitlines():
            if line.strip():
                title = line.strip()
                break
    if not title:
        title = pdf_path.stem
    doc.close()
    return pages, title


def chunk_pages(pages):
    """ページ境界を保ちつつ文字数ベースでチャンク化。(text, page) を yield。"""
    size, overlap = config.CHUNK_CHARS, config.CHUNK_OVERLAP
    for page_no, text in pages:
        text = text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + size]
            if piece.strip():
                yield piece, page_no
            if start + size >= len(text):
                break
            start += size - overlap


# --- メイン処理 ---------------------------------------------------------
def process_pdf(pdf_path, collection):
    stem = pdf_path.stem
    pages, title = extract_pages(pdf_path)

    # 人間確認用に txt/ へ保存
    config.TXT_DIR.mkdir(parents=True, exist_ok=True)
    txt_out = config.TXT_DIR / f"{stem}.txt"
    txt_out.write_text(
        "\n\n".join(f"[page {p}]\n{t}" for p, t in pages), encoding="utf-8"
    )

    chunks = list(chunk_pages(pages))
    if not chunks:
        print(f"  ! {pdf_path.name}: テキスト抽出ゼロ（スキャンPDFの可能性）。スキップ")
        return 0

    # 既存チャンクを消してから入れ直す（更新対応）
    collection.delete(where={"source": stem})

    texts = [c[0] for c in chunks]
    embeddings = config.embed(texts, kind="passage")
    ids = [f"{stem}::{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": stem, "title": title, "page": page, "chunk_idx": i}
        for i, (_, page) in enumerate(chunks)
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  + {pdf_path.name}  「{title}」  {len(chunks)} チャンク")
    return len(chunks)


def main():
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    collection = config.get_collection()

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    present = {p.stem for p in pdfs}

    added = updated = removed = 0

    # 追加・更新
    for pdf in pdfs:
        stem = pdf.stem
        digest = md5(pdf)
        prev = manifest.get(stem)
        if prev and prev.get("hash") == digest:
            continue  # 変更なし
        n = process_pdf(pdf, collection)
        if n == 0:
            continue
        if prev:
            updated += 1
        else:
            added += 1
        manifest[stem] = {"hash": digest, "chunks": n}

    # 削除（PDFが消えたもの）
    for stem in list(manifest.keys()):
        if stem not in present:
            collection.delete(where={"source": stem})
            del manifest[stem]
            removed += 1
            print(f"  - {stem}: PDF削除を検知 → DBから除去")

    save_manifest(manifest)
    total = collection.count()
    print(
        f"\n完了: 追加 {added} / 更新 {updated} / 削除 {removed}  "
        f"（DB総チャンク数 {total}, 登録論文 {len(manifest)} 本）"
    )


if __name__ == "__main__":
    sys.exit(main())
