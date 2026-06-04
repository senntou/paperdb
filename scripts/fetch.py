"""urls.txt に記載された URL から PDF をダウンロードし、タイトルをファイル名にして pdf/ へ保存する。

使い方:
    python scripts/fetch.py          # urls.txt を読んで未取得のURLをダウンロード
    python scripts/fetch.py --force  # 取得済みURLも再ダウンロード

urls.txt 書式:
    # コメント行は無視
    https://arxiv.org/pdf/xxxx.xxxxx   # 行末コメントも可
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import config


FETCH_STATE_FILE = config.STATE_DIR / "urls.json"

# タイトル → ファイル名の変換で残す文字
_SAFE = re.compile(r"[^\w\s\-\(\)\[\],.]+", re.UNICODE)
# 連続スペースや先頭末尾のトリム用
_MULTI_SPACE = re.compile(r"\s+")


def load_url_state() -> dict:
    if FETCH_STATE_FILE.exists():
        return json.loads(FETCH_STATE_FILE.read_text())
    return {}


def save_url_state(state: dict) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    FETCH_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def read_urls(urls_file: Path) -> list[str]:
    """urls.txt を読んでURLリストを返す。コメント・空行を除去。"""
    if not urls_file.exists():
        return []
    urls = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            urls.append(line)
    return urls


def extract_title_from_pdf(pdf_path: Path) -> str:
    """PDFからタイトルを抽出する。ingest.py と同じロジック。"""
    import fitz

    doc = fitz.open(str(pdf_path))
    title = (doc.metadata or {}).get("title", "").strip()
    if not title:
        # 1ページ目の最初の非空行をタイトルとみなす
        if doc.page_count > 0:
            for line in doc[0].get_text("text").splitlines():
                if line.strip():
                    title = line.strip()
                    break
    doc.close()
    return title or pdf_path.stem


def title_to_filename(title: str, existing: set[str]) -> str:
    """タイトル文字列をファイル名（拡張子なし）に変換する。重複時は _2, _3 を付与。"""
    # 特殊文字を除去してスペース正規化
    name = _SAFE.sub(" ", title)
    name = _MULTI_SPACE.sub(" ", name).strip()
    # 長すぎる場合は最初の100文字に収める（単語境界で切る）
    if len(name) > 100:
        name = name[:100].rsplit(" ", 1)[0].rstrip()
    if not name:
        name = "paper"

    candidate = name
    n = 2
    while candidate in existing:
        candidate = f"{name}_{n}"
        n += 1
    return candidate


def download_pdf(url: str, dest: Path) -> None:
    """URLからPDFをダウンロードしてdestに保存する。"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 paperdb/1.0 (+https://github.com/user/paperdb)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)


def main(force: bool = False) -> int:
    urls_file = config.ROOT / "urls.txt"
    urls = read_urls(urls_file)

    if not urls:
        print(f"取得するURLがありません（{urls_file} が空か存在しない）")
        return 0

    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    state = load_url_state()

    # 既存PDFのstemセット（ファイル名重複回避用）
    existing_stems = {p.stem for p in config.PDF_DIR.glob("*.pdf")}

    added = skipped = failed = 0

    for url in urls:
        if not force and url in state:
            skipped += 1
            continue

        print(f"  ↓ {url}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            download_pdf(url, tmp_path)

            title = extract_title_from_pdf(tmp_path)
            stem = title_to_filename(title, existing_stems)
            dest = config.PDF_DIR / f"{stem}.pdf"

            tmp_path.rename(dest)
            existing_stems.add(stem)
            state[url] = {"filename": f"{stem}.pdf", "title": title}
            save_url_state(state)

            print(f"     → {dest.name}  「{title}」")
            added += 1

        except Exception as e:
            print(f"  ! エラー: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            failed += 1

    print(f"\n完了: ダウンロード {added} / スキップ（取得済） {skipped} / 失敗 {failed}")
    if added > 0:
        print("次に 'make ingest' を実行して取り込んでください。")
    return 0


if __name__ == "__main__":
    force = "--force" in sys.argv
    sys.exit(main(force=force))
