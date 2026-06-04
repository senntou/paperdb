# paperdb — 論文ベクトルDB

このリポジトリは、`pdf/` に置いた論文PDFをテキスト化・ベクトル化して ChromaDB に保存し、
自然言語（日本語可）で検索できる個人用の論文知識ベースです。

## Claude への指示（重要）

ユーザーが**論文の内容についての質問**（例:「SelectiveNetの損失関数ってどうだっけ？」
「○○手法の評価指標は？」など）をしてきたら、自分の記憶だけで答えず、**必ず次のコマンドで
DBを検索**し、その結果に基づいて回答してください。

```
# 意味検索（既定: 上位5件、抜粋300文字）
uv run python scripts/query.py search "<ユーザーの質問をそのまま、または検索向けに整えた文>"

# 1論文1件に絞って一覧表示（重複排除）
uv run python scripts/query.py search "<質問>" -u

# タイトルのみ一覧（抜粋なし）
uv run python scripts/query.py search "<質問>" -t

# ヒットしたチャンクの前後を展開（source::chunk_idx は検索結果の「チャンクID」欄から）
uv run python scripts/query.py expand "<source::chunk_idx>" -w 2
```

- 回答には**必ず根拠となる論文タイトルと元PDF（`pdf/xxx.pdf` のp.N）を明示**する。
- ヒットが弱い／無関係そうなら、言い換えてもう一度検索する、または「DBに該当が無さそう」と正直に伝える。
- 検索結果の「チャンクID」（例: `selectivenet::3`）を `expand` に渡すと前後の文脈を確認できます。
- それでも文脈が足りなければ `txt/<source>.txt` を Read して補ってください。

## 取り込み（ユーザー操作）

新しいPDFを `pdf/` に入れたら、ユーザー（またはあなた）が次を実行して取り込みます:

```
make ingest    # 新規・更新されたPDFだけを増分処理
```

PDFを削除して `make ingest` すると、そのチャンクはDBから自動的に除去されます。

## セットアップ（初回のみ）

```
make setup     # 依存導入（sentence-transformers / chromadb / pymupdf / torch-CUDA）
```

初回の `make ingest` / `make query` 時に埋め込みモデル
`intfloat/multilingual-e5-large`（~1GB）が自動DLされます。以降はオフラインで動作。

## 構成

| パス | 役割 |
|------|------|
| `pdf/` | 入力。論文PDFをここに置く |
| `txt/` | 抽出テキスト（自動生成、確認用） |
| `db/`  | ChromaDB 永続ストア（自動生成） |
| `scripts/config.py` | パス・モデル名・チャンク設定・埋め込み関数 |
| `scripts/ingest.py` | 取り込みパイプライン（増分） |
| `scripts/query.py`  | 検索CLI |

## 注意
- 画像のみのスキャンPDFはテキスト抽出できません（OCR未対応）。
