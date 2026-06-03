PY := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := ingest

.PHONY: setup ingest query clean help

help:
	@echo "make setup           # 仮想環境(.venv)作成 + 依存インストール（初回のみ）"
	@echo "make ingest          # pdf/ の新規・更新PDFを取り込み（テキスト化+ベクトル化+DB保存）"
	@echo 'make query Q="質問"  # DBに質問して上位ヒットを表示'
	@echo "make clean           # txt/ db/ .state/ を削除（PDFは消えない）"

setup:
	uv venv .venv
	uv pip install --python .venv -r requirements.txt
	@echo "セットアップ完了。次に: pdf/ にPDFを入れて 'make ingest'"

ingest:
	$(PY) scripts/ingest.py

query:
	@test -n "$(Q)" || (echo 'Q を指定してください: make query Q="質問文"'; exit 1)
	@$(PY) scripts/query.py search "$(Q)"

clean:
	rm -rf txt db .state
	@echo "txt/ db/ .state/ を削除しました（pdf/ は保持）"
