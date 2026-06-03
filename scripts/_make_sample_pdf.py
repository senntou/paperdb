"""スモークテスト用のサンプルPDFを生成する（PyMuPDFのみ使用、追加依存なし）。
本番では不要。pdf/_sample_selectivenet.pdf を作る。"""
import fitz

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "pdf" / "_sample_selectivenet.pdf"

title = "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"
body = [
    title,
    "",
    "Abstract: We propose SelectiveNet, a deep neural network architecture",
    "that learns to abstain (reject) on uncertain inputs.",
    "",
    "The loss function combines an empirical selective risk term with a",
    "coverage constraint enforced via a quadratic penalty. Specifically, the",
    "objective is the selective risk r divided by the coverage phi, plus a",
    "lambda-weighted squared hinge penalty (c - phi)^2 that pushes the model",
    "to reach the target coverage c. An auxiliary head trained with standard",
    "cross-entropy regularizes the shared representation.",
]

doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "\n".join(body), fontsize=12)
# メタデータにもタイトルを入れておく
doc.set_metadata({"title": title})
doc.save(out)
doc.close()
print(f"wrote {out}")
