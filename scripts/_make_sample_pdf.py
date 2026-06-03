"""スモークテスト用のサンプルPDFを3本生成する（PyMuPDFのみ使用）。
本番では不要。pdf/_sample_*.pdf を作る。"""
import fitz
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "pdf"
PDF_DIR.mkdir(exist_ok=True)

PAPERS = [
    {
        "filename": "_sample_selectivenet.pdf",
        "title": "SelectiveNet: A Deep Neural Network with an Integrated Reject Option",
        "pages": [
            """\
SelectiveNet: A Deep Neural Network with an Integrated Reject Option

Abstract
We propose SelectiveNet, a deep neural network architecture that learns to abstain
(reject) on uncertain inputs. Unlike post-hoc thresholding approaches, SelectiveNet
jointly optimizes classification accuracy and coverage through a purpose-built loss.

1. Introduction
In many real-world applications, it is preferable for a model to abstain on ambiguous
inputs rather than make an incorrect prediction with false confidence. This is known
as the selective prediction or classification-with-rejection problem. Existing methods
typically add a threshold on top of a trained classifier, which does not optimize the
model directly for rejection.

SelectiveNet addresses this gap by introducing a dedicated selection head alongside
the prediction head. Both heads share a common feature extractor. The selection head
outputs a scalar g(x) in [0,1] representing the model's confidence that input x
should be classified rather than rejected.
""",
            """\
2. Method: Loss Function

The training objective of SelectiveNet is composed of three terms.

Let f(x) denote the prediction head output and g(x) the selection head output.
The selective risk is defined as:

    r(f, g) = E[L(f(x), y) * g(x)] / E[g(x)]

where L is the task loss (e.g. cross-entropy) and the denominator is the empirical
coverage phi = E[g(x)].

The full loss is:

    L_total = r(f, g) + lambda * max(0, c - phi)^2 + alpha * L_aux(h(x), y)

where:
  - c is the target coverage (a hyperparameter, e.g. 0.8)
  - lambda controls the penalty strength for deviating from target coverage
  - h(x) is an auxiliary prediction head trained with standard cross-entropy
  - alpha balances the auxiliary regularization term

The quadratic penalty (c - phi)^2 pushes the model to achieve exactly the desired
coverage, while the auxiliary head prevents the shared representation from collapsing.
""",
            """\
3. Experiments

We evaluate SelectiveNet on CIFAR-10, CIFAR-100, and CatsVsDogs.

3.1 Baselines
We compare against:
  - SR (Selective Risk): post-hoc rejection via softmax threshold
  - MC-Dropout: uncertainty estimated via dropout at test time
  - Deep Ensemble: uncertainty via variance across multiple trained models

3.2 Results
At coverage c=0.8, SelectiveNet achieves a selective risk of 0.031 on CIFAR-10,
outperforming SR (0.041) and MC-Dropout (0.038).

On CIFAR-100 at c=0.7, the improvement is even larger: SelectiveNet reaches 0.18
vs SR at 0.27, a 33% relative reduction.

3.3 Ablation
Removing the auxiliary head increases selective risk by ~12%, confirming that the
auxiliary loss is essential for representation quality.

4. Conclusion
SelectiveNet demonstrates that jointly optimizing rejection and classification
significantly improves selective risk compared to post-hoc thresholding, especially
at low coverage regimes.
""",
        ],
    },
    {
        "filename": "_sample_deep_ensemble.pdf",
        "title": "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
        "pages": [
            """\
Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

Abstract
We propose Deep Ensembles, a simple yet powerful method for estimating predictive
uncertainty in neural networks. By training M networks independently from different
random initializations and averaging their predictions, we obtain well-calibrated
uncertainty estimates that outperform Bayesian approximations such as MC-Dropout
on both in-distribution and out-of-distribution (OOD) data.

1. Introduction
Reliable uncertainty quantification is critical for safety-sensitive applications
such as medical diagnosis and autonomous driving. Bayesian neural networks offer
a principled framework but are computationally intractable for large models.
Variational inference and Monte Carlo Dropout provide approximations, but their
calibration is often poor.

Deep Ensembles bypass Bayesian inference entirely. Each ensemble member is trained
with the same data but a different random seed, leading to diverse function
representations due to the non-convexity of the loss landscape.
""",
            """\
2. Method

2.1 Training
Given a dataset D = {(x_i, y_i)}, we train M neural networks theta_1, ..., theta_M
independently by minimizing the negative log-likelihood:

    L(theta) = -sum_i log p(y_i | x_i, theta)

For regression, we predict both mean mu(x) and variance sigma^2(x) using a Gaussian
output layer. For classification, we use standard softmax cross-entropy.

2.2 Inference
Predictions are combined via a uniform mixture:

    p(y | x) = (1/M) * sum_m p(y | x, theta_m)

Uncertainty is captured by the predictive variance, which decomposes into:
  - Aleatoric uncertainty: average of individual variances
  - Epistemic uncertainty: variance of individual means (model disagreement)

2.3 Adversarial Training
We optionally add adversarial examples (FGSM) during training to improve the
smoothness of prediction boundaries, which further improves calibration.
""",
            """\
3. Evaluation

3.1 Calibration
We measure calibration via Expected Calibration Error (ECE) and reliability diagrams.
On CIFAR-10, Deep Ensembles (M=5) achieves ECE=0.012, compared to MC-Dropout at
ECE=0.032 and a single deterministic network at ECE=0.041.

3.2 Out-of-Distribution Detection
On SVHN (used as OOD for CIFAR-10 trained models), Deep Ensembles yields AUROC=0.91,
while MC-Dropout achieves 0.83. Ensembles correctly assign high entropy to OOD inputs.

3.3 Scalability
Deep Ensembles scale linearly in compute with M. In practice M=5 provides most of the
benefit; gains plateau beyond M=10.

4. Conclusion
Deep Ensembles are a simple, scalable, and well-calibrated approach to uncertainty
estimation. They require no modifications to the training procedure beyond running
multiple independent training runs, making them easy to adopt in practice.
""",
        ],
    },
    {
        "filename": "_sample_attention.pdf",
        "title": "Attention Is All You Need",
        "pages": [
            """\
Attention Is All You Need

Abstract
We propose the Transformer, a novel network architecture based solely on attention
mechanisms, dispensing with recurrence and convolutions entirely. On machine
translation tasks, the Transformer achieves state-of-the-art results while being
significantly more parallelizable and requiring substantially less training time
than recurrent architectures.

1. Introduction
Recurrent neural networks (RNNs) and their variants (LSTMs, GRUs) have been the
dominant approach for sequence modeling and transduction tasks. However, sequential
computation in RNNs prevents parallelization within training examples and leads to
memory constraints for long sequences.

Attention mechanisms have been used in conjunction with RNNs to allow modeling of
dependencies without regard to their distance in input or output sequences. We take
this idea further: the Transformer relies entirely on a self-attention mechanism
to compute representations, removing recurrence altogether.
""",
            """\
2. Model Architecture

The Transformer follows an encoder-decoder structure.

2.1 Scaled Dot-Product Attention
The attention function maps a query Q and a set of key-value pairs (K, V) to output:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Dividing by sqrt(d_k) prevents the dot products from growing large and pushing the
softmax into regions with tiny gradients.

2.2 Multi-Head Attention
Rather than applying a single attention function, we linearly project Q, K, V h times
into d_k, d_k, d_v dimensions respectively:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Multiple heads allow the model to jointly attend to information from different
representation subspaces at different positions.

2.3 Positional Encoding
Since the model contains no recurrence or convolution, we inject positional
information via sinusoidal positional encodings added to the input embeddings:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
""",
            """\
3. Training and Results

3.1 Training Details
We train on the WMT 2014 English-German dataset (4.5M sentence pairs) and
English-French dataset (36M pairs). Optimization uses Adam with a custom learning
rate schedule that warms up for 4000 steps then decays proportionally to the
inverse square root of the step number.

3.2 Results: Machine Translation
On EN-DE (newstest2014), the Transformer (big) achieves 28.4 BLEU, surpassing
all previously published models, including ensembles. Training cost: 3.5 days on
8 P100 GPUs.

On EN-FR, the big model achieves 41.0 BLEU at 1/4 of the training cost of the
previous best single model.

3.3 Results: English Constituency Parsing
The Transformer generalizes well to other tasks. With only 40K training sentences,
it achieves 91.3 F1 on the WSJ test set, outperforming the Berkeley Parser.

4. Conclusion
The Transformer is the first sequence transduction model based entirely on attention.
It trains significantly faster than architectures based on recurrent or convolutional
layers, and achieves new state of the art on machine translation tasks.
""",
        ],
    },
]


def make_pdf(spec):
    out = PDF_DIR / spec["filename"]
    doc = fitz.open()
    for page_text in spec["pages"]:
        page = doc.new_page()
        page.insert_text((50, 60), page_text, fontsize=10)
    doc.set_metadata({"title": spec["title"]})
    doc.save(out)
    doc.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    for spec in PAPERS:
        make_pdf(spec)
