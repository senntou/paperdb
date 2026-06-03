# JEM の学習方法まとめ

出典: Grathwohl et al., *"Your Classifier is Secretly an Energy Based Model and You Should Treat It Like One"*, ICLR 2020
(`pdf/Grathwohl et al. - 2020 - YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED MODEL AND YOU SHOULD TREAT IT LIKE ONE.pdf`)

## 1. 基本アイデア：分類器を EBM として読み替える

普通の分類器 $f_\theta: \mathbb{R}^D \to \mathbb{R}^K$ はロジット $f_\theta(x)[y]$ を出し、softmax で $p_\theta(y\mid x)$ を定義します（p.2-3）。

JEM の鍵は、**ロジットをそのまま再利用して同時分布のエネルギーモデルを定義する**点です:

$$
p_\theta(x, y) = \frac{\exp\big(f_\theta(x)[y]\big)}{Z(\theta)}, \qquad E_\theta(x,y) = -f_\theta(x)[y]
$$

$y$ を周辺化すると $x$ の（非正規化）密度が得られます:

$$
p_\theta(x) = \sum_y p_\theta(x,y) = \frac{\sum_y \exp\big(f_\theta(x)[y]\big)}{Z(\theta)}
$$

したがって $x$ のエネルギーは LogSumExp で書けます（p.3, Eq.7）:

$$
E_\theta(x) = -\operatorname{LogSumExp}_y\big(f_\theta(x)[y]\big) = -\log \sum_y \exp\big(f_\theta(x)[y]\big)
$$

なお $p_\theta(y\mid x) = p_\theta(x,y)/p_\theta(x)$ を計算すると $Z(\theta)$ が消えて、いつもの softmax に戻ります。つまり**ネットワーク構造を一切変えず**、ロジットの「定数シフトの自由度」を使って密度モデルを仕込んでいます。

## 2. 学習目的：尤度を 2 つに分解

同時尤度を次のように分解します（p.3, Eq.8）:

$$
\log p_\theta(x, y) = \underbrace{\log p_\theta(x)}_{\text{生成側}} + \underbrace{\log p_\theta(y\mid x)}_{\text{識別側}}
$$

- $\log p_\theta(y\mid x)$ → **通常のクロスエントロピー**で最大化（正規化済みなので簡単）
- $\log p_\theta(x)$ → **正規化定数 $Z(\theta)$ が扱えない**ので、勾配を MCMC で近似

この分解で学習することで、最終的に欲しい $p(y\mid x)$ に偏り（bias）を入れずに済むのがポイントです（別の分解だと識別性能が落ちる、p.4 Sec.5.1）。

## 3. $\log p_\theta(x)$ の勾配と SGLD

EBM の対数尤度の勾配は次の形になります（p.2, Eq.2）:

$$
\frac{\partial \log p_\theta(x)}{\partial \theta} = \mathbb{E}_{p_\theta(x')}\!\left[\frac{\partial E_\theta(x')}{\partial \theta}\right] - \frac{\partial E_\theta(x)}{\partial \theta}
$$

第1項の「モデル分布からのサンプル」が必要なので、**SGLD (Stochastic Gradient Langevin Dynamics)** でサンプル $\hat{x}$ を生成します（p.2, Eq.3）:

$$
x_{i+1} = x_i - \frac{\alpha}{2}\frac{\partial E_\theta(x_i)}{\partial x_i} + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \alpha)
$$

実際にはステップ幅 $\alpha$ とノイズの分散を別々に選ぶ（biased sampler だが学習が速い）。さらに **persistent contrastive divergence（リプレイバッファ）** を使い、毎回チェーンを初期化せず計算を節約します（p.4）。

## 4. 学習アルゴリズム（Algorithm 1, p.13）

各イテレーションで:

1. データから $(x, y)$ をサンプル
2. 識別損失: $\;L_{\text{clf}}(\theta) = \mathrm{xent}\big(f_\theta(x), y\big)$
3. SGLD の初期点 $\hat{x}_0$ を、確率 $1-\rho$ でバッファ $B$ から、確率 $\rho$ で一様分布 $U(-1,1)$ から取得
4. $\eta$ ステップの SGLD でサンプル更新（エネルギーは $-\operatorname{LogSumExp}$）:
$$
\hat{x}_t = \hat{x}_{t-1} + \alpha \cdot \frac{\partial\, \operatorname{LogSumExp}_{y'}\big(f_\theta(\hat{x}_{t-1})[y']\big)}{\partial \hat{x}_{t-1}} + \sigma \cdot \mathcal{N}(0, I)
$$
5. 生成損失（Eq.2 の代理）:
$$
L_{\text{gen}}(\theta) = \operatorname{LogSumExp}_{y'}\big(f(x)[y']\big) - \operatorname{LogSumExp}_{y'}\big(f(\hat{x}_t)[y']\big)
$$
（= 実データのエネルギーを下げ、生成サンプルのエネルギーを上げる）
6. 合計損失で更新:
$$
L(\theta) = L_{\text{clf}}(\theta) + L_{\text{gen}}(\theta)
$$

## 5. サンプリングの選択（p.13）

$p_\theta(x)$ からのサンプル取得に 2 通り試しています:

- **方法1**: $y\sim p(y)$ を引いて $p_\theta(x\mid y)$（エネルギー $-f_\theta(x)[y]$）から SGLD → 見た目が綺麗
- **方法2**: 直接 $p_\theta(x)$（エネルギー $-\operatorname{LogSumExp}_y f_\theta(x)[y]$）から SGLD → **識別性能が良い (92.9% vs 91.2% on CIFAR10)**

→ 論文では**方法2を採用**。

## 6. 実装上の主な設定（Appendix A, p.13）

- 最適化: **Adam**、150 epochs、staircase decay スケジュール
- アーキテクチャ: **WideResNet-28-10**（batch normalization なし）
- サンプリング: PCD、各イテレーションで **20 ステップ**の SGLD、確率 0.05 でチェーンを一様ノイズに再初期化
- 前処理: 画像を $[-1, 1]$ にスケール、stddev = 0.03 のガウシアンノイズを付加

## まとめ（一言で）

> **「分類器のロジットを LogSumExp でエネルギーに読み替えるだけで、追加のネットワークなしに同時分布 $p(x,y)$ を学習できる。学習は『クロスエントロピー（識別）』＋『SGLD で生成したサンプルとのエネルギー差（生成）』の和を最小化する。」**
