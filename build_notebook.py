#!/usr/bin/env python3
"""Build diabetic_retinopathy_samd.ipynb (Colab-ready teaching notebook)."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text,
    })

# ---------------------------------------------------------------- title
md("""# 糖尿病網膜症の眼底画像自動分類 ― Colab 実習ノートブック

糖尿病網膜症（diabetic retinopathy; DR）の眼底画像を、深層学習で **5段階の重症度** に自動分類するモデルを構築し、医療AIとしての性能評価までを一通り体験するノートブックです。

**進め方（Google Colab）**
1. メニュー「ランタイム」→「ランタイムのタイプを変更」で **GPU（T4 など）** を選ぶ
2. 「ランタイム」→「すべてのセルを実行」で上から順に実行する（データのダウンロードと学習で 15〜25 分程度）

**このノートブックで行うこと**
- データの読み込みとクラス分布の確認（クラス不均衡の観察）
- EfficientNet-B0 の転移学習による 5 クラス分類（クラス重み付き損失）
- 評価：混同行列・二次重み付き κ（QWK）・紹介要否（referable DR）の ROC/AUC・感度／特異度
- Grad-CAM による判定根拠の可視化

> 注意：これは教育用の演習コードです。実在の承認済み医療機器（SaMD）の性能とは別物であり、ここで得られる数値は「同一データセット内での検証結果」に過ぎません。講義で扱った IDx-DR の治験成績（感度 87.2%・特異度 90.7%）のような前向き検証とは評価条件が根本的に異なります。""")

# ---------------------------------------------------------------- setup
code("""# 必要なライブラリのインストール（Colab では torch は導入済み）
!pip install -q datasets grad-cam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image

import sklearn.metrics
from sklearn.model_selection import train_test_split

import torch
import torchvision

print(f"torch v{torch.__version__}; torchvision v{torchvision.__version__}")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device:', device)""")

# ---------------------------------------------------------------- data
md("""## 1. データセット

Hugging Face で公開されている眼底画像データセット（Kaggle の糖尿病網膜症画像を 256×256 にリサイズしたもの、2,750 枚）を使います。認証不要でそのままダウンロードできます。

ラベルは国際臨床分類（ICDR）に沿った 5 段階に変換します。

| grade | 意味 |
|---|---|
| 0 | 網膜症なし（Healthy） |
| 1 | 軽度非増殖網膜症（Mild） |
| 2 | 中等度非増殖網膜症（Moderate） |
| 3 | 重度非増殖網膜症（Severe） |
| 4 | 増殖網膜症（Proliferative） |

臨床では grade 2 以上（中等度以上）を **referable DR（眼科紹介が必要）** とみなすのが代表的なスクリーニング設計です。""")

code("""from datasets import load_dataset

ds = load_dataset('Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256', split='train')
print(ds)

LABEL2GRADE = {
    'healthy': 0, 'no_dr': 0, 'no dr': 0, 'normal': 0, '0': 0,
    'mild dr': 1, 'mild': 1, '1': 1,
    'moderate dr': 2, 'moderate': 2, '2': 2,
    'severe dr': 3, 'severe': 3, '3': 3,
    'proliferative dr': 4, 'proliferative': 4, 'proliferate dr': 4, 'proliferate': 4, 'pdr': 4, '4': 4,
}

def to_grade(label):
    key = str(label).strip().lower()
    if key not in LABEL2GRADE:
        raise ValueError(f'未知のラベルです: {label!r}（対応表に加えてください）')
    return LABEL2GRADE[key]

images = [ex.convert('RGB') for ex in ds['image']]
grades = np.array([to_grade(lb) for lb in ds['label']])

GRADE_NAMES = ['0: なし', '1: 軽度', '2: 中等度', '3: 重度', '4: 増殖']
counts = np.bincount(grades, minlength=5)
for name, c in zip(GRADE_NAMES, counts):
    print(f'{name}: {c} 枚 ({c / len(grades):.1%})')

plt.figure(figsize=(6, 3.5))
plt.bar(GRADE_NAMES, counts)
plt.ylabel('images')
plt.title('クラス分布（クラス不均衡に注目）')
plt.show()""")

code("""# 層化分割： train 70% / valid 15% / test 15%（grade の比率を各分割で維持）
idx = np.arange(len(grades))
idx_train, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=grades)
idx_valid, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=grades[idx_tmp])
print(f'train {len(idx_train)} / valid {len(idx_valid)} / test {len(idx_test)}')

# サンプル画像の表示
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for g in range(5):
    j = idx_train[grades[idx_train] == g][0]
    axes[g].imshow(images[j])
    axes[g].set_title(GRADE_NAMES[g])
    axes[g].axis('off')
plt.tight_layout()
plt.show()""")

# ---------------------------------------------------------------- preprocessing
md("""## 2. 前処理とデータ拡張

- 入力サイズは EfficientNet-B0 の標準に合わせて 224×224 にリサイズ
- 正規化の平均・標準偏差は、転移学習元の ImageNet 学習時の値に合わせる
- 訓練データには拡張（左右・上下反転、回転、色調の揺らぎ）を入れる。眼底画像は上下左右の反転で病変の意味が変わらないため、反転は安全な拡張です""")

code("""from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ColorJitter(brightness=0.15, contrast=0.15),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class FundusDataset(Dataset):
    def __init__(self, indices, transform):
        self.indices = list(indices)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(images[j]), int(grades[j])

BATCH_SIZE = 32
train_loader = DataLoader(FundusDataset(idx_train, train_transform), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(FundusDataset(idx_valid, eval_transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(FundusDataset(idx_test, eval_transform), batch_size=BATCH_SIZE, shuffle=False)""")

# ---------------------------------------------------------------- model
md("""## 3. モデル：EfficientNet-B0 の転移学習

ImageNet で学習済みの EfficientNet-B0 を土台に、最終層だけを 5 クラス用に付け替えて全体を微調整（fine-tuning）します。

クラス不均衡への対処として、損失関数に **クラス重み**（頻度の逆数に比例）を付け、枚数の少ない重症クラスでの誤りを重く扱います。""")

code("""def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 5)
    return model

model = build_model().to(device)

train_counts = np.bincount(grades[idx_train], minlength=5).astype(np.float64)
class_weights = torch.tensor(train_counts.sum() / (5 * train_counts), dtype=torch.float32).to(device)
print('class weights:', class_weights.cpu().numpy().round(3))

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)""")

# ---------------------------------------------------------------- training
md("""## 4. 学習（検証 QWK で最良モデルを保存）

各エポック後に検証データで **二次重み付き κ 係数（Quadratic Weighted Kappa; QWK）** を計算し、最良の時点の重みを保存します。QWK は「隣の grade への誤り」より「遠い grade への誤り」を重く罰する指標で、順序のある重症度分類の標準的な評価尺度です（APTOS 2019 コンペの公式指標でもあります）。""")

code("""def predict(loader, model):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            y_prob.append(prob)
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

NUM_EPOCHS = 8
PATIENCE = 3
best_qwk, wait, history = -1.0, 0, []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / len(train_loader)

    y_true, y_prob = predict(valid_loader, model)
    qwk = sklearn.metrics.cohen_kappa_score(y_true, y_prob.argmax(axis=1), weights='quadratic')
    history.append({'epoch': epoch + 1, 'train_loss': running_loss, 'valid_qwk': qwk})
    print(history[-1])

    if qwk > best_qwk:
        best_qwk, wait = qwk, 0
        torch.save(model.state_dict(), 'best_model.pth')
        print('  -> best model saved (QWK = %.4f)' % qwk)
    else:
        wait += 1
        if wait >= PATIENCE:
            print('early stopping')
            break

hist = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist['epoch'], hist['train_loss'], marker='o'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('train loss')
ax[1].plot(hist['epoch'], hist['valid_qwk'], marker='o'); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('valid QWK'); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- evaluation
md("""## 5. テストデータでの評価

学習に使っていないテストデータで最終評価します。見るべきものは正解率だけではありません。

- **混同行列**：どの grade 同士を取り違えるか（隣接 grade の誤りが多いのが普通）
- **QWK**：順序を考慮した一致度
- **referable DR（grade ≥ 2）の 2 値判定**：スクリーニングの実務に対応する評価。感度（見逃しの少なさ）と特異度（不要な紹介の少なさ）のトレードオフを ROC 曲線で見る""")

code("""model = build_model()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)

y_true, y_prob = predict(test_loader, model)
y_pred = y_prob.argmax(axis=1)

acc = sklearn.metrics.accuracy_score(y_true, y_pred)
qwk = sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f'test accuracy: {acc:.3f}')
print(f'test QWK     : {qwk:.3f}')
print()
print(pd.DataFrame(sklearn.metrics.classification_report(
    y_true, y_pred, target_names=GRADE_NAMES, output_dict=True)).T.round(3))

cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=['0', '1', '2', '3', '4'])
disp.plot(cmap='Blues')
plt.title('Confusion matrix (test)')
plt.show()""")

code("""# referable DR（grade >= 2）の 2 値評価
y_bin = (y_true >= 2).astype(int)
p_ref = y_prob[:, 2:].sum(axis=1)  # grade 2〜4 の確率の合計

fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_bin, p_ref)
auc = sklearn.metrics.roc_auc_score(y_bin, p_ref)

# 閾値 0.5 のとき
pred_bin = (p_ref >= 0.5).astype(int)
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_bin, pred_bin).ravel()
print(f'閾値 0.5: 感度 {tp / (tp + fn):.3f} / 特異度 {tn / (tn + fp):.3f}')

# Youden の J 統計量が最大になる閾値
j = np.argmax(tpr - fpr)
print(f'Youden 閾値 {thresholds[j]:.3f}: 感度 {tpr[j]:.3f} / 特異度 {1 - fpr[j]:.3f}')
print(f'AUC: {auc:.3f}')

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.legend(); plt.title('Referable DR (grade >= 2)')
plt.show()""")

# ---------------------------------------------------------------- gradcam
md("""## 6. Grad-CAM：モデルはどこを見ているか

分類の根拠となった画像領域を Grad-CAM で可視化します。病変（出血・白斑など）の周辺が熱くなっていれば望ましいですが、**それらしく見えるヒートマップが正しい根拠とは限らない**点も、説明可能性の限界として講義で触れたとおりです。""")

code("""!pip install -q grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# EfficientNet の最後の畳み込みブロックを対象にする
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# テストから grade 2〜4 のサンプルを 3 枚選ぶ
shown = 0
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
model.eval()
for j in idx_test:
    if grades[j] < 2 or shown >= 3:
        continue
    img = images[j]
    x = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.softmax(model(x), dim=1)[0]
    pred_grade = int(pred.argmax().item())
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_grade)])[0]
    rgb = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    axes[shown].imshow(overlay)
    axes[shown].set_title(f'true {grades[j]} -> pred {pred_grade} (p={pred[pred_grade]:.2f})')
    axes[shown].axis('off')
    shown += 1
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- discussion
md("""## 7. 考察：この結果をどう読むか

- テスト成績は **同じデータセットからの分割** に対するものです。撮影機器・施設・人種構成の異なる外部データでは性能が下がるのが普通で（ドメインシフト）、SaMD の審査で外部検証が重視される理由がここにあります。
- ラベル自体に判定者間の揺れがあります。grade 1 と 2 の境界など、専門医同士でも一致しない症例があり、それがモデルの到達可能な上限を決めます。
- スクリーニング用途では「感度を優先して閾値を下げる」設計が一般的です。ROC 曲線上でどの点で運用するかは、見逃しのコストと紹介過多のコストの社会的判断であり、技術だけでは決まりません。
- 実際の製品（IDx-DR / LumineticsCore、EyeArt など）は、特定の眼底カメラ・撮影手順・対象患者をセットで検証し、その範囲でのみ承認されています。「AI の性能」はモデル単体ではなく、撮影から紹介までのワークフロー全体の性能として評価されます。

**発展課題**
1. 閾値を変えたときの感度・特異度・紹介率の変化を表にまとめる
2. EfficientNet-B0 を ResNet18 に変えて QWK と AUC を比較する（去年の OCT 演習との接続）
3. テストデータに意図的にノイズ（ぼかし・明度変化）を加え、性能低下の様子を見る（ドメインシフトの疑似体験）""")

# ---------------------------------------------------------------- appendix
md("""## 付録：APTOS 2019（Kaggle）で学習する場合

講義で紹介した APTOS 2019 は Kaggle からのダウンロードにアカウントと API トークンが必要です。Colab のファイルパネルに `kaggle.json` をアップロードしてから、以下のセルを実行すると同じ形式で読み込めます（未アップロードならセルは自動でスキップされます）。""")

code("""import os, glob

if os.path.exists('kaggle.json'):
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle competitions download -c aptos2019-blindness-detection
    !unzip -q aptos2019-blindness-detection.zip -d aptos2019
    aptos = pd.read_csv('aptos2019/train.csv')
    print(aptos['diagnosis'].value_counts().sort_index())
    print('画像は aptos2019/train_images/ 以下（id_code.png）。')
    print('images / grades をこの形式に作り直せば、以降のセルをそのまま使えます。')
else:
    print('kaggle.json がないためスキップしました（本文のデータセットで実習できます）。')""")

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = "/home/hatch/workspace/dr-samd-colab/diabetic_retinopathy_samd.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written:", out, "cells:", len(cells))
