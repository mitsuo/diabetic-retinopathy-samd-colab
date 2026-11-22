# 糖尿病網膜症の眼底画像自動分類 ― Colab 実習ノートブック

糖尿病網膜症（diabetic retinopathy）の眼底画像を深層学習で 5 段階の重症度に自動分類し、医療AIとしての性能評価（混同行列・二次重み付き κ・ROC/AUC・感度／特異度・Grad-CAM）までを行う、Google Colab 用の実習ノートブックです。大学院講義「医療DX・AI特論」（糖尿病網膜症の眼底画像を自動分類する SaMD の回）の演習用に作成しました。

## Colab で開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitsuo/diabetic-retinopathy-samd-colab/blob/main/diabetic_retinopathy_samd.ipynb)

1. Colab で「ランタイム」→「ランタイムのタイプを変更」→ GPU（T4 など）を選択
2. 「ランタイム」→「すべてのセルを実行」（データのダウンロードと学習で 15〜25 分程度）

## 内容

- データ：Hugging Face 公開データセット（Kaggle の糖尿病網膜症画像を 256×256 にリサイズしたもの、2,750 枚、5 段階ラベル）。認証不要でダウンロード可能
- モデル：EfficientNet-B0 の転移学習、クラス重み付き交差エントロピー損失
- 学習：検証データの二次重み付き κ（QWK）で最良モデルを保存、早期終了あり
- 評価：混同行列、QWK、紹介要否（referable DR, grade ≥ 2）の ROC/AUC、閾値 0.5 と Youden 閾値での感度・特異度
- Grad-CAM による判定根拠の可視化
- 付録：APTOS 2019（Kaggle）をデータに使う手順

## 注意

- 教育用の演習コードです。実在の承認済み医療機器（SaMD）の性能とは別物であり、得られる数値は同一データセット内での検証結果に過ぎません
- データセットの出典とライセンスは配布元（Hugging Face / Kaggle）の条件に従います

## ファイル

- `diabetic_retinopathy_samd.ipynb` ― ノートブック本体
- `diabetic_retinopathy_samd_executed.ipynb` ― 実行結果入りのノートブック（QUICK_MODE での確認実行、2026-11-23）。実行前に出力結果を確認したい場合はこちらを参照
  - 参考値（QUICK_MODE：土台凍結・訓練データ半分・5 エポック）：テスト QWK 0.687、referable DR の AUC 0.908、閾値 0.5 で感度 0.903・特異度 0.738
- `build_notebook.py` ― ノートブックを生成するスクリプト
