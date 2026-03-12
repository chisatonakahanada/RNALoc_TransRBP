# Transformer_RBP

This repository implements a Transformer-based model for predicting the intracellular localization of RNA using RNA sequence and RBP binding site information.

Data and pre-trained model weights can be downloaded from the following link.
https://drive.google.com/drive/folders/1ZOX8geocHbAbyK7ic47sOPfzjQ1jt7FH?usp=sharing

## Quick Start

## Data
https://drive.google.com/drive/folders/1zNymyjfZVhROodZz1M75lk51AzXEolkd?usp=drive_link

### 内容

- RNA局在データ（train / valid / test）
- RNA配列のRNAErnieによる embedding
- RBP結合行列（eCLIP 実験データ）
- RBP結合行列（Reformer による予測データ）

## 設定
config.yaml

## 実行コマンド
学習の実行：bash run.sh
