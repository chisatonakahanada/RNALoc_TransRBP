# Transformer_RBP

RNA 配列およびRBP 結合サイト情報を使用した、Transformer ベースの RNA の細胞内局在予測モデル。

## データ

データは以下の Google Drive からダウンロードできます。  
https://drive.google.com/drive/folders/1ZOX8geocHbAbyK7ic47sOPfzjQ1jt7FH?usp=drive_link

### 内容

- RNA局在データ（train / valid / test）
- RNA配列のRNAErnieによる embedding
- RBP結合行列（eCLIP 実験データ）
- RBP結合行列（Reformer による予測データ）


## 実行
bash run.sh
