# Transformer_RBP

This repository implements a Transformer-based model for predicting the intracellular localization of RNA using RNA sequence and RBP binding site information.

Data and pre-trained model weights can be downloaded from the following link.
https://drive.google.com/drive/folders/1ZOX8geocHbAbyK7ic47sOPfzjQ1jt7FH?usp=sharing

## Quick Start

### 1. Setup

### 2. Download Data

The data used for training and evaluation can be downloaded from the following link.

```bash
https://drive.google.com/drive/folders/1zNymyjfZVhROodZz1M75lk51AzXEolkd?usp=drive_link
```
#### ① RNA sequences and localizations
- **loc_rna.zip**  
  RNA sequences and their subcellular localization.  
  Unzip the file and save the data in `data/rna_loc`.

- **rna_embeddeings.zip**  
  Embedding of RNA sequences obtained using RNAErnie.  
  Unzip the file and save the data in `data/rna_embeddings`.  

#### ② RBP binding sites

- **rbp_matrix_eclip.zip**  
  RBP binding site matrix based on eCLIP data.  
  Unzip the file and save the data in `data/rbp_matrix_eclip`.  

- **rbp_matrix_reformer.zip**  
  RBP binding site matrix predicted by Reformer.  
  Unzip the file and save the data in `data/rbp_matrix_reformer`.  
  
  
   
### 3. Download model weights
The model weights used can be downloaded from the following link.
```bash
https://drive.google.com/drive/folders/1pfKXi3r1E5GiL2yzF2vqjQC4mdSZ3rbb?usp=drive_link
```   

## Prediction

### 1. Specifying test data and model weights  
Specify the test data and model weights in 'config.yaml'.  
- **input_path_test_list**: Specify the path to the test data
- **rna_type_filter**: Specify the RNA type to use
- **model_path**: Specify the path to the model weights

### 2. Run inference
```bash
python test.py --output-dir-name [dir_name]
```
The inference results are output to 'result/'

## Training


## 設定
config.yaml


