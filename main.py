import os
import csv
import json
import torch
import yaml
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TransformerEncoder
from metrics import loss_and_metrics
from pandas.api.types import CategoricalDtype


class FeatureVectorDataset(Dataset):
    def __init__(self, df, embeddings_dict, rbp_matrices_dict, config):
        self.ids = df["ID"].tolist()  
        self.refseq_ids = df["Refseq_id"].tolist()  

        label_df = df.iloc[:, config["label_start_index"]:config["feature_start_index"]]
        label_numeric = label_df.apply(pd.to_numeric, errors='coerce')
        if label_numeric.isnull().values.any():
            print("Warning: NaN detected in label columns. Please check your label data.")
        self.labels = torch.tensor(label_numeric.values, dtype=torch.float32)

        rna_type_list = config["rna_type_list"]
        species_list = config["species_list"]

        # Category -> One-hot
        df["RNA_Type"] = df["RNA_Type"].astype(CategoricalDtype(categories=rna_type_list))
        df["Species"] = df["Species"].astype(CategoricalDtype(categories=species_list))

        self.species_onehot = torch.tensor(
            pd.get_dummies(df["Species"], prefix="Species").astype(float).values,
            dtype=torch.float32
        )
        self.rna_type_onehot = torch.tensor(
            pd.get_dummies(df["RNA_Type"], prefix="RNA_Type").astype(float).values,
            dtype=torch.float32
        )

        self.embeddings_dict = embeddings_dict
        self.rbp_matrices_dict = rbp_matrices_dict
        self.config = config

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]  # エンベディング用のキー（ID）
        refseq_id = self.refseq_ids[idx]  # RBP matrix 用のキー（Refseq_id）

        # 埋め込み (seq_len, emb_dim) - ID をキーとして取得
        embedding = self.embeddings_dict[id_]
        embedding_tensor = torch.from_numpy(embedding).float()

        # RBP 0/1 行列 (seq_len, num_rbps) - Refseq_id をキーとして取得
        rbp_tensor = None
        if refseq_id in self.rbp_matrices_dict and self.rbp_matrices_dict[refseq_id] is not None:
            rbp_mat = self.rbp_matrices_dict[refseq_id]
            rbp_tensor = torch.from_numpy(rbp_mat).float()
        else:
            pass

        rna_type_tensor = self.rna_type_onehot[idx]
        species_tensor = self.species_onehot[idx]
        label_tensor = self.labels[idx]

        return embedding_tensor, rna_type_tensor, species_tensor, rbp_tensor, label_tensor


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_embeddings_npy(directory, ids):
    embeddings = {}
    missing_ids = []
    
    for id_ in tqdm(ids, desc="Loading embeddings", mininterval=10):
        path = os.path.join(directory, f"{id_}.npy")
        if os.path.exists(path):
            try:
                embeddings[id_] = np.load(path)
            except Exception:
                missing_ids.append(id_)
        else:
            missing_ids.append(id_)
    
    loaded = len(embeddings)
    total = len(ids)
    print(f"  Loaded: {loaded}/{total} ({100*loaded/total:.1f}%)")
    
    return embeddings, missing_ids

def load_rbp_matrices_csv(directory, ids, key_name="", num_rbps=None):
    """CSVファイルから RBP マトリックスと ID 情報をロード
    
    Args:
        directory: RBP行列ファイルが格納されたディレクトリ
        ids: ロード対象のIDのリスト（Refseq_idまたはID）
        key_name: ログ用のキー名（例: "eCLIP", "Reformer"）
        num_rbps: RBPの数。150の場合、重複列を除外する
    
    Returns:
        tuple: (matrices_dict, ids_dict)
            - matrices_dict: {id: rbp_matrix_array (seq_len, num_rbps) or None}
            - ids_dict: {id: {"refseq_id": str, "length": int}}
    """
    if not os.path.exists(directory):
        print(f"  Directory not found: {directory}")
        return {}, {}
    
    matrices = {}
    ids_info = {}
    loaded_count = 0
    
    desc = f"Loading RBP ({key_name})" if key_name else "Loading RBP matrices"
    for id_ in tqdm(ids, desc=desc, mininterval=10, disable=len(ids)==0):
        matrices[id_] = None
        path = os.path.join(directory, f"{id_}.csv")
        
        if not os.path.exists(path):
            continue
        
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            
            # num_rbps が 150 の場合、重複する列（前半が一致するもの）を除外
            if num_rbps == 150:
                columns = df.columns.tolist()
                # RBP名（_より前）でグルーピング
                rbp_groups = {}
                for col in columns:
                    rbp_name = col.split('_')[0]  # AARS, AATF, ABCF1, AGGF1 など
                    if rbp_name not in rbp_groups:
                        rbp_groups[rbp_name] = []
                    rbp_groups[rbp_name].append(col)
                
                # 各グループでアルファベット順に最初のものを選択
                selected_columns = []
                for rbp_name in sorted(rbp_groups.keys()):
                    sorted_cols = sorted(rbp_groups[rbp_name])
                    selected_columns.append(sorted_cols[0])  # 最初のものを選択
                
                # 選択された列のみを抽出
                df = df[selected_columns]
            
            rbp_matrix = df.values.astype(np.float32)
            if rbp_matrix.size == 0:
                continue
            
            seq_len, num_rbps_actual = rbp_matrix.shape
            ids_info[id_] = {"refseq_id": str(id_), "length": int(seq_len)}
            matrices[id_] = rbp_matrix
            loaded_count += 1
            
        except Exception:
            continue
    
    if len(ids) > 0:
        print(f"  Loaded: {loaded_count}/{len(ids)} ({100*loaded_count/len(ids):.1f}%)")
    
    return matrices, ids_info

def collate_fn_with_none_rbp(batch):
    """None を含む RBP tensor を処理するカスタム collate 関数
    
    batch: [(embedding, rna_type, species, rbp_tensor or None, label), ...]
    
    Returns: (embedding, rna_type, species, rbp_tensor or None, label)
             rbp_tensor が全て None の場合は None、それ以外は stack したもの
    """
    embeddings = []
    rna_types = []
    species = []
    rbp_tensors = []
    labels = []
    
    for embedding, rna_type, sp, rbp, label in batch:
        embeddings.append(embedding)
        rna_types.append(rna_type)
        species.append(sp)
        rbp_tensors.append(rbp)
        labels.append(label)
    
    embeddings = torch.stack(embeddings)
    rna_types = torch.stack(rna_types)
    species = torch.stack(species)
    labels = torch.stack(labels)
    
    # rbp_tensors: 全て None か、全て tensor か、混在か
    if all(r is None for r in rbp_tensors):
        # 全て None の場合
        rbp_stacked = None
    else:
        # 混在している場合、None は None のままリストに保持
        # （モデルの forward で None チェックが入っているため）
        rbp_stacked = rbp_tensors
    
    return embeddings, rna_types, species, rbp_stacked, labels


def get_dataloaders(df_train, df_valid, embedding_dir, rbp_matrices, config):
    print(f"\n=== Creating DataLoaders ===")
    
    # ID をキーとして embeddings をロード
    train_ids = df_train["ID"].tolist()
    valid_ids = df_valid["ID"].tolist()

    print("Loading train embeddings...")
    train_embeddings, train_missing = load_embeddings_npy(embedding_dir, train_ids)
    print("Loading valid embeddings...")
    valid_embeddings, valid_missing = load_embeddings_npy(embedding_dir, valid_ids)

    # 見つかった ID のみでフィルタリング
    df_train = df_train[df_train["ID"].isin(train_embeddings.keys())]
    df_valid = df_valid[df_valid["ID"].isin(valid_embeddings.keys())]

    train_dataset = FeatureVectorDataset(df_train, train_embeddings, rbp_matrices, config)
    val_dataset = FeatureVectorDataset(df_valid, valid_embeddings, rbp_matrices, config)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_with_none_rbp)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_with_none_rbp)

    print(f"Batches: Train={len(train_loader)}, Valid={len(val_loader)}, Batch size={config['batch_size']}\n")
    
    return train_loader, val_loader


def train(model, loader, optimizer, device, epoch, writer, config):
    model.train()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]

    all_preds, all_labels = [], []
    all_metrics = {
        'sample_avg_precision': 0.0,
        'ranking_loss': 0.0,
        'hamming_loss': 0.0,
        'coverage': 0.0,
        'one_error': 0.0,
        'sample_accuracy': 0.0,
        'per_label_acc': torch.zeros(num_labels, device=device),
        'precision': torch.zeros(num_labels, device=device),
        'recall': torch.zeros(num_labels, device=device),
        'specificity': torch.zeros(num_labels, device=device),
        'npv': torch.zeros(num_labels, device=device),
        'f1': torch.zeros(num_labels, device=device),
    }

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch in loop:
        x, rna_type_tensor, species_tensor, rbp_tensor, y = batch
        x = x.to(device)
        rna_type_tensor = rna_type_tensor.to(device)
        species_tensor = species_tensor.to(device)
        y = y.to(device)
        
        # rbp_tensor の処理: None / tensor / list のいずれか
        if rbp_tensor is not None:
            if isinstance(rbp_tensor, list):
                rbp_tensor = [r.to(device) if r is not None else None for r in rbp_tensor]
            else:
                rbp_tensor = rbp_tensor.to(device)

        optimizer.zero_grad()

        if isinstance(model, torch.nn.DataParallel):
            out = model.module.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)
        else:
            out = model.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)

        loss = out["loss"]
        acc = out["acc"]
        y_pred = out["y_pred"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

        for key in all_metrics:
            if isinstance(out[key], torch.Tensor):
                all_metrics[key] += out[key].detach()
            else:
                all_metrics[key] += out[key]

        all_preds.append(y_pred.detach().cpu())
        all_labels.append(y.detach().cpu())

        loop.set_postfix(loss=total_loss / len(all_preds), acc=total_acc / len(all_preds))

    num_batches = len(loader)
    avg_metrics = {k: (v / num_batches if isinstance(v, float) else v.cpu().numpy() / num_batches)
                   for k, v in all_metrics.items()}
    macro_f1 = avg_metrics["f1"].mean()

    # Micro F1
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    if epoch == 0:
        header = ['epoch', 'loss', 'acc', 'macro_f1', 'micro_f1']
        header += ["sample_avg_precision", "ranking_loss", "hamming_loss", "coverage", "one_error", "sample_accuracy"]
        for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
            header += [f"{key}_{i}" for i in range(num_labels)]
        writer.writerow(header)

    row = [epoch, total_loss / num_batches, total_acc / num_batches, macro_f1, micro_f1]
    row += [avg_metrics[k] for k in ["sample_avg_precision", "ranking_loss", "hamming_loss",
                                     "coverage", "one_error", "sample_accuracy"]]
    for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
        row += avg_metrics[key].tolist()

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def validate(model, loader, device, epoch, writer, config):
    model.eval()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]

    all_preds, all_labels = [], []
    all_metrics = {
        'sample_avg_precision': 0.0,
        'ranking_loss': 0.0,
        'hamming_loss': 0.0,
        'coverage': 0.0,
        'one_error': 0.0,
        'sample_accuracy': 0.0,
        'per_label_acc': torch.zeros(num_labels, device=device),
        'precision': torch.zeros(num_labels, device=device),
        'recall': torch.zeros(num_labels, device=device),
        'specificity': torch.zeros(num_labels, device=device),
        'npv': torch.zeros(num_labels, device=device),
        'f1': torch.zeros(num_labels, device=device),
    }

    loop = tqdm(loader, desc=f"Val Epoch {epoch}")

    with torch.no_grad():
        for batch in loop:
            x, rna_type_tensor, species_tensor, rbp_tensor, y = batch
            x = x.to(device)
            rna_type_tensor = rna_type_tensor.to(device)
            species_tensor = species_tensor.to(device)
            y = y.to(device)
            
            # rbp_tensor の処理: None / tensor / list のいずれか
            if rbp_tensor is not None:
                if isinstance(rbp_tensor, list):
                    rbp_tensor = [r.to(device) if r is not None else None for r in rbp_tensor]
                else:
                    # Tensor の場合
                    rbp_tensor = rbp_tensor.to(device)

            if isinstance(model, torch.nn.DataParallel):
                out = model.module.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)
            else:
                out = model.compute_loss_and_metrics(x, rna_type_tensor, species_tensor, rbp_tensor, y)

            loss = out["loss"]
            acc = out["acc"]
            y_pred = out["y_pred"]

            total_loss += loss.item()
            total_acc += acc.item()

            for key in all_metrics:
                if isinstance(out[key], torch.Tensor):
                    all_metrics[key] += out[key].detach()
                else:
                    all_metrics[key] += out[key]

            all_preds.append(y_pred.detach().cpu())
            all_labels.append(y.detach().cpu())

            loop.set_postfix(loss=total_loss / len(all_preds), acc=total_acc / len(all_preds))

    num_batches = len(loader)
    avg_metrics = {k: (v / num_batches if isinstance(v, float) else v.cpu().numpy() / num_batches)
                   for k, v in all_metrics.items()}
    macro_f1 = avg_metrics["f1"].mean()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    if epoch == 0:
        header = ['epoch', 'loss', 'acc', 'macro_f1', 'micro_f1']
        header += ["sample_avg_precision", "ranking_loss", "hamming_loss", "coverage", "one_error", "sample_accuracy"]
        for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
            header += [f"{key}_{i}" for i in range(num_labels)]
        writer.writerow(header)

    row = [epoch, total_loss / num_batches, total_acc / num_batches, macro_f1, micro_f1]
    row += [avg_metrics[k] for k in ["sample_avg_precision", "ranking_loss", "hamming_loss",
                                     "coverage", "one_error", "sample_accuracy"]]
    for key in ['per_label_acc', 'precision', 'recall', 'specificity', 'npv', 'f1']:
        row += avg_metrics[key].tolist()

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def main(log_dir):
    config = load_config("config.yaml")

    print("=== GPU環境確認 ===")
    print(f"CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}'")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    print("=== パラメーター ===")
    print(f"num_heads: {config['num_heads']}")
    print(f"num_layers: {config['num_layers']}")
    print(f"lr: {config['lr']}")
    print(f"input_data_train: {config['input_path_train_list']}")
    print(f"Batch size: {config['batch_size']}")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\n=== Loading Data Files ===")
    df_train = pd.read_csv(config["input_path_train_list"])
    df_valid = pd.read_csv(config["input_path_valid_list"])
    
    embeddings_dir = config["input_embeddings_dir"]
    label_cols = df_train.iloc[:, config["label_start_index"]:config["feature_start_index"]].columns

    df_train = df_train.dropna(subset=label_cols)
    df_valid = df_valid.dropna(subset=label_cols)

    # RNA_Type フィルタリング
    print(f"\n=== RNA Type Filtering ===")
    rna_type_filter = config.get("rna_type_filter", "ALL")
    print(f"Filter: {rna_type_filter}")
    if rna_type_filter != "ALL":
        if rna_type_filter not in config["rna_type_list"]:
            raise ValueError(
                f"Invalid rna_type_filter '{rna_type_filter}'. "
                f"Expected one of {config['rna_type_list']} or 'ALL'."
            )
        df_train_before_filter = len(df_train)
        df_valid_before_filter = len(df_valid)
        df_train = df_train[df_train["RNA_Type"] == rna_type_filter]
        df_valid = df_valid[df_valid["RNA_Type"] == rna_type_filter]
        print(f"After filtering by RNA_Type='{rna_type_filter}':")
        print(f"  Train: {df_train_before_filter} → {len(df_train)} (filtered: {df_train_before_filter - len(df_train)})")
        print(f"  Valid: {df_valid_before_filter} → {len(df_valid)} (filtered: {df_valid_before_filter - len(df_valid)})")
    else:
        print(f"No RNA_Type filtering applied")

    # RBP 行列をロード
    print(f"\n=== Loading RBP Matrices ===")
    
    # Refseq_id と ID のマッピングを作成
    train_refseq_to_id = dict(zip(df_train["Refseq_id"], df_train["ID"]))
    valid_refseq_to_id = dict(zip(df_valid["Refseq_id"], df_valid["ID"]))
    
    train_refseq_ids = df_train["Refseq_id"].tolist()
    valid_refseq_ids = df_valid["Refseq_id"].tolist()
    
    # Train用: eCLIP (Refseq_id) → Reformer (ID) の順でロード
    print("Train: Loading eCLIP (key=Refseq_id)...")
    train_rbp_matrices_eclip, train_rbp_ids_info_eclip = load_rbp_matrices_csv(
        config["rbp_matrix_dir_eclip"],
        train_refseq_ids,
        "eCLIP",
        num_rbps=config.get("num_rbps")
    )
    
    # eCLIPで見つからなかったもののIDを取得
    train_missing_refseq = [rid for rid in train_refseq_ids 
                            if rid not in train_rbp_matrices_eclip or train_rbp_matrices_eclip[rid] is None]
    train_missing_ids = [train_refseq_to_id[rid] for rid in train_missing_refseq if rid in train_refseq_to_id]
    
    # eCLIPで見つからなかったものをReformer (ID) から取得
    print(f"Train: Loading Reformer (key=ID) for {len(train_missing_ids)} missing samples...")
    train_rbp_matrices_reformer_by_id, train_rbp_ids_info_reformer_by_id = load_rbp_matrices_csv(
        config["rbp_matrix_dir_reformer"],
        train_missing_ids,
        "Reformer",
        num_rbps=config.get("num_rbps")
    )
    
    # IDからRefseq_idへのマッピングを作成
    train_id_to_refseq = {v: k for k, v in train_refseq_to_id.items()}
    
    # Reformerの結果をRefseq_idキーに変換
    train_rbp_matrices_reformer = {}
    train_rbp_ids_info_reformer = {}
    for id_, matrix in train_rbp_matrices_reformer_by_id.items():
        if id_ in train_id_to_refseq:
            refseq_id = train_id_to_refseq[id_]
            train_rbp_matrices_reformer[refseq_id] = matrix
            if id_ in train_rbp_ids_info_reformer_by_id:
                train_rbp_ids_info_reformer[refseq_id] = train_rbp_ids_info_reformer_by_id[id_]
    
    # Train用のマトリックスをマージ（eCLIP優先、キーはRefseq_id）
    train_rbp_matrices = {}
    train_rbp_ids_info = {}
    for rid in train_refseq_ids:
        if rid in train_rbp_matrices_eclip and train_rbp_matrices_eclip[rid] is not None:
            train_rbp_matrices[rid] = train_rbp_matrices_eclip[rid]
            train_rbp_ids_info[rid] = train_rbp_ids_info_eclip[rid]
        elif rid in train_rbp_matrices_reformer and train_rbp_matrices_reformer[rid] is not None:
            train_rbp_matrices[rid] = train_rbp_matrices_reformer[rid]
            train_rbp_ids_info[rid] = train_rbp_ids_info_reformer[rid]
        else:
            train_rbp_matrices[rid] = None
    
    # Valid用: Reformer (ID) からのみロード
    valid_ids = df_valid["ID"].tolist()
    print(f"Valid: Loading Reformer (key=ID)...")
    valid_rbp_matrices_by_id, valid_rbp_ids_info_by_id = load_rbp_matrices_csv(
        config["rbp_matrix_dir_reformer"],
        valid_ids,
        "Reformer",
        num_rbps=config.get("num_rbps")
    )
    
    # ValidのIDからRefseq_idへのマッピング
    valid_id_to_refseq = {v: k for k, v in valid_refseq_to_id.items()}
    
    # ValidのReformer結果をRefseq_idキーに変換
    valid_rbp_matrices = {}
    valid_rbp_ids_info = {}
    for id_, matrix in valid_rbp_matrices_by_id.items():
        if id_ in valid_id_to_refseq:
            refseq_id = valid_id_to_refseq[id_]
            valid_rbp_matrices[refseq_id] = matrix
            if id_ in valid_rbp_ids_info_by_id:
                valid_rbp_ids_info[refseq_id] = valid_rbp_ids_info_by_id[id_]
    
    # 全体のRBPマトリックスを統合（キーはRefseq_id）
    rbp_matrices = {**train_rbp_matrices, **valid_rbp_matrices}
    rbp_ids_info = {**train_rbp_ids_info, **valid_rbp_ids_info}

    # 統計表示
    train_from_eclip = [rid for rid in train_refseq_ids 
                        if rid in train_rbp_matrices_eclip and train_rbp_matrices_eclip[rid] is not None]
    train_from_reformer = [rid for rid in train_refseq_ids 
                           if rid not in train_from_eclip and rid in train_rbp_matrices_reformer 
                           and train_rbp_matrices_reformer[rid] is not None]
    train_with_data = len(train_from_eclip) + len(train_from_reformer)
    valid_with_data = len([rid for rid in valid_refseq_ids if rid in valid_rbp_matrices and valid_rbp_matrices[rid] is not None])
    
    print(f"\nRBP Statistics:")
    print(f"  Train: {train_with_data}/{len(train_refseq_ids)} ({100*train_with_data/len(train_refseq_ids):.1f}%) - eCLIP:{len(train_from_eclip)}, Reformer:{len(train_from_reformer)}")
    print(f"  Valid: {valid_with_data}/{len(valid_refseq_ids)} ({100*valid_with_data/len(valid_refseq_ids):.1f}%)\n")

    train_loader, val_loader = get_dataloaders(df_train, df_valid, embeddings_dir, rbp_matrices, config)

    print("=== Model Initialization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TransformerEncoder(
        config=config,
        num_rna_type=len(config["rna_type_list"]),
        num_species=len(config["species_list"]),
        rbp_dim=config["num_rbps"],
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    if torch.cuda.device_count() > 0 and config["batch_size"] % torch.cuda.device_count() != 0:
        print(f"Warning: Batch size ({config['batch_size']}) is not divisible by GPU count ({torch.cuda.device_count()})")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print(f"Optimizer: Adam (lr={config['lr']})")

    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_log_path = os.path.join(log_dir, "train_log.csv")
    val_log_path = os.path.join(log_dir, "val_log.csv")

    print(f"\n=== Training Configuration ===")
    print(f"Log directory: {log_dir}")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Model save interval: {config['model_save_interval']}")
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(config["max_epochs"]):
        with open(train_log_path, 'a', newline='') as f_train, open(val_log_path, 'a', newline='') as f_val:
            train_writer = csv.writer(f_train, delimiter='\t')
            val_writer = csv.writer(f_val, delimiter='\t')

            train(model, train_loader, optimizer, device, epoch, train_writer, config)
            validate(model, val_loader, device, epoch, val_writer, config)

        if (epoch + 1) % config["model_save_interval"] == 0:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))
            print(f"Model saved to model_epoch{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/default_run")
    args = parser.parse_args()
    main(args.log_dir)
