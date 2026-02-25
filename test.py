import os
import csv
import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import TransformerEncoder
from main import (
    FeatureVectorDataset,
    load_config,
    load_embeddings_npy,
    load_rbp_matrices_csv,
    collate_fn_with_none_rbp,
)

def test(
    model,
    loader,
    device,
    writer,
    config,
    log_dir,
    write_header=False
):
    model.eval()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]
    
    # ラベル名を取得（input_path_test_listから）
    label_names = None
    if "input_path_test_list" in config:
        try:
            test_df = pd.read_csv(config["input_path_test_list"], nrows=1)
            non_label_cols = ['ID', 'ncbi_ID', 'RNALocate_ID', 'Refseq_id', 'RNA_Type', 'Species', 'RNA_Symbol', 'Sequence']
            label_names = [col for col in test_df.columns if col not in non_label_cols]
            print(f"[Info] Label names loaded from test_list: {label_names[:3]}... (total {len(label_names)})")
        except Exception as e:
            print(f"[Warning] Failed to load label names: {e}")
    
    if label_names is None:
        label_names = [f"label_{i}" for i in range(num_labels)]

    all_preds = []
    all_probs = []
    all_labels = []

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

    loop = tqdm(loader, desc="Test")

    with torch.no_grad():
        for batch in loop:
            x, rna_type_tensor, species_tensor, rbp_tensor, y = batch
            x = x.to(device)
            rna_type_tensor = rna_type_tensor.to(device)
            species_tensor = species_tensor.to(device)
            y = y.to(device)

            if rbp_tensor is not None:
                if isinstance(rbp_tensor, list):
                    rbp_tensor = [
                        r.to(device) if r is not None else None for r in rbp_tensor
                    ]
                else:
                    rbp_tensor = rbp_tensor.to(device)

            if isinstance(model, torch.nn.DataParallel):
                out = model.module.compute_loss_and_metrics(
                    x, rna_type_tensor, species_tensor, rbp_tensor, y
                )
            else:
                out = model.compute_loss_and_metrics(
                    x, rna_type_tensor, species_tensor, rbp_tensor, y
                )

            total_loss += out["loss"].item()
            total_acc += out["acc"].item()

            for key in all_metrics:
                all_metrics[key] += (
                    out[key].detach() if isinstance(out[key], torch.Tensor) else out[key]
                )

            all_preds.append(out["y_pred"].detach().cpu())
            all_probs.append(out["y_prob"].detach().cpu())
            all_labels.append(y.detach().cpu())

    # ==========================
    # 集約
    # ==========================
    num_batches = len(loader)
    
    if num_batches == 0:
        print("[Error] No batches to process (empty dataloader)")
        return

    avg_metrics = {
        k: (v / num_batches if isinstance(v, float)
            else v.cpu().numpy() / num_batches)
        for k, v in all_metrics.items()
    }

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # ==========================
    # 確率CSV保存
    # ==========================
    prob_csv_path = os.path.join(log_dir, "test_probs.csv")

    prob_df = pd.DataFrame(
        all_probs.numpy(),
        columns=label_names
    )
    prob_df.to_csv(prob_csv_path, index=False)

    print(f"[Saved] prediction probabilities → {prob_csv_path}")
    
    # ==========================
    # 真ラベルCSV保存（ROC曲線用）
    # ==========================
    labels_csv_path = os.path.join(log_dir, "test_labels.csv")
    labels_df = pd.DataFrame(
        all_labels.numpy(),
        columns=label_names
    )
    labels_df.to_csv(labels_csv_path, index=False)
    
    print(f"[Saved] true labels → {labels_csv_path}")

    # # ==========================
    # # ROC曲線
    # # ==========================
    # for i in range(num_labels):
    #     y_true = all_labels[:, i].numpy()
    #     y_score = all_probs[:, i].numpy()
    #     label_name = label_names[i]

    #     if y_true.sum() == 0:
    #         print(f"[Skip] label {label_name} ({i}) has no positive samples")
    #         continue

    #     fpr, tpr, _ = roc_curve(y_true, y_score)
    #     roc_auc = auc(fpr, tpr)

    #     plt.figure()
    #     plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    #     plt.plot([0, 1], [0, 1], "k--")
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title(f"ROC curve ({label_name})")
    #     plt.legend()

    #     save_path = os.path.join(log_dir, f"roc_{label_name}.png")
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()

    # print("[Saved] ROC curves")

    # ==========================
    # test_log CSV
    # ==========================
    macro_f1 = avg_metrics["f1"].mean()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall + 1e-8)
    )

    if write_header:
        header = ['loss', 'acc', 'macro_f1', 'micro_f1']
        header += [
            "sample_avg_precision", "ranking_loss", "hamming_loss",
            "coverage", "one_error", "sample_accuracy"
        ]
        for key in ['per_label_acc', 'precision', 'recall',
                    'specificity', 'npv', 'f1']:
            header += [f"{key}_{i}" for i in range(num_labels)]
        writer.writerow(header)

    row = [
        total_loss / num_batches,
        total_acc / num_batches,
        macro_f1,
        micro_f1
    ]

    row += [
        avg_metrics[k] for k in
        ["sample_avg_precision", "ranking_loss", "hamming_loss",
         "coverage", "one_error", "sample_accuracy"]
    ]

    for key in ['per_label_acc', 'precision', 'recall',
                'specificity', 'npv', 'f1']:
        row += avg_metrics[key].tolist()

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])



def main():
    config = load_config("config.yaml")
    
    # ===========================
    # model_pathから log_dir と test_log_path を自動設定
    # ===========================
    import re
    model_path = config["model_path"]
    
    # log_dirの親: model_path の checkpoints/ の親ディレクトリ
    if "/checkpoints/" in model_path:
        base_log_dir = model_path.split("/checkpoints/")[0]
    else:
        base_log_dir = os.path.dirname(model_path)
    
    # test_log_path と エポック番号を取得
    match = re.search(r"model_epoch(\d+)\.pt", model_path)
    if match:
        epoch_num = match.group(1)
        config["test_log_path"] = f"test_log_{epoch_num}.csv"
    else:
        epoch_num = "unknown"
        config["test_log_path"] = "test_log_unknown.csv"
    
    # テスト結果用サブディレクトリを作成
    log_dir = os.path.join(base_log_dir, f"test_{epoch_num}")
    config["log_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[Auto Config] base_log_dir: {base_log_dir}")
    print(f"[Auto Config] result_dir (log_dir): {log_dir}")
    print(f"[Auto Config] test_log_path: {config['test_log_path']}")

    df_test = pd.read_csv(config["input_path_test_list"])

    label_cols = df_test.iloc[
        :, config["label_start_index"]:config["feature_start_index"]
    ].columns
    df_test = df_test.dropna(subset=label_cols)

    refseq_ids = df_test["Refseq_id"].tolist()
    rbp_matrices, _ = load_rbp_matrices_csv(
        config["rbp_matrix_dir_reformer"], refseq_ids
    )

    embeddings = load_embeddings_npy(
        config["input_embeddings_dir"],
        df_test["ID"].tolist()
    )

    test_dataset = FeatureVectorDataset(
        df_test, embeddings, rbp_matrices, config
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_with_none_rbp
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerEncoder(
        config=config,
        num_rna_type=len(config["rna_type_list"]),
        num_species=len(config["species_list"]),
        rbp_dim=config["num_rbps"],
    ).to(device)

    model.load_state_dict(
        torch.load(config["model_path"], map_location=device)
    )
    model.eval()

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    test_log_path = os.path.join(log_dir, config["test_log_path"])

    with open(test_log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        for i in range(1):
            print(f"=== Test run {i+1} ===")
            test(
                model=model,
                loader=test_loader,
                device=device,
                writer=writer,
                config=config,
                log_dir=log_dir,
                write_header=(i == 0)
            )


    print("=== All tests finished ===")
    print(f"Saved to: {test_log_path}")


if __name__ == "__main__":
    main()
