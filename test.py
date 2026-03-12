import os
import csv
import argparse
from datetime import datetime
import torch
import yaml
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, coverage_error

from model import TransformerEncoder
from main import (
    FeatureVectorDataset,
    load_config,
    load_embeddings_npy,
    load_rbp_matrices_csv,
    collate_fn_with_none_rbp,
)

from metrics import label_metrics, sample_metrics


def test(model, loader, device, writer, config, log_dir, write_header=False):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss, total_acc = 0, 0
    num_labels = config["num_labels"]
    
    # ラベル名を取得（input_path_test_listから）
    label_names = None
    if "input_path_test_list" in config:
        try:
            test_df = pd.read_csv(config["input_path_test_list"], nrows=1)
            non_label_cols = ['ID', 'RNALocate_ID', 'Refseq_id', 'RNA_Type', 'Species', 'RNA_Symbol']
            label_names = [col for col in test_df.columns if col not in non_label_cols]
            print(f"[Info] Label names loaded from test_list: {label_names[:3]}... (total {len(label_names)})")
        except Exception as e:
            print(f"[Warning] Failed to load label names: {e}")
    
    if label_names is None:
        label_names = [f"label_{i}" for i in range(num_labels)]

    rna_type_filter = config.get("rna_type_filter", "ALL")
    if isinstance(rna_type_filter, str):
        rna_type_filter = rna_type_filter.strip()

    selected_label_sets = {
        "mRNA": [
            "Axon", "Cell body", "Chromatin", "Cytoplasm", "Cytosol",
            "Cytosolsub", "Endoplasmic reticulum", "Extracellular exosome",
            "Extracellular vesicle", "Membrane", "Microvesicle",
            "Mitochondrion", "Neurite", "Nucleolus", "Nucleoplasm",
            "Nucleus", "Nucleussub", "Pseudopodium", "Ribosomal partner",
            "Ribosome",
        ],
        "lncRNA": [
            "Chromatin", "Cytoplasm", "Cytosol",
            "Extracellular exosome", "Extracellular vesicle", "Membrane",
            "Mitochondrion", "Nucleolus", "Nucleoplasm",
            "Nucleus", "Ribosome",
        ],
        "miRNA": [
            "Axon", "Cytoplasm", "Exomere", "Extracellular exosome",
            "Extracellular vesicle", "Microvesicle", "Mitochondrion",
            "Nucleus", "Supermere",
        ],
    }

    if rna_type_filter in selected_label_sets:
        subset_labels = [l for l in label_names if l in set(selected_label_sets[rna_type_filter])]
        if len(subset_labels) > 0:
            metric_label_names = subset_labels
        else:
            metric_label_names = label_names
    else:
        metric_label_names = label_names

    metric_label_indices = [label_names.index(l) for l in metric_label_names]

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
                    rbp_tensor = [r.to(device) if r is not None else None for r in rbp_tensor]
                else:
                    rbp_tensor = rbp_tensor.to(device)

            logits = model(x, rna_type_tensor, species_tensor, rbp_tensor)

            loss = loss_fn(logits, y.float())
            y_prob = torch.sigmoid(logits)

            gamma = 0.2
            y_prob = y_prob ** gamma
            y_pred = (y_prob > 0.5).float()

            y_for_metrics = y[:, metric_label_indices]
            logits_for_metrics = logits[:, metric_label_indices]
            y_pred_for_metrics = y_pred[:, metric_label_indices]

            label_dict = label_metrics(y_for_metrics, y_pred_for_metrics)
            sample_dict = sample_metrics(y_for_metrics, logits_for_metrics)

            full_label_dict = {}
            for k, v in label_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.size(0) == len(metric_label_indices):
                    full = torch.zeros(num_labels, device=device)
                    full[metric_label_indices] = v
                    full_label_dict[k] = full
                else:
                    full_label_dict[k] = v

            out = {
                'loss': loss,
                'y_hat': logits,
                'y_prob': y_prob,
                'y_pred': y_pred,
                **full_label_dict,
                **sample_dict,
            }

            total_loss += out["loss"].item()
            total_acc += out["acc"].item()

            for key in all_metrics:
                all_metrics[key] += (
                    out[key].detach() if isinstance(out[key], torch.Tensor) else out[key]
                )

            all_preds.append(out["y_pred"].detach().cpu())
            all_probs.append(out["y_prob"].detach().cpu())
            all_labels.append(y.detach().cpu())

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
    # 真ラベルCSV保存
    # ==========================
    labels_csv_path = os.path.join(log_dir, "test_labels.csv")
    labels_df = pd.DataFrame(
        all_labels.numpy(),
        columns=label_names
    )
    labels_df.to_csv(labels_csv_path, index=False)
    
    print(f"[Saved] true labels → {labels_csv_path}")

    # ==========================
    # test_log CSV保存
    # ==========================
    subset_indices = metric_label_indices
    if len(subset_indices) > 0 and len(subset_indices) != num_labels:
        f1_subset = avg_metrics["f1"][subset_indices]
        macro_f1 = f1_subset.mean()

        all_preds_subset = all_preds[:, subset_indices]
        all_labels_subset = all_labels[:, subset_indices]
        tp = ((all_preds_subset == 1) & (all_labels_subset == 1)).sum().item()
        fp = ((all_preds_subset == 1) & (all_labels_subset == 0)).sum().item()
        fn = ((all_preds_subset == 0) & (all_labels_subset == 1)).sum().item()
    else:
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
    parser = argparse.ArgumentParser(description="Run test and save outputs")
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default=None,
        help="Directory name under result/. If omitted, timestamp is used.",
    )
    args = parser.parse_args()

    config = load_config("config.yaml")

    # ===========================
    # 出力先ディレクトリを result/ 配下に固定
    # ディレクトリ名は --output-dir-name、未指定時は timestamp
    # ===========================
    run_name = args.output_dir_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_base_dir = os.path.join(script_dir, "result")
    os.makedirs(result_base_dir, exist_ok=True)

    log_dir = os.path.join(result_base_dir, run_name)
    if os.path.exists(log_dir):
        suffix = 1
        while os.path.exists(f"{log_dir}_{suffix}"):
            suffix += 1
        log_dir = f"{log_dir}_{suffix}"

    config["log_dir"] = log_dir
    config["test_log_path"] = "test_result.csv"
    os.makedirs(log_dir, exist_ok=False)

    print(f"[Output] result_dir: {log_dir}")
    print(f"[Output] test_result_path: {os.path.join(log_dir, config['test_log_path'])}")

    df_test = pd.read_csv(config["input_path_test_list"])

    label_cols = df_test.iloc[
        :, config["label_start_index"]:config["feature_start_index"]
    ].columns
    df_test = df_test.dropna(subset=label_cols)

    # RNA_Type フィルタリング（main.py と同じロジック）
    print(f"\n=== RNA Type Filtering ===")
    rna_type_filter = config.get("rna_type_filter", "ALL")
    print(f"Filter: {rna_type_filter}")
    if rna_type_filter != "ALL":
        if rna_type_filter not in config.get("rna_type_list", []):
            raise ValueError(
                f"Invalid rna_type_filter '{rna_type_filter}'. "
                f"Expected one of {config.get('rna_type_list', [])} or 'ALL'."
            )
        before_count = len(df_test)
        df_test = df_test[df_test["RNA_Type"] == rna_type_filter]
        after_count = len(df_test)
        print(f"After filtering by RNA_Type='{rna_type_filter}': {before_count} -> {after_count} (filtered: {before_count - after_count})")
    else:
        print("No RNA_Type filtering applied")

    refseq_ids = df_test["Refseq_id"].tolist()
    rbp_matrices, _ = load_rbp_matrices_csv(
        config["rbp_matrix_dir_reformer"], refseq_ids
    )

    embeddings, _ = load_embeddings_npy(
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

# python test.py --output-dir-name [dir_name]
