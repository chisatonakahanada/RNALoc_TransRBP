import os
import csv
import argparse
from datetime import datetime

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TransformerEncoder
from load_data import (
    FeatureVectorDataset,
    load_config,
    load_embeddings_npy,
    load_rbp_matrices_csv,
    collate_fn_with_none_rbp,
)

from metrics import label_metrics, sample_metrics


def ensure_list(x, n):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().tolist()
    elif isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [float(x)] * n


def run_inference(model, loader, device, label_names, log_dir):

    model.eval()

    all_probs = []
    all_labels = []

    loop = tqdm(loader, desc="Inference")

    with torch.no_grad():
        for batch in loop:

            x, rna_type_tensor, species_tensor, rbp_tensor, y = batch

            x = x.to(device)
            rna_type_tensor = rna_type_tensor.to(device)
            species_tensor = species_tensor.to(device)

            if rbp_tensor is not None:
                if isinstance(rbp_tensor, list):
                    rbp_tensor = [
                        r.to(device) if r is not None else None
                        for r in rbp_tensor
                    ]
                else:
                    rbp_tensor = rbp_tensor.to(device)

            logits = model(x, rna_type_tensor, species_tensor, rbp_tensor)

            y_prob = torch.sigmoid(logits)

            all_probs.append(y_prob.cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    prob_path = os.path.join(log_dir, "test_probs.csv")
    label_path = os.path.join(log_dir, "test_labels.csv")

    pd.DataFrame(all_probs.numpy(), columns=label_names).to_csv(prob_path, index=False)
    pd.DataFrame(all_labels.numpy(), columns=label_names).to_csv(label_path, index=False)

    print(f"[Saved] {prob_path}")
    print(f"[Saved] {label_path}")

    return prob_path, label_path


def evaluate(prob_path, label_path,
             label_names, metric_label_names, metric_label_indices,
             writer):

    y_true_df = pd.read_csv(label_path)
    y_prob_df = pd.read_csv(prob_path)

    y_true_np = y_true_df[label_names].values.astype("float32")
    y_prob_np = y_prob_df[label_names].values.astype("float32")

    # ===== ラベルサブセット適用 =====
    y_true_np = y_true_np[:, metric_label_indices]
    y_prob_np = y_prob_np[:, metric_label_indices]

    y_true = torch.from_numpy(y_true_np)
    y_prob = torch.from_numpy(y_prob_np)

    # gamma補正
    gamma = 0.2
    y_prob = y_prob ** gamma

    y_pred = (y_prob > 0.5).float()

    eps = 1e-7
    y_prob_clamped = y_prob.clamp(min=eps, max=1 - eps)
    y_logits = torch.log(y_prob_clamped / (1 - y_prob_clamped))

    label_m = label_metrics(y_true, y_pred)
    sample_m = sample_metrics(y_true, y_logits)

    from sklearn.metrics import f1_score

    macro_f1 = f1_score(y_true_np, y_pred.numpy(), average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_np, y_pred.numpy(), average="micro", zero_division=0)

    num_labels = len(metric_label_names)

    header = [
        "macro_f1",
        "micro_f1",
        "sample_avg_precision",
        "ranking_loss",
        "hamming_loss",
        "coverage",
        "one_error",
        "sample_accuracy",
    ]

    for key in ["acc","precision","recall","specificity","npv","f1"]:
        header += [f"{key}_{l}" for l in metric_label_names]

    writer.writerow(header)

    row = [
        macro_f1,
        micro_f1,
        sample_m["sample_avg_precision"],
        sample_m["ranking_loss"],
        sample_m["hamming_loss"],
        sample_m["coverage"],
        sample_m["one_error"],
        sample_m["sample_accuracy"],
    ]

    for key in ["acc","precision","recall","specificity","npv","f1"]:
        row += ensure_list(label_m[key], num_labels)

    writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir-name", type=str, default=None)
    args = parser.parse_args()

    config = load_config("config.yaml")

    run_name = args.output_dir_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "result_2", run_name)
    os.makedirs(result_dir)

    print(f"[Output] {result_dir}")

    df_test = pd.read_csv(config["input_path_test_list"])

    label_cols = df_test.iloc[
        :, config["label_start_index"]:config["feature_start_index"]
    ].columns

    label_names = label_cols.tolist()

    df_test = df_test.dropna(subset=label_cols)

    # ===============================
    # RNA_Type フィルタ
    # ===============================
    print("\n=== RNA Type Filtering ===")
    rna_type_filter = config.get("rna_type_filter", "ALL")

    if rna_type_filter != "ALL":
        before = len(df_test)
        df_test = df_test[df_test["RNA_Type"] == rna_type_filter]
        print(f"{before} -> {len(df_test)}")
    else:
        print("No filtering")

    # ===============================
    # ラベルフィルタ
    # ===============================
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
        subset = [l for l in label_names if l in selected_label_sets[rna_type_filter]]
        metric_label_names = subset if len(subset) > 0 else label_names
    else:
        metric_label_names = label_names

    metric_label_indices = [label_names.index(l) for l in metric_label_names]

    print(f"[Label Filter] {len(metric_label_names)} labels")
    
    ids = df_test["ID"].tolist()
    refseq_ids = df_test["Refseq_id"].tolist()

    # ===============================
    # RBP (Reformer)
    # ===============================
    print("\n=== RBP Loading ===")

    rbp_matrices, _ = load_rbp_matrices_csv(
        config["rbp_matrix_dir_reformer"],
        refseq_ids,
        "Reformer",
        num_rbps=config.get("num_rbps")
    )

    # ===============================
    # embeddings
    # ===============================
    print("\n=== Embeddings Loading  ===")
    embeddings, _ = load_embeddings_npy(
        config["input_embeddings_dir"],
        ids
    )

    dataset = FeatureVectorDataset(
        df_test,
        embeddings,
        rbp_matrices,
        config
    )

    loader = DataLoader(
        dataset,
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

    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    prob_path, label_path = run_inference(
        model,
        loader,
        device,
        label_names,
        result_dir
    )

    result_path = os.path.join(result_dir, "test_result.csv")

    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        evaluate(
            prob_path,
            label_path,
            label_names,
            metric_label_names,
            metric_label_indices,
            writer
        )

    print("=== Finished ===")
    print(result_path)


if __name__ == "__main__":
    main()
