import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader,Subset
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────
# MPI-INF-3DHP CKA / Representation-Stage Diagnostic Analysis v4
#
# Purpose:
#   1) Use Hybrid Block OUTPUT representations for CKA.
#   2) Keep the current manuscript Stage partition as a REFERENCE,
#      not an automatically replaced partition.
#   3) Diagnose persistent low-change regions from CKA(L0, Ll).
#   4) Diagnose current Representative Blocks using:
#        R = intra-stage representativeness
#        T = transition salience
#        N = representation novelty
#        Local = local-neighbor CKA (redundancy indicator)
#   5) Restrict representative candidates to layers retained by
#      the current MPI student structure:
#        retained = L0-L4 + L11-L15
#
# IMPORTANT:
#   This script does NOT modify pruning/KD/Warm-up experiments.
#   It is an analysis/diagnostic script only.
# ─────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from utils.tools import get_config, set_random_seed
from utils.learning import load_model
from data.reader.motion_dataset import Fusion
from utils.utils_3dhp import get_variable


def compute_hsic(K, L):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return torch.trace(KH @ LH) / ((n - 1) ** 2)


def compute_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = compute_hsic(K, L)
    hsic_kk = compute_hsic(K, K)
    hsic_ll = compute_hsic(L, L)

    denom = torch.sqrt(hsic_kk) * torch.sqrt(hsic_ll)
    if denom.item() <= 1e-12:
        return 0.0

    return (hsic_kl / denom).item()


def mean_pairwise_cka(cka_matrix, layers):
    layers = list(layers)
    values = []
    for i, li in enumerate(layers):
        for lj in layers[i + 1:]:
            values.append(float(cka_matrix[li, lj]))
    if not values:
        return 1.0
    return float(np.mean(values))


def local_neighbor_cka(cka_matrix, layer, n_layers):
    vals = []
    if layer > 0:
        vals.append(float(cka_matrix[layer - 1, layer]))
    if layer < n_layers - 1:
        vals.append(float(cka_matrix[layer, layer + 1]))
    return float(np.mean(vals)) if vals else 1.0


def transition_salience(global_cka, layer):
    """Max local change around a layer using CKA(L0, Ll)."""
    n = len(global_cka)
    vals = []
    if layer > 0:
        vals.append(abs(float(global_cka[layer] - global_cka[layer - 1])))
    if layer < n - 1:
        vals.append(abs(float(global_cka[layer + 1] - global_cka[layer])))
    return float(max(vals)) if vals else 0.0


def novelty_score(global_cka, layer):
    """Novelty relative to the initial representation: 1 - CKA(L0, Ll)."""
    return float(1.0 - global_cka[layer])


def pareto_mask(rows, maximize=("R", "T"), minimize=("Local",)):
    """Return non-dominated rows for selected diagnostic objectives.

    For the default (R, T maximize; Local minimize), row a dominates row b
    if it is no worse on every objective and strictly better on at least one.
    """
    if not rows:
        return []

    keep = [True] * len(rows)

    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i == j:
                continue

            no_worse = True
            strictly_better = False

            for key in maximize:
                if a[key] < b[key]:
                    no_worse = False
                    break
                if a[key] > b[key]:
                    strictly_better = True

            if not no_worse:
                continue

            for key in minimize:
                if a[key] > b[key]:
                    no_worse = False
                    break
                if a[key] < b[key]:
                    strictly_better = True

            if no_worse and strictly_better:
                keep[j] = False

    return [rows[i]["Layer"] for i, flag in enumerate(keep) if flag]


def minmax_norm(values):
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if abs(vmax - vmin) < 1e-12:
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="cka_analysis_mpi_v4")
    args_cli = parser.parse_args()

    set_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] ============================================================")
    print("[INFO] MPI CKA / Representation-Stage Diagnostic Analysis v4")
    print("[INFO] ============================================================")
    print(f"[INFO] Device: {device}")
    print("[INFO] Analysis split: train")
    print(f"[INFO] MAX_BATCHES: {args_cli.max_batches}")
    print("[INFO] CKA representation: Hybrid Block OUTPUT")
    print("[INFO] Current Stage reference: L0-L2 / L3-L6 / L7-L15")
    print("[INFO] Current representative reference: L2 / L4 / L11")
    print("[INFO] Current retained Teacher layers: L0-L4 + L11-L15")
    print("[INFO] IMPORTANT: no Student/KD experiment will be changed by this script.")

    args = get_config("configs/mpi/MotionAGFormer-DCF.yaml")
    model = load_model(args)

    ckpt = torch.load(
        "checkpoint/MPI_ours_sd42/best_epoch.pth.tr",
        map_location=device,
    )

    new_state_dict = OrderedDict()
    for k, v in ckpt["model"].items():
        name = k[7:] if k.startswith("module.") else k
    
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    print("[INFO] Model loaded successfully")

    n_layers = int(args.n_layers)
    layer_names = [f"layers.{i}" for i in range(n_layers)]

    features = {name: [] for name in layer_names}
    hooks = []

    def make_output_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feat = output.detach().cpu().float()
                feat = feat.reshape(feat.shape[0], -1)
                features[name].append(feat)
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_output_hook(name)))

    analysis_dataset = Fusion(args, train=True)

    # 固定随机抽取 100 个训练样本作为 CKA analysis subset
    rng = np.random.default_rng(42)

    num_samples = min(
        args_cli.max_batches,
        len(analysis_dataset)
    )

    selected_indices = rng.choice(
        len(analysis_dataset),
        size=num_samples,
        replace=False
    )

    analysis_subset = Subset(
        analysis_dataset,
        selected_indices.tolist()
    )

    analysis_loader = DataLoader(
        analysis_subset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    print(f"[INFO] Full train dataset size: {len(analysis_dataset)}")
    print(f"[INFO] CKA analysis subset size: {len(analysis_subset)}")

    with torch.no_grad():
        for i, data in enumerate(analysis_loader):
            if i >= args_cli.max_batches:
                break

            cam, gt_3D, input_2D, seq, scale, bb_box = data

            input_2D, gt_3D, cam, scale, bb_box = get_variable(
                'train',
                [input_2D, gt_3D, cam, scale, bb_box]
            )

            # Match existing MPI evaluation convention: use non-flip input.
            if input_2D.dim() == 5:
                # 测试/增强数据：第 0 个为非 flip 输入
                input_2D = input_2D[:, 0]
            elif input_2D.dim() == 4:
                # 训练数据：已经是 [B, T, J, C]
                pass
            else:
                raise RuntimeError(
                    f"Unexpected input_2D shape: {tuple(input_2D.shape)}"
                )

            _ = model(input_2D)

            if i % 20 == 0:
                print(f"  Processed {i} / {args_cli.max_batches}")

    torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    for name in layer_names:
        if not features[name]:
            raise RuntimeError(f"No features collected for {name}")
        features[name] = torch.cat(features[name], dim=0)
        print(f"[INFO] {name}: {tuple(features[name].shape)}")

    print("[INFO] Computing full CKA matrix...")

    cka_matrix = np.zeros((n_layers, n_layers), dtype=np.float64)

    for i in range(n_layers):
        for j in range(n_layers):
            X = features[layer_names[i]]
            Y = features[layer_names[j]]
            cka_matrix[i, j] = compute_cka(X, Y)
        print(f"  Layer {i} done")

    out_dir = args_cli.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Save raw CKA matrix + heatmap
    # -----------------------------------------------------------------
    np.save(os.path.join(out_dir, "cka_matrix.npy"), cka_matrix)
    np.savetxt(
        os.path.join(out_dir, "cka_matrix.csv"),
        cka_matrix,
        delimiter=",",
        fmt="%.8f",
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        ax=ax,
        vmin=cka_matrix.min(),
        vmax=1.0,
    )
    ax.set_title("DSR-Net Block-level CKA Similarity (Hybrid Block Output)", fontsize=13)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Layer index", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "cka_heatmap_output.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------------------------------------------
    # 2. Adjacent CKA + global CKA(L0, Ll)
    # -----------------------------------------------------------------
    adjacent_cka = np.array(
        [cka_matrix[i, i + 1] for i in range(n_layers - 1)],
        dtype=np.float64,
    )

    global_cka = cka_matrix[0, :].copy()
    global_change = np.array(
        [abs(global_cka[i] - global_cka[i - 1]) for i in range(1, n_layers)],
        dtype=np.float64,
    )

    np.save(os.path.join(out_dir, "global_cka_from_L0.npy"), global_cka)
    np.save(os.path.join(out_dir, "global_change_magnitude.npy"), global_change)
    np.savetxt(
        os.path.join(out_dir, "adjacent_cka.csv"),
        adjacent_cka.reshape(1, -1),
        delimiter=",",
        fmt="%.8f",
    )

        # -----------------------------------------------------------------
    # 2. Global CKA(L0, Li) + change magnitude
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()

    x = np.arange(n_layers)

    # Global CKA similarity to the initial reference layer L0
    ax1.plot(
        x,
        global_cka,
        "o-",
        linewidth=2,
        label=r"CKA($L_0,L_i$)",
    )
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("CKA similarity to L0")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax1.set_ylim(0.0, 1.05)

    # Each change value belongs to the boundary L_{i-1}|L_i.
    # Place its marker at the midpoint between the two layers.
    change_x = np.arange(1, n_layers) - 0.5

    ax2.plot(
        change_x,
        global_change,
        "s--",
        linewidth=1.8,
        label=r"$|g_i-g_{i-1}|$",
    )
    ax2.set_ylabel("Change in CKA similarity to L0")
    ax2.set_ylim(bottom=0.0)

    # Reference stage boundaries: L2|L3 and L6|L7
    for boundary_x in (2.5, 6.5):
        ax1.axvline(
            x=boundary_x,
            linestyle=":",
            linewidth=1.5,
            color="gray",
            alpha=0.8,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title("Global Representation Evolution from L0")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "global_cka_and_change.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------------------------------------------
    # 3. Stable-onset diagnostic
    # -----------------------------------------------------------------
    q1 = float(np.quantile(global_change, 0.25))
    stable_run = 3
    stable_candidates = []

    for start_layer in range(1, n_layers - stable_run):
        # start_layer corresponds to the first layer in the low-change run.
        vals = global_change[start_layer - 1:start_layer - 1 + stable_run]
        if len(vals) < stable_run:
            continue
        if np.all(vals <= q1):
            stable_candidates.append(
                {
                    "start_layer": start_layer,
                    "run": stable_run,
                    "mean_change": float(np.mean(vals)),
                    "max_change": float(np.max(vals)),
                }
            )

    write_csv(
        os.path.join(out_dir, "stable_onset_candidates.csv"),
        stable_candidates,
        ["start_layer", "run", "mean_change", "max_change"],
    )

    # -----------------------------------------------------------------
    # 4. Current manuscript Stage definitions
    # -----------------------------------------------------------------
    stages = {
        "Stage1": list(range(0, 3)),     # L0-L2
        "Stage2": list(range(3, 7)),     # L3-L6
        "Stage3": list(range(7, 16)),    # L7-L15
    }

    retained_teacher_layers = set(
        list(range(0, 5)) + list(range(11, 16))
    )

    current_reps = {
        "Stage1": 2,
        "Stage2": 4,
        "Stage3": 11,
    }

    # -----------------------------------------------------------------
    # 5. Stage-level stability diagnostics
    # -----------------------------------------------------------------
    stage_rows = []

    print("\n[INFO] ============================================================")
    print("[INFO] Reference-stage diagnostics")
    print("[INFO] ============================================================")

    for stage_name, stage_layers in stages.items():
        internal_changes = []

        for l in stage_layers:
            if l > min(stage_layers):
                internal_changes.append(
                    abs(global_cka[l] - global_cka[l - 1])
                )

        mean_change = float(np.mean(internal_changes)) if internal_changes else 0.0
        max_change = float(np.max(internal_changes)) if internal_changes else 0.0

        stage_mean_cka_all = mean_pairwise_cka(cka_matrix, stage_layers)

        retained_in_stage = [
            l for l in stage_layers if l in retained_teacher_layers
        ]
        stage_mean_cka_retained = mean_pairwise_cka(
            cka_matrix,
            retained_in_stage,
        )

        stage_rows.append(
            {
                "Stage": stage_name,
                "Layers": "-".join(f"L{x}" for x in stage_layers),
                "RetainedLayers": "-".join(
                    f"L{x}" for x in retained_in_stage
                ),
                "MeanIntraStageCKA_All": stage_mean_cka_all,
                "MeanIntraStageCKA_Retained": stage_mean_cka_retained,
                "MeanInternalChange": mean_change,
                "MaxInternalChange": max_change,
            }
        )

        rep = current_reps[stage_name]
        rep_rank_layers = sorted(
            retained_in_stage,
            key=lambda l: mean_pairwise_cka(
                cka_matrix,
                [x for x in retained_in_stage if x != l]
            ) if len(retained_in_stage) > 1 else 1.0
        )

        # Direct rank based on each layer's actual within-stage mean CKA.
        rep_scores = {}
        for l in retained_in_stage:
            others = [x for x in retained_in_stage if x != l]
            vals = [cka_matrix[l, x] for x in others]
            rep_scores[l] = float(np.mean(vals)) if vals else 1.0

        sorted_layers = sorted(
            rep_scores,
            key=rep_scores.get,
            reverse=True,
        )
        rep_rank = sorted_layers.index(rep) + 1

        print(
            f"  {stage_name}: current manuscript rep = L{rep}; "
            f"retained={','.join('L'+str(x) for x in retained_in_stage)}; "
            f"mean intra-stage CKA (all)={stage_mean_cka_all:.4f}; "
            f"mean intra-stage CKA (retained)={stage_mean_cka_retained:.4f}; "
            f"rep-rank among retained={rep_rank}/{len(retained_in_stage)}"
        )

    write_csv(
        os.path.join(out_dir, "stage_summary_v4.csv"),
        stage_rows,
        [
            "Stage",
            "Layers",
            "RetainedLayers",
            "MeanIntraStageCKA_All",
            "MeanIntraStageCKA_Retained",
            "MeanInternalChange",
            "MaxInternalChange",
        ],
    )

    # -----------------------------------------------------------------
    # 6. Transition-aware representative diagnostics
    #
    # Candidate restriction:
    #   only Teacher layers actually retained in Student.
    #
    # Metrics:
    #   R = mean CKA to other RETAINED layers in same Stage
    #   T = max adjacent global-CKA change around the layer
    #   N = 1 - CKA(L0, Ll)
    #   Local = mean CKA with immediate Teacher neighbors
    # -----------------------------------------------------------------
    diagnostic_rows = []

    print("\n[INFO] ============================================================")
    print("[INFO] Transition-aware Representative-Block diagnostics")
    print("[INFO] ============================================================")

    for stage_name, stage_layers in stages.items():
        retained = [
            l for l in stage_layers if l in retained_teacher_layers
        ]

        stage_rows_for_print = []

        for l in retained:
            others = [x for x in retained if x != l]
            if others:
                R = float(np.mean([cka_matrix[l, x] for x in others]))
            else:
                R = 1.0

            T = transition_salience(global_cka, l)
            N = novelty_score(global_cka, l)
            Local = local_neighbor_cka(cka_matrix, l, n_layers)

            row = {
                "Stage": stage_name,
                "Layer": l,
                "R": R,
                "T": T,
                "N": N,
                "Local": Local,
                "IsCurrentRep": int(l == current_reps[stage_name]),
                "IsRetained": 1,
            }
            diagnostic_rows.append(row)
            stage_rows_for_print.append(row)

        # Pareto candidate sets.
        pareto_rt = pareto_mask(
            stage_rows_for_print,
            maximize=("R", "T"),
            minimize=(),
        )
        pareto_rtn = pareto_mask(
            stage_rows_for_print,
            maximize=("R", "T", "N"),
            minimize=("Local",),
        )

        # Balanced diagnostic score ONLY for inspection, not for automatic replacement.
        Rn = minmax_norm([r["R"] for r in stage_rows_for_print])
        Tn = minmax_norm([r["T"] for r in stage_rows_for_print])
        Nn = minmax_norm([r["N"] for r in stage_rows_for_print])
        Ln = minmax_norm([r["Local"] for r in stage_rows_for_print])

        for idx, row in enumerate(stage_rows_for_print):
            # Favor representativeness + transition + novelty, penalize local redundancy.
            score = (
                0.35 * Rn[idx]
                + 0.35 * Tn[idx]
                + 0.20 * Nn[idx]
                + 0.10 * (1.0 - Ln[idx])
            )
            row["DiagnosticScore"] = float(score)
            row["Pareto_RT"] = int(row["Layer"] in pareto_rt)
            row["Pareto_RTN"] = int(row["Layer"] in pareto_rtn)

        print(f"  {stage_name}:")
        for row in sorted(
            stage_rows_for_print,
            key=lambda x: x["DiagnosticScore"],
            reverse=True,
        ):
            tags = []
            if row["IsCurrentRep"]:
                tags.append("CURRENT")
            if row["Pareto_RT"]:
                tags.append("PARETO(R+T)")
            if row["Pareto_RTN"]:
                tags.append("PARETO(R+T+N)")
            tag_text = ", ".join(tags) if tags else "-"
            print(
                f"    L{row['Layer']}: "
                f"R={row['R']:.4f}, "
                f"T={row['T']:.4f}, "
                f"N={row['N']:.4f}, "
                f"Local={row['Local']:.4f}, "
                f"score={row['DiagnosticScore']:.4f}, "
                f"{tag_text}"
            )

    write_csv(
        os.path.join(out_dir, "representative_transition_diagnostics_v4.csv"),
        diagnostic_rows,
        [
            "Stage",
            "Layer",
            "R",
            "T",
            "N",
            "Local",
            "DiagnosticScore",
            "IsCurrentRep",
            "IsRetained",
            "Pareto_RT",
            "Pareto_RTN",
        ],
    )

    # -----------------------------------------------------------------
    # 7. Boundary table
    # -----------------------------------------------------------------
    boundary_rows = []
    order = np.argsort(-global_change)

    rank_map = {
        int(layer + 1): int(rank + 1)
        for rank, layer in enumerate(order)
    }

    print("\n" + "=" * 60)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print("Global CKA(L0, Ll):")
    for l, val in enumerate(global_cka):
        print(f"  L{l}: {val:.6f}")

    print("\nBoundary changes |CKA(L0,Lk)-CKA(L0,Lk-1)|:")
    for k, change in enumerate(global_change, start=1):
        print(
            f"  L{k-1}|L{k}: {change:.6f} "
            f"(rank {rank_map[k]})"
        )
        boundary_rows.append(
            {
                "Boundary": f"L{k-1}|L{k}",
                "Layer": k,
                "GlobalChange": float(change),
                "Rank": rank_map[k],
            }
        )

    write_csv(
        os.path.join(out_dir, "stage_boundary_candidates.csv"),
        boundary_rows,
        ["Boundary", "Layer", "GlobalChange", "Rank"],
    )

    print("\nPersistent low-change onset diagnostic:")
    print(f"  Lower-quartile threshold = {q1:.6f}")
    if stable_candidates:
        for c in stable_candidates:
            print(
                f"  Candidate stable onset: L{c['start_layer']} | "
                f"run={c['run']} | "
                f"mean_change={c['mean_change']:.6f} | "
                f"max_change={c['max_change']:.6f}"
            )
    else:
        print("  No persistent low-change onset found under this diagnostic.")

    print("\nStage-level stability:")
    for row in stage_rows:
        print(
            f"  {row['Stage']} {row['Layers']}: "
            f"mean intra-stage CKA(all)={row['MeanIntraStageCKA_All']:.4f}, "
            f"retained={row['MeanIntraStageCKA_Retained']:.4f}, "
            f"mean internal change={row['MeanInternalChange']:.6f}, "
            f"max internal change={row['MaxInternalChange']:.6f}"
        )

    print("\nCurrent manuscript Representative Blocks:")
    for stage_name, rep in current_reps.items():
        rows = [
            r for r in diagnostic_rows
            if r["Stage"] == stage_name and r["Layer"] == rep
        ]
        r = rows[0]
        print(
            f"  {stage_name}: L{rep} | "
            f"R={r['R']:.4f}, "
            f"T={r['T']:.4f}, "
            f"N={r['N']:.4f}, "
            f"Local={r['Local']:.4f}, "
            f"score={r['DiagnosticScore']:.4f}"
        )

    print("\nTop mean-intra-stage-CKA candidates (diagnostic ONLY):")
    for stage_name in stages:
        rows = [r for r in diagnostic_rows if r["Stage"] == stage_name]
        rows = sorted(rows, key=lambda x: x["R"], reverse=True)
        top_text = ", ".join(
            f"L{r['Layer']}({r['R']:.4f})" for r in rows
        )
        print(f"  {stage_name}: {top_text}")

    print("\n[INFO] No model training, pruning, KD, or Warm-up experiment was changed.")
    print(f"[INFO] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
