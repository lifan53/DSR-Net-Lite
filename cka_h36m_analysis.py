#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- For the current CKA analysis, the train split is used with a fixed random
  analysis subset. The subset indices are saved for reproducibility.
- This analysis subset is used to avoid using test-set samples for structural
  analysis. It should not be described as a formal validation set unless the
  project defines an explicit validation split.
- No KD experiment is retrained by this script.
- No pruning structure is changed automatically.
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

from utils.tools import get_config, set_random_seed
from utils.learning import load_model
from data.reader.motion_dataset import MotionDataset3D


# ============================================================
# Defaults
# ============================================================
DEFAULT_CONFIG_PATH = 'configs/h36m/MotionAGFormer_DCF3.yaml'
DEFAULT_CHECKPOINT_PATH = 'checkpoint/ours_DCF3_sd42/best_epoch.pth.tr'
DEFAULT_ANALYSIS_SPLIT = 'train'
DEFAULT_SEED = 42
DEFAULT_MAX_BATCHES = 200
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 4
DEFAULT_OUTPUT_DIR = 'cka_analysis_v4'

LAYER_IDS = list(range(12))
LAYER_NAMES = [f'layers.{i}' for i in LAYER_IDS]

# Final H36M Student retains original teacher L0-L9 after removing L10-L11.
RETAINED_TEACHER_LAYERS = set(range(10))

# Current manuscript partition. This is a REFERENCE, not an automatically imposed truth.
REFERENCE_STAGES = {
    'Stage1': list(range(0, 3)),   # L0-L2
    'Stage2': list(range(3, 7)),   # L3-L6
    'Stage3': list(range(7, 12)),  # L7-L11
}

# Current manuscript representative layers. We explicitly test these rather than silently replacing them.
CURRENT_MANUSCRIPT_REPRESENTATIVES = {
    'Stage1': 2,
    'Stage2': 4,
    'Stage3': 7,
}


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='H36M CKA / Stage diagnostic analysis v4')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument('--split', default=DEFAULT_ANALYSIS_SPLIT)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-batches', type=int, default=DEFAULT_MAX_BATCHES)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--bootstrap', action='store_true',
                        help='Run bootstrap stability diagnostics on extracted features.')
    parser.add_argument('--bootstrap-runs', type=int, default=50)
    parser.add_argument('--bootstrap-samples', type=int, default=0,
                        help='Samples per bootstrap run; 0 means min(100, N).')
    return parser.parse_args()


# ============================================================
# CKA
# ============================================================
def compute_hsic(K, L):
    """Linear HSIC used by linear CKA."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return torch.trace(KH @ LH) / ((n - 1) ** 2)


def compute_cka(X, Y):
    """Compute linear CKA between two [N, D] representations."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = compute_hsic(K, L)
    hsic_xx = compute_hsic(K, K)
    hsic_yy = compute_hsic(L, L)

    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-20))
    value = hsic_xy / denom
    return float(value.item())


# ============================================================
# Model / feature extraction
# ============================================================
def unwrap_tensor(output):
    """Extract a tensor from a tensor / tuple / list module output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f'Unsupported module output type: {type(output)}')


def register_output_hooks(model, features):
    """Collect the Hybrid Block OUTPUT, not output-input residual change."""
    hooks = []

    def make_hook(name):
        def hook(module, _input, output):
            feat = unwrap_tensor(output).detach().cpu().float()
            feat = feat.reshape(feat.shape[0], -1)
            features[name].append(feat)
        return hook

    for name, module in model.named_modules():
        if name in LAYER_NAMES:
            hooks.append(module.register_forward_hook(make_hook(name)))

    if len(hooks) != len(LAYER_NAMES):
        found = len(hooks)
        raise RuntimeError(
            f'Expected {len(LAYER_NAMES)} layer hooks but registered {found}. '
            f'Check model layer names.'
        )

    return hooks


def load_teacher(args, device):
    cfg = get_config(args.config)
    model = load_model(cfg)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' not in ckpt:
        raise KeyError('Checkpoint does not contain key: model')

    new_state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f'[WARN] Missing keys: {len(missing)}')
    if unexpected:
        print(f'[WARN] Unexpected keys: {len(unexpected)}')

    model.to(device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return cfg, model


def extract_features(args, model, cfg, device, output_dir):
    features = {name: [] for name in LAYER_NAMES}
    hooks = register_output_hooks(model, features)

    dataset = MotionDataset3D(cfg, cfg.subset_list, args.split)

    # ------------------------------------------------------------
    # Fixed random analysis subset
    # For train split, use a fixed random subset rather than the
    # first N samples, so the CKA analysis is not biased by dataset order.
    # ------------------------------------------------------------
    if args.split == 'train':
        rng = np.random.default_rng(args.seed)

        # Current setting: batch_size=1, so max-batches = number of samples.
        num_samples = min(
            args.max_batches * args.batch_size,
            len(dataset)
        )

        selected_indices = rng.choice(
            len(dataset),
            size=num_samples,
            replace=False
        )

        # Save exact indices for reproducibility.
        np.save(
            os.path.join(
                output_dir,
                'cka_analysis_indices.npy'
            ),
            selected_indices
        )

        analysis_dataset = Subset(
            dataset,
            selected_indices.tolist()
        )

        print(f'[INFO] Full train dataset size: {len(dataset)}')
        print(f'[INFO] Random CKA analysis subset size: {len(analysis_dataset)}')
        print(f'[INFO] Random subset seed: {args.seed}')
    else:
        # For test/other splits, keep the original sequential behavior.
        analysis_dataset = dataset
        print(f'[INFO] Dataset size: {len(dataset)}')

    loader = DataLoader(
        analysis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print(f'[INFO] Extracting features from {args.max_batches} batches...')

    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.max_batches:
                    break

                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device, non_blocking=True)
                _ = model(x)

                if i % 20 == 0:
                    print(f'  Processed {i} / {args.max_batches}')
    finally:
        for hook in hooks:
            hook.remove()

    for name in LAYER_NAMES:
        if not features[name]:
            raise RuntimeError(f'No features collected for {name}')
        features[name] = torch.cat(features[name], dim=0)
        print(f'[INFO] {name}: {tuple(features[name].shape)}')

    np.savez(
        os.path.join(output_dir, 'raw_feature_shapes.npz'),
        **{name: np.array(features[name].shape, dtype=np.int64)
           for name in LAYER_NAMES}
    )

    return features


# ============================================================
# CKA matrix / representation evolution
# ============================================================
def compute_cka_matrix(features):
    n_layers = len(LAYER_NAMES)
    cka_matrix = np.zeros((n_layers, n_layers), dtype=np.float64)

    print('[INFO] Computing full CKA matrix...')
    for i in range(n_layers):
        for j in range(i, n_layers):
            value = compute_cka(
                features[LAYER_NAMES[i]],
                features[LAYER_NAMES[j]],
            )
            cka_matrix[i, j] = value
            cka_matrix[j, i] = value
        print(f'  Layer {i} done')

    return cka_matrix


def save_cka_matrix(cka_matrix, output_dir):
    np.save(os.path.join(output_dir, 'cka_matrix.npy'), cka_matrix)
    pd.DataFrame(
        cka_matrix,
        index=[f'L{i}' for i in LAYER_IDS],
        columns=[f'L{i}' for i in LAYER_IDS],
    ).to_csv(
        os.path.join(output_dir, 'cka_matrix.csv'),
        encoding='utf-8-sig'
    )


def compute_global_evolution(cka_matrix):
    global_cka = np.array([cka_matrix[0, k] for k in LAYER_IDS], dtype=np.float64)
    change_magnitude = np.abs(np.diff(global_cka))

    boundary_df = pd.DataFrame({
        'boundary': [f'L{k-1}|L{k}' for k in range(1, len(LAYER_IDS))],
        'left_layer': list(range(len(LAYER_IDS) - 1)),
        'right_layer': list(range(1, len(LAYER_IDS))),
        'global_cka_left': global_cka[:-1],
        'global_cka_right': global_cka[1:],
        'change_magnitude': change_magnitude,
    })
    boundary_df['rank'] = boundary_df['change_magnitude'].rank(
        ascending=False, method='min'
    ).astype(int)

    return global_cka, change_magnitude, boundary_df


def compute_adjacent_cka(cka_matrix):
    adjacent_cka = np.array(
        [cka_matrix[k - 1, k] for k in range(1, len(LAYER_IDS))],
        dtype=np.float64,
    )
    adjacent_change = np.abs(np.diff(adjacent_cka))
    return adjacent_cka, adjacent_change


# ============================================================
# Reference Stage diagnostics
# ============================================================
def stage_pair_values(cka_matrix, stage_ids):
    values = []
    for i, li in enumerate(stage_ids):
        for lj in stage_ids[i + 1:]:
            values.append(float(cka_matrix[li, lj]))
    return np.array(values, dtype=np.float64)


def stage_global_change_values(change_magnitude, stage_ids):
    """
    For a Stage [a, b], internal evolution changes correspond to boundaries:
        L_a|L_{a+1}, ..., L_{b-1}|L_b
    """
    if len(stage_ids) <= 1:
        return np.array([], dtype=np.float64)
    return np.array(
        [change_magnitude[k - 1] for k in range(stage_ids[0] + 1, stage_ids[-1] + 1)],
        dtype=np.float64,
    )


def local_neighbor_redundancy(cka_matrix, layer, stage_ids):
    ids = set(stage_ids)
    neighbor_values = []
    for nb in [layer - 1, layer + 1]:
        if nb in ids:
            neighbor_values.append(float(cka_matrix[layer, nb]))
    return float(np.mean(neighbor_values)) if neighbor_values else np.nan


def mean_intra_stage_score(cka_matrix, layer, stage_ids):
    others = [j for j in stage_ids if j != layer]
    if not others:
        return 1.0
    return float(np.mean([cka_matrix[layer, j] for j in others]))


def stage_position(layer, stage_ids):
    a, b = stage_ids[0], stage_ids[-1]
    if a == b:
        return 0.0
    return float((layer - a) / (b - a))


def build_representative_diagnostics(cka_matrix, change_magnitude):
    rows = []
    summary_rows = []

    for stage_name, stage_ids in REFERENCE_STAGES.items():
        retained = [l for l in stage_ids if l in RETAINED_TEACHER_LAYERS]
        pair_values = stage_pair_values(cka_matrix, stage_ids)
        internal_change = stage_global_change_values(change_magnitude, stage_ids)

        summary_rows.append({
            'stage': stage_name,
            'layer_range': f'L{stage_ids[0]}-L{stage_ids[-1]}',
            'num_layers': len(stage_ids),
            'mean_intra_stage_cka': float(pair_values.mean()) if len(pair_values) else 1.0,
            'median_intra_stage_cka': float(np.median(pair_values)) if len(pair_values) else 1.0,
            'max_intra_stage_cka': float(pair_values.max()) if len(pair_values) else 1.0,
            'mean_global_change_inside_stage': float(internal_change.mean()) if len(internal_change) else 0.0,
            'median_global_change_inside_stage': float(np.median(internal_change)) if len(internal_change) else 0.0,
            'max_global_change_inside_stage': float(internal_change.max()) if len(internal_change) else 0.0,
        })

        # Current manuscript representative is explicitly flagged.
        current_rep = CURRENT_MANUSCRIPT_REPRESENTATIVES[stage_name]

        # Stage boundary change magnitudes, useful for transition-aware diagnostics.
        start_boundary_change = (
            float(change_magnitude[stage_ids[0] - 1]) if stage_ids[0] > 0 else np.nan
        )
        end_boundary_change = (
            float(change_magnitude[stage_ids[-1]]) if stage_ids[-1] < len(LAYER_IDS) - 1 else np.nan
        )

        for layer in retained:
            intra = mean_intra_stage_score(cka_matrix, layer, stage_ids)
            local_red = local_neighbor_redundancy(cka_matrix, layer, stage_ids)
            pos = stage_position(layer, stage_ids)

            # Distances to stage start/end, purely descriptive and normalized to [0, 1].
            dist_start = pos
            dist_end = 1.0 - pos

            # Local representation-change magnitudes around the layer.
            before_change = (
                float(change_magnitude[layer - 1]) if layer > 0 else np.nan
            )
            after_change = (
                float(change_magnitude[layer]) if layer < len(LAYER_IDS) - 1 else np.nan
            )

            rows.append({
                'stage': stage_name,
                'layer': layer,
                'layer_label': f'L{layer}',
                'retained_by_student': True,
                'mean_intra_stage_cka': intra,
                'local_neighbor_cka': local_red,
                'stage_position_0_to_1': pos,
                'distance_from_stage_start': dist_start,
                'distance_from_stage_end': dist_end,
                'change_before_layer': before_change,
                'change_after_layer': after_change,
                'current_manuscript_rep': layer == current_rep,
            })

        # Print stage summary and current representative position.
        candidate_df = pd.DataFrame([r for r in rows if r['stage'] == stage_name])
        if not candidate_df.empty:
            candidate_df['rank_mean_cka'] = candidate_df['mean_intra_stage_cka'].rank(
                ascending=False, method='min'
            ).astype(int)
            candidate_df['rank_local_neighbor_cka'] = candidate_df['local_neighbor_cka'].rank(
                ascending=True, method='min'
            ).astype(int)
            current_row = candidate_df[candidate_df['layer'] == current_rep]
            if not current_row.empty:
                r = current_row.iloc[0]
                print(
                    f'  {stage_name}: current manuscript rep = L{current_rep}; '
                    f'mean intra-stage CKA={r["mean_intra_stage_cka"]:.4f}; '
                    f'rank among retained layers={int(r["rank_mean_cka"])} / {len(candidate_df)}; '
                    f'local-neighbor CKA={r["local_neighbor_cka"]:.4f}'
                )
            else:
                print(f'  [WARN] Current representative L{current_rep} not retained in {stage_name}.')

    return pd.DataFrame(summary_rows), pd.DataFrame(rows)


# ============================================================
# Transition-aware Representative-Block diagnostics
# ============================================================
def minmax_normalize(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def transition_salience(change_magnitude, layer):
    """Max representation-change magnitude immediately before/after a layer."""
    vals = []
    if layer > 0:
        vals.append(float(change_magnitude[layer - 1]))
    if layer < len(LAYER_IDS) - 1:
        vals.append(float(change_magnitude[layer]))
    return float(max(vals)) if vals else 0.0


def stage_novelty_score(cka_matrix, layer, stage_ids):
    """
    Novelty relative to earlier layers in the same stage.
    Higher means the layer departs more from earlier within-stage representations.
    For the first stage layer, use the mean distance to all other stage layers.
    """
    earlier = [j for j in stage_ids if j < layer]
    if earlier:
        return float(np.mean([1.0 - cka_matrix[layer, j] for j in earlier]))
    others = [j for j in stage_ids if j != layer]
    if not others:
        return 0.0
    return float(np.mean([1.0 - cka_matrix[layer, j] for j in others]))


def pareto_front(df, maximize_cols):
    """Return boolean Pareto-front mask for selected columns to maximize."""
    vals = df[maximize_cols].to_numpy(dtype=np.float64)
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                keep[i] = False
                break
    return keep


def build_transition_aware_diagnostics(cka_matrix, change_magnitude):
    """
    Build a diagnostic table for the current retained layers.

    IMPORTANT:
    This function does not hard-code or automatically replace the manuscript's
    representative blocks. It reports multiple measurable properties so that
    the final selection can be justified from data rather than from a single
    mean-CKA criterion.
    """
    rows = []
    for stage_name, stage_ids in REFERENCE_STAGES.items():
        retained = [l for l in stage_ids if l in RETAINED_TEACHER_LAYERS]
        current_rep = CURRENT_MANUSCRIPT_REPRESENTATIVES[stage_name]

        for layer in retained:
            intra = mean_intra_stage_score(cka_matrix, layer, stage_ids)
            local_red = local_neighbor_redundancy(cka_matrix, layer, stage_ids)
            trans = transition_salience(change_magnitude, layer)
            novelty = stage_novelty_score(cka_matrix, layer, stage_ids)
            pos = stage_position(layer, stage_ids)

            rows.append({
                'stage': stage_name,
                'layer': int(layer),
                'layer_label': f'L{layer}',
                'retained_by_student': True,
                'mean_intra_stage_cka': intra,
                'local_neighbor_cka': local_red,
                'transition_salience': trans,
                'stage_novelty': novelty,
                'stage_position_0_to_1': pos,
                'is_stage_start': layer == stage_ids[0],
                'is_stage_end': layer == stage_ids[-1],
                'current_manuscript_rep': layer == current_rep,
            })

    df = pd.DataFrame(rows)
    df['rank_mean_intra_stage_cka'] = (
        df.groupby('stage')['mean_intra_stage_cka']
        .rank(ascending=False, method='min').astype(int)
    )
    df['rank_transition_salience'] = (
        df.groupby('stage')['transition_salience']
        .rank(ascending=False, method='min').astype(int)
    )
    df['rank_stage_novelty'] = (
        df.groupby('stage')['stage_novelty']
        .rank(ascending=False, method='min').astype(int)
    )
    df['rank_local_neighbor_cka_low_redundancy'] = (
        df.groupby('stage')['local_neighbor_cka']
        .rank(ascending=True, method='min').astype(int)
    )

    # Pareto front: favor representativeness + transition relevance,
    # while retaining the redundancy metric as a separate diagnostic.
    df['rep_norm_within_stage'] = (
        df.groupby('stage')['mean_intra_stage_cka']
        .transform(lambda x: minmax_normalize(x.to_numpy()))
    )
    df['transition_norm_within_stage'] = (
        df.groupby('stage')['transition_salience']
        .transform(lambda x: minmax_normalize(x.to_numpy()))
    )
    df['novelty_norm_within_stage'] = (
        df.groupby('stage')['stage_novelty']
        .transform(lambda x: minmax_normalize(x.to_numpy()))
    )
    df['low_redundancy_norm_within_stage'] = (
        df.groupby('stage')['local_neighbor_cka']
        .transform(lambda x: 1.0 - minmax_normalize(x.to_numpy()))
    )

    df['pareto_rep_transition'] = False
    df['pareto_rep_transition_novelty'] = False
    for stage_name in REFERENCE_STAGES:
        mask = df['stage'] == stage_name
        sub = df.loc[mask].copy()
        df.loc[mask, 'pareto_rep_transition'] = pareto_front(
            sub, ['rep_norm_within_stage', 'transition_norm_within_stage']
        )
        df.loc[mask, 'pareto_rep_transition_novelty'] = pareto_front(
            sub, ['rep_norm_within_stage', 'transition_norm_within_stage', 'novelty_norm_within_stage']
        )

    # A descriptive, not normative, balanced diagnostic score.
    # It is explicitly labeled as diagnostic; final manuscript selection is not
    # automatically changed by it.
    df['diagnostic_balanced_score'] = (
        0.45 * df['rep_norm_within_stage']
        + 0.30 * df['transition_norm_within_stage']
        + 0.15 * df['novelty_norm_within_stage']
        + 0.10 * df['low_redundancy_norm_within_stage']
    )
    df['rank_diagnostic_balanced_score'] = (
        df.groupby('stage')['diagnostic_balanced_score']
        .rank(ascending=False, method='min').astype(int)
    )

    return df


# ============================================================
# Persistent low-change / stable onset diagnostic
# ============================================================
def detect_stable_onsets(change_magnitude, quantile=0.25, run_length=3):
    """
    Detect the beginning of a persistent low-change regime.

    Threshold is data-driven: the lower quantile of all boundary change magnitudes.
    A candidate onset at layer Lk means the next `run_length` boundaries
    Lk|L(k+1), ... remain below/equal to that threshold.

    This is a DIAGNOSTIC, not an automatic replacement of the manuscript stages.
    """
    if len(change_magnitude) < run_length:
        return [], np.nan

    threshold = float(np.quantile(change_magnitude, quantile))
    candidates = []

    # change_magnitude index k-1 corresponds to boundary L(k-1)|Lk.
    # For onset at Lk, inspect boundaries Lk|L(k+1) onward.
    for onset_layer in range(1, len(LAYER_IDS) - run_length + 1):
        first_idx = onset_layer  # boundary L(onset)|L(onset+1) -> array idx onset
        last_idx_exclusive = first_idx + run_length
        if last_idx_exclusive > len(change_magnitude):
            break
        run = change_magnitude[first_idx:last_idx_exclusive]
        if np.all(run <= threshold):
            candidates.append({
                'onset_layer': onset_layer,
                'onset_label': f'L{onset_layer}',
                'threshold': threshold,
                'run_length': run_length,
                'run_mean_change': float(run.mean()),
                'run_max_change': float(run.max()),
            })

    return candidates, threshold


# ============================================================
# Bootstrap stability diagnostics
# ============================================================
def bootstrap_reference_stage_scores(features, runs, sample_count, seed):
    rng = np.random.default_rng(seed)

    n = features[LAYER_NAMES[0]].shape[0]
    sample_count = min(sample_count, n)

    rows = []

    print(f'[INFO] Bootstrap: runs={runs}, samples/run={sample_count}, N={n}')

    for run_id in range(runs):
        idx = rng.choice(n, size=sample_count, replace=True)
        subset_features = {
            name: features[name][idx]
            for name in LAYER_NAMES
        }

        cka = compute_cka_matrix(subset_features)
        global_cka, changes, _ = compute_global_evolution(cka)

        for stage_name, stage_ids in REFERENCE_STAGES.items():
            retained = [l for l in stage_ids if l in RETAINED_TEACHER_LAYERS]
            for layer in retained:
                score = mean_intra_stage_score(cka, layer, stage_ids)
                rows.append({
                    'run': run_id,
                    'stage': stage_name,
                    'layer': layer,
                    'mean_intra_stage_cka': score,
                })

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR_GLOBAL, 'bootstrap_representative_scores_raw.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    summary = []
    for stage_name in REFERENCE_STAGES:
        stage_df = df[df['stage'] == stage_name]
        # For each run, identify the highest mean-CKA retained layer.
        winners = (
            stage_df.sort_values(['run', 'mean_intra_stage_cka'], ascending=[True, False])
            .groupby('run', as_index=False)
            .first()
        )
        win_counts = winners['layer'].value_counts().to_dict()

        for layer in sorted(stage_df['layer'].unique()):
            layer_df = stage_df[stage_df['layer'] == layer]
            summary.append({
                'stage': stage_name,
                'layer': int(layer),
                'score_mean': float(layer_df['mean_intra_stage_cka'].mean()),
                'score_std': float(layer_df['mean_intra_stage_cka'].std(ddof=1)),
                'selected_as_top_mean_cka_count': int(win_counts.get(layer, 0)),
                'selection_frequency': float(win_counts.get(layer, 0) / runs),
                'current_manuscript_rep': layer == CURRENT_MANUSCRIPT_REPRESENTATIVES[stage_name],
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR_GLOBAL, 'bootstrap_representative_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    return summary_df


# Global used only by bootstrap helper after args are parsed.
OUTPUT_DIR_GLOBAL = DEFAULT_OUTPUT_DIR


# ============================================================
# Plots
# ============================================================
def save_heatmap(cka_matrix, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        xticklabels=[f'L{i}' for i in LAYER_IDS],
        yticklabels=[f'L{i}' for i in LAYER_IDS],
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        square=True,
    )
    ax.set_title('DSR-Net Block-level CKA Similarity (Block Output)')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Layer index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cka_heatmap_output.png'), dpi=200)
    plt.close()


def save_global_evolution_plot(global_cka, change_magnitude, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(LAYER_IDS))

    ax1.plot(x, global_cka, 'o-', linewidth=2, label='CKA(L0, Ll)')
    ax1.set_xlabel('Layer index')
    ax1.set_ylabel('Global CKA to L0')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in LAYER_IDS])
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    # Each change value corresponds to the boundary between L_{i-1} and L_i.
    change_x = np.arange(1, len(LAYER_IDS)) - 0.5

    ax2.plot(
        change_x,
        change_magnitude,
        's--',
        linewidth=1.8,
        label=r'$|g_i-g_{i-1}|$'
    )
    ax2.set_ylabel('Change in CKA similarity to L0')
    ax2.set_ylim(bottom=0.0)

    # Current manuscript stage boundaries: L2|L3 and L6|L7.
    for boundary_layer in [3, 7]:
        ax1.axvline(
            boundary_layer - 0.5,
            linestyle=':',
            linewidth=1.5,
            alpha=0.8,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.set_title('Global Representation Evolution from L0')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_cka_and_change.png'), dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    global OUTPUT_DIR_GLOBAL
    args = parse_args()
    OUTPUT_DIR_GLOBAL = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[INFO] ============================================================')
    print('[INFO] H36M CKA / Representation-Stage Diagnostic Analysis v4')
    print('[INFO] ============================================================')
    print(f'[INFO] Device: {device}')
    print(f'[INFO] Analysis split: {args.split}')
    print(f'[INFO] MAX_BATCHES: {args.max_batches}')
    print('[INFO] CKA representation: Hybrid Block OUTPUT')
    print('[INFO] Current Stage reference: L0-L2 / L3-L6 / L7-L11')
    print('[INFO] Current representative reference: L2 / L4 / L7')
    print('[INFO] IMPORTANT: no Student/KD experiment will be changed by this script.')

    # 1. Model
    cfg, model = load_teacher(args, device)
    print('[INFO] Model loaded successfully')

    # 2. Features
    features = extract_features(args, model, cfg, device, args.output_dir)

    # 3. Full CKA
    cka_matrix = compute_cka_matrix(features)
    save_cka_matrix(cka_matrix, args.output_dir)
    save_heatmap(cka_matrix, args.output_dir)

    # 4. Global evolution
    global_cka, change_magnitude, boundary_df = compute_global_evolution(cka_matrix)
    adjacent_cka, adjacent_change = compute_adjacent_cka(cka_matrix)

    boundary_df.to_csv(
        os.path.join(args.output_dir, 'stage_boundary_candidates.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    np.save(os.path.join(args.output_dir, 'global_cka_from_L0.npy'), global_cka)
    np.save(os.path.join(args.output_dir, 'global_change_magnitude.npy'), change_magnitude)
    np.save(os.path.join(args.output_dir, 'adjacent_cka.npy'), adjacent_cka)
    np.save(os.path.join(args.output_dir, 'adjacent_change_magnitude.npy'), adjacent_change)

    save_global_evolution_plot(global_cka, change_magnitude, args.output_dir)

    adjacent_df = pd.DataFrame({
        'pair': [f'L{k-1}-L{k}' for k in range(1, len(LAYER_IDS))],
        'adjacent_cka': adjacent_cka,
        'adjacent_cka_change': np.r_[np.nan, adjacent_change],
    })
    adjacent_df.to_csv(
        os.path.join(args.output_dir, 'adjacent_cka.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 5. Stable-onset diagnostic
    stable_candidates, stable_threshold = detect_stable_onsets(
        change_magnitude,
        quantile=0.25,
        run_length=3,
    )

    stable_df = pd.DataFrame(stable_candidates)
    stable_df.to_csv(
        os.path.join(args.output_dir, 'stable_onset_candidates.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 6. Stage diagnostics
    print('\n[INFO] ============================================================')
    print('[INFO] Reference-stage diagnostics')
    print('[INFO] ============================================================')

    stage_summary_df, rep_diag_df = build_representative_diagnostics(
        cka_matrix,
        change_magnitude,
    )

    stage_summary_df.to_csv(
        os.path.join(args.output_dir, 'reference_stage_summary_v4.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    rep_diag_df['rank_mean_cka'] = (
        rep_diag_df.groupby('stage')['mean_intra_stage_cka']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    rep_diag_df['rank_local_neighbor_cka'] = (
        rep_diag_df.groupby('stage')['local_neighbor_cka']
        .rank(ascending=True, method='min')
        .astype(int)
    )
    rep_diag_df.to_csv(
        os.path.join(args.output_dir, 'representative_block_diagnostics_v4.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 7. Transition-aware Representative-Block diagnostics
    transition_df = build_transition_aware_diagnostics(cka_matrix, change_magnitude)
    transition_df.to_csv(
        os.path.join(args.output_dir, 'representative_transition_diagnostics_v4.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print('\n[INFO] ============================================================')
    print('[INFO] Transition-aware Representative-Block diagnostics')
    print('[INFO] ============================================================')
    for stage_name in REFERENCE_STAGES:
        sub = transition_df[transition_df['stage'] == stage_name].copy()
        print(f'  {stage_name}:')
        for _, r in sub.sort_values('diagnostic_balanced_score', ascending=False).iterrows():
            flags = []
            if bool(r['current_manuscript_rep']):
                flags.append('CURRENT')
            if bool(r['pareto_rep_transition']):
                flags.append('PARETO(R+T)')
            if bool(r['pareto_rep_transition_novelty']):
                flags.append('PARETO(R+T+N)')
            flag_str = ', '.join(flags) if flags else '-'
            print(
                f"    L{int(r['layer'])}: "
                f"R={r['mean_intra_stage_cka']:.4f}, "
                f"T={r['transition_salience']:.4f}, "
                f"N={r['stage_novelty']:.4f}, "
                f"Local={r['local_neighbor_cka']:.4f}, "
                f"score={r['diagnostic_balanced_score']:.4f}, "
                f"{flag_str}"
            )

    # 8. Bootstrap if requested
    if args.bootstrap:
        sample_count = args.bootstrap_samples or min(100, features[LAYER_NAMES[0]].shape[0])
        bootstrap_reference_stage_scores(
            features,
            runs=args.bootstrap_runs,
            sample_count=sample_count,
            seed=args.seed + 1000,
        )

    # 9. Final console summary
    print('\n============================================================')
    print('FINAL DIAGNOSTIC SUMMARY')
    print('============================================================')

    print('Global CKA(L0, Ll):')
    for layer, value in enumerate(global_cka):
        print(f'  L{layer}: {value:.6f}')

    print('\nBoundary changes |CKA(L0,Lk)-CKA(L0,Lk-1)|:')
    for _, row in boundary_df.sort_values('right_layer').iterrows():
        print(
            f"  {row['boundary']}: {row['change_magnitude']:.6f} "
            f"(rank {int(row['rank'])})"
        )

    print('\nPersistent low-change onset diagnostic:')
    print(f'  Lower-quartile threshold = {stable_threshold:.6f}')
    if stable_candidates:
        for item in stable_candidates:
            print(
                f"  Candidate stable onset: {item['onset_label']} | "
                f"run={item['run_length']} | "
                f"mean_change={item['run_mean_change']:.6f} | "
                f"max_change={item['run_max_change']:.6f}"
            )
    else:
        print('  No persistent low-change onset found under current diagnostic rule.')

    print('\nStage-level stability:')
    for _, row in stage_summary_df.iterrows():
        print(
            f"  {row['stage']} {row['layer_range']}: "
            f"mean intra-stage CKA={row['mean_intra_stage_cka']:.4f}, "
            f"mean internal change={row['mean_global_change_inside_stage']:.6f}, "
            f"max internal change={row['max_global_change_inside_stage']:.6f}"
        )

    print('\nCurrent manuscript Representative Blocks:')
    for stage_name, layer in CURRENT_MANUSCRIPT_REPRESENTATIVES.items():
        row = rep_diag_df[
            (rep_diag_df['stage'] == stage_name) &
            (rep_diag_df['layer'] == layer)
        ]
        if row.empty:
            print(f'  {stage_name}: L{layer} not found')
            continue
        row = row.iloc[0]
        print(
            f"  {stage_name}: L{layer} | "
            f"mean CKA={row['mean_intra_stage_cka']:.4f}, "
            f"local-neighbor CKA={row['local_neighbor_cka']:.4f}, "
            f"mean-CKA rank={int(row['rank_mean_cka'])}/{sum(1 for x in REFERENCE_STAGES[stage_name] if x in RETAINED_TEACHER_LAYERS)}-within-stage"
        )

    print('\nTop mean-intra-stage-CKA candidates (diagnostic ONLY):')
    for stage_name in REFERENCE_STAGES:
        stage_df = rep_diag_df[rep_diag_df['stage'] == stage_name]
        stage_df = stage_df.sort_values('mean_intra_stage_cka', ascending=False)
        text = ', '.join(
            [f"L{int(r.layer)}({r.mean_intra_stage_cka:.4f})" for _, r in stage_df.iterrows()]
        )
        print(f'  {stage_name}: {text}')

    print('\n[INFO] No model training, pruning, KD, or Warm-up experiment was changed.')
    print(f'[INFO] Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
