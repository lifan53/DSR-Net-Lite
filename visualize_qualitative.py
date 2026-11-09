#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qualitative 3D pose visualization for Human3.6M and MPI-INF-3DHP.

Four columns:
    Ground Truth | Teacher | Student | KD Student

Design:
1. Models are built with the project's own get_config() -> load_model().
2. Checkpoints use checkpoint['model'] and DataParallel 'module.' prefixes
   are handled automatically.
3. H36M and MPI evaluation preprocessing are kept separate.
4. The same test sample/frame is used for all three models.
5. Candidate mode uses uniformly spaced frames only; it does not select
   frames according to model quality.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.tools import get_config
from utils.learning import load_model
from utils.data import flip_data, denormalize
from data.reader.motion_dataset import MotionDataset3D, Fusion
from data.reader.h36m import DataReaderH36M


CONFIGS = {
    "h36m": {
        "teacher": {
            "config": "configs/h36m/MotionAGFormer_DCF3.yaml",
            "checkpoint": "checkpoint/ours_DCF3_sd42/best_epoch.pth.tr",
        },
        "student": {
            "config": "configs/h36m/MotionAGFormer_DCF3_CKA_student_w96.yaml",
            "checkpoint": "checkpoint/student_tail_w96_train/best_epoch.pth.tr",
        },
        "kd": {
            "config": "configs/h36m/MotionAGFormer_DCF3_CKA_student_w96.yaml",
            "checkpoint": "checkpoint/student_tail_w96_kd_stage247_warmup/best_epoch.pth.tr",
        },
    },
    "mpi": {
        "teacher": {
            "config": "configs/mpi/MotionAGFormer-DCF.yaml",
            "checkpoint": "checkpoint/MPI_ours_sd42/best_epoch.pth.tr",
        },
        "student": {
            "config": "configs/mpi/MotionAGFormer-DCF-T10-W112.yaml",
            "checkpoint": "checkpoint/MPI_student_T10W112_Hybrid_sd42/best_epoch.pth.tr",
        },
        "kd": {
            "config": "configs/mpi/MotionAGFormer-DCF-T10-W112.yaml",
            "checkpoint": "checkpoint/MPI_student_T10W112_LinearWarmup10_sd42/best_epoch.pth.tr",
        },
    },
}

# H36M 17-joint topology used in the current project/evaluation setup.
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# MPI joint order is explicitly defined in the supplied evaluation code:
# right leg = 2,3,4; left leg = 5,6,7; right arm = 8,9,10;
# left arm = 11,12,13; root = 14.
# The torso/head links below are therefore kept as a local visualization
# topology; replace this list with a project-specific 3DHP skeleton utility
# if your repository already provides one.
MPI_SKELETON = [
    (14, 2), (2, 3), (3, 4),
    (14, 5), (5, 6), (6, 7),
    (14, 0), (0, 1),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]


def parse_int_list(text):
    if text is None or not text.strip():
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint_from_config(config_path, checkpoint_path, device):
    """Build the model exactly as training did, then load checkpoint['model']."""
    args = get_config(config_path)
    model = load_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' key: {checkpoint_path}")

    state_dict = strip_module_prefix(checkpoint["model"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch.\n"
            f"config={config_path}\n"
            f"checkpoint={checkpoint_path}\n"
            f"missing ({len(missing)}): {missing[:10]}\n"
            f"unexpected ({len(unexpected)}): {unexpected[:10]}"
        )

    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {config_path}")
    print(f"        ckpt   = {checkpoint_path}")
    print(f"        params = {n_params:,}")
    return model, args


def ensure_paths(dataset_name):
    for name in ("teacher", "student", "kd"):
        cfg = CONFIGS[dataset_name][name]["config"]
        ckpt = CONFIGS[dataset_name][name]["checkpoint"]
        if not os.path.isfile(cfg):
            raise FileNotFoundError(cfg)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)


def make_loader(dataset, batch_size, num_workers=0):
    kwargs = {
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


# -------------------------------------------------------------------------
# Human3.6M
# -------------------------------------------------------------------------

def collect_h36m_inputs(args, sample_indices):
    dataset = MotionDataset3D(args, args.subset_list, "test")
    loader = make_loader(dataset, getattr(args, "batch_size", 4), num_workers=0)
    wanted = set(sample_indices)
    samples = {}
    global_idx = 0

    for x, y in loader:
        for i in range(x.shape[0]):
            idx = global_idx + i
            if idx in wanted:
                samples[idx] = {
                    "x": x[i].clone(),
                    "y": y[i].clone(),
                }
        global_idx += x.shape[0]
        if len(samples) == len(wanted):
            break

    missing = wanted - set(samples)
    if missing:
        raise IndexError(f"H36M sample indices not found: {sorted(missing)}")
    return samples


def h36m_predict_one(model, x, args, device):
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        if args.flip:
            pred1 = model(x)
            pred_flip = model(flip_data(x))
            pred2 = flip_data(pred_flip)
            pred = (pred1 + pred2) / 2.0
        else:
            pred = model(x)

    if args.root_rel:
        pred[:, :, 0, :] = 0
    return pred[0].detach().cpu().numpy()

def h36m_predict_batch(model, x, args, device):
    """
    x: [B, T, J, C]
    return: [B, T, J, 3]
    """
    x = x.to(device)

    with torch.no_grad():
        if args.flip:
            pred1 = model(x)

            pred_flip = model(flip_data(x))
            pred2 = flip_data(pred_flip)

            pred = (pred1 + pred2) / 2.0
        else:
            pred = model(x)

    if args.root_rel:
        pred[:, :, 0, :] = 0

    return pred.detach().cpu().numpy()

def h36m_gt_for_sample(datareader, sample_idx):
    _, split_id_test = datareader.get_split_id()

    gts = np.array(
        datareader.dt_dataset["test"]["joints_2.5d_image"]
    )
    actions = np.array(
        datareader.dt_dataset["test"]["action"]
    )
    sources = np.array(
        datareader.dt_dataset["test"]["source"]
    )

    gt_clips = gts[split_id_test]
    action_clips = actions[split_id_test]
    source_clips = sources[split_id_test]

    # IMPORTANT:
    # GT 不乘 2.5d_factor。
    # 与正式 H36M evaluation 保持一致。
    gt = gt_clips[sample_idx].copy()
    gt = gt - gt[:, 0:1, :]

    return (
        gt,
        action_clips[sample_idx],
        source_clips[sample_idx],
    )


def h36m_prepare_pred(pred, datareader, sample_idx):
    """
    Prepare prediction for qualitative visualization of one H36M clip.

    The original datareader.denormalize() expects predictions for all
    test clips because it matches test_hw with shape [N_test, 2].
    Here we manually denormalize only the selected sample.
    """

    _, split_id_test = datareader.get_split_id()

    # ---------------------------------------------------------
    # 1. Get image resolution (width, height) for this clip
    # ---------------------------------------------------------
    test_hw = datareader.get_hw()
    res_w, res_h = test_hw[sample_idx]

    # ---------------------------------------------------------
    # 2. Reshape prediction to [T, J, 3]
    # ---------------------------------------------------------
    pred = np.asarray(pred).copy()

    if pred.ndim == 2:
        # [T, 51] -> [T, 17, 3]
        pred = pred.reshape(-1, 17, 3)
    elif pred.ndim != 3:
        raise ValueError(
            f"Unexpected prediction shape: {pred.shape}. "
            f"Expected [T, 51] or [T, 17, 3]."
        )

    # ---------------------------------------------------------
    # 3. Denormalize (x, y, z)
    #    Same operation as datareader.denormalize()
    # ---------------------------------------------------------
    pred[:, :, :2] = (
        pred[:, :, :2] + np.array([1.0, res_h / res_w])
    ) * res_w / 2.0

    pred[:, :, 2:] = pred[:, :, 2:] * res_w / 2.0

    # ---------------------------------------------------------
    # 4. Apply H36M 2.5D factor
    # ---------------------------------------------------------
    factors = np.array(datareader.dt_dataset["test"]["2.5d_factor"])
    factor = factors[split_id_test][sample_idx][:, None, None]

    pred = pred * factor

    # ---------------------------------------------------------
    # 5. Convert to root-relative coordinates
    # ---------------------------------------------------------
    pred = pred - pred[:, 0:1, :]

    return pred

def frame_mpjpe(pred, gt):
    """
    pred: [T, J, 3]
    gt:   [T, J, 3]

    return:
        [T]，每一帧一个 MPJPE
    """
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)

# -------------------------------------------------------------------------
# MPI-INF-3DHP
# -------------------------------------------------------------------------

def collect_mpi_inputs(args, sample_indices):
    dataset = Fusion(args, train=False)
    batch_size = getattr(args, "test_batch_size", getattr(args, "batch_size", 8))
    loader = make_loader(dataset, batch_size, num_workers=0)
    wanted = set(sample_indices)
    samples = {}
    global_idx = 0

    for data in loader:
        _, gt_3d, input_2d, seq, scale, _ = data
        bsz = input_2d.shape[0]
        for i in range(bsz):
            idx = global_idx + i
            if idx in wanted:
                scale_i = scale[i].clone() if torch.is_tensor(scale[i]) else scale[i]
                samples[idx] = {
                    "input_2d": input_2d[i].clone(),
                    "gt_3d": gt_3d[i].clone(),
                    "seq": seq[i],
                    "scale": scale_i,
                }
        global_idx += bsz
        if len(samples) == len(wanted):
            break

    missing = wanted - set(samples)
    if missing:
        raise IndexError(f"MPI sample indices not found: {sorted(missing)}")
    return samples


def mpi_predict_one(model, sample, args, device):
    input_2d = sample["input_2d"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )

    scale = sample["scale"]

    if not torch.is_tensor(scale):
        scale = torch.tensor(
            scale,
            dtype=torch.float32,
        )

    scale = scale.reshape(1).to(
        device=device,
        dtype=torch.float32,
    )

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    if input_2d.ndim != 5:
        raise ValueError(
            f"MPI input must be [B,2,T,J,C], "
            f"got {tuple(input_2d.shape)}"
        )

    input_non_flip = input_2d[:, 0]
    input_flip = input_2d[:, 1]

    with torch.no_grad():

        # Same as input_augmentation()
        out_flip = model(input_flip)

        out_flip[..., 0] *= -1

        out_flip[:, :, joints_left + joints_right, :] = \
            out_flip[:, :, joints_right + joints_left, :]

        out_non_flip = model(input_non_flip)

        output = (
            out_non_flip + out_flip
        ) / 2.0

    # Same as official evaluation:
    # output_3D = output_3D * scale.unsqueeze(...)

    output = output * scale[:, None, None, None]

    # Same center-frame selection
    n_frames = getattr(
        args,
        "n_frames",
        output.shape[1],
    )

    pad = (n_frames - 1) // 2

    pred = output[:, pad].unsqueeze(1)

    # Same as official evaluation
    pred[..., 14, :] = 0

    # Same denormalization route
    pred = denormalize(
        pred.detach().cpu(),
        [sample["seq"]],
    )

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    pred = np.asarray(pred)

    # [1, 1, 17, 3] -> [17, 3]
    pred = pred[0, 0]

    # Same as:
    # pred_out_rel = pred_out - pred_out[..., 14:15, :]
    pred = pred - pred[14:15, :]

    return pred

def mpi_gt_for_sample(sample, args):
    """
    MPI-INF-3DHP qualitative GT.

    Exactly follows the GT processing in the official
    evaluation() function.
    """
    gt = sample["gt_3d"]

    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()

    gt = np.asarray(gt)

    # Same as:
    # out_target = gt_3D.clone().view(N, -1, 17, 3)
    gt = gt.reshape(-1, 17, 3)

    # Fusion(test) provides the target GT corresponding to this sample.
    # We use the same target-frame convention as evaluation.
    if gt.shape[0] == 1:
        gt = gt[0]
    else:
        pad = (getattr(args, "n_frames", 243) - 1) // 2
        gt = gt[pad]

    gt = gt.copy()

    # Same as:
    # out_target[:, :, 14] = 0
    gt[14, :] = 0

    # Same as:
    # out_target_rel = out_target - out_target[..., 14:15, :]
    #
    # Since joint 14 has already been set to zero,
    # this operation does not change the GT.
    gt = gt - gt[14:15, :]

    return gt
# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def shared_limits(poses):
    points = np.concatenate([np.asarray(p).reshape(-1, 3) for p in poses], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins) / 2.0) * 1.15, 1e-3)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def _visual_pose(pose):
    """
    Convert H36M camera/image coordinates to a more intuitive
    3D visualization coordinate system.

    Original:
        x = image horizontal
        y = image vertical
        z = depth

    Visualization:
        X = image horizontal
        Y = depth
        Z = vertical
    """
    pose = np.asarray(pose, dtype=np.float32).copy()

    x = pose[:, 0]
    y = pose[:, 1]
    z = pose[:, 2]

    # Put depth on the visualization Y-axis
    # and vertical direction on the visualization Z-axis.
    visual_pose = np.stack(
        [
            x,
            z,
            -y,
        ],
        axis=1,
    )

    return visual_pose




def draw_pose(ax, pose, skeleton, panel_title, elev, azim, pose_color):
    pose = _visual_pose(pose)

    for a, b in skeleton:
        if a >= len(pose) or b >= len(pose):
            continue

        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            [pose[a, 2], pose[b, 2]],
            color=pose_color,
            linewidth=2.8,
            solid_capstyle="round",
        )

    ax.scatter(
        pose[:, 0],
        pose[:, 1],
        pose[:, 2],
        s=24,
        c=pose_color,
        edgecolors="white",
        linewidths=0.35,
        depthshade=True,
    )

    ax.set_title(panel_title, fontsize=12, pad=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))

    ax.grid(True, linewidth=0.5, alpha=0.35)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
def set_pose_limits(ax, poses, margin=0.10):
    """
    为同一组 GT / Teacher / Student / KD Student
    设置统一且紧凑的 3D 坐标范围。
    """
    poses = [np.asarray(p) for p in poses]

    all_points = np.concatenate(poses, axis=0)

    xyz_min = all_points.min(axis=0)
    xyz_max = all_points.max(axis=0)

    center = (xyz_min + xyz_max) / 2.0
    span = xyz_max - xyz_min

    # 三个坐标轴采用统一尺度，保持人体比例
    max_span = np.max(span)

    # 留出一定边距
    half = max_span * (1.0 + margin) / 2.0

    ax.set_xlim(
        center[0] - half,
        center[0] + half
    )

    ax.set_ylim(
        center[1] - half,
        center[1] + half
    )

    ax.set_zlim(
        center[2] - half,
        center[2] + half
    )
    
def render_comparison(poses, skeleton, output_path, title, elev, azim):
    display_poses = {k: _visual_pose(v) for k, v in poses.items()}

    labels = [
        ("gt", "Ground Truth", "#333333"),
        ("teacher", "Teacher", "#1F77B4"),
        ("student", "Student", "#FF7F0E"),
        ("kd", "KD Student", "#2CA02C"),
    ]

    fig = plt.figure(figsize=(16, 4.5), dpi=180)
    pose_list = list(display_poses.values())

    for i, (key, label, color) in enumerate(labels, start=1):
        ax = fig.add_subplot(
            1, 4, i,
            projection="3d"
        )

        draw_pose(
            ax,
            poses[key],
            skeleton,
            label,
            elev,
            azim,
            color,
        )

        set_pose_limits(
            ax,
            pose_list,
            margin=0.10,
        )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.02,
        top=0.94,
        wspace=0.02,
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVE] {output_path}")

def render_qualitative_grid(rows, skeleton, output_path, elev, azim):
    labels = [
        ("gt", "Ground Truth", "#333333"),
        ("teacher", "Teacher", "#1F77B4"),
        ("student", "Student", "#FF7F0E"),
        ("kd", "KD Student", "#2CA02C"),
    ]

    if not rows:
        raise ValueError("No valid rows to render.")

    n_rows = len(rows)

    row_limits = []

    for row in rows:
        display_poses = [
            _visual_pose(row["poses"]["gt"]),
            _visual_pose(row["poses"]["teacher"]),
            _visual_pose(row["poses"]["student"]),
            _visual_pose(row["poses"]["kd"]),
        ]

        points = np.concatenate(display_poses, axis=0)

        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)

        center = (xyz_min + xyz_max) / 2.0
        span = xyz_max - xyz_min

        max_span = max(float(np.max(span)), 1e-3)
        half = max_span * 1.10 / 2.0

        row_limits.append(
            (
                (center[0] - half, center[0] + half),
                (center[1] - half, center[1] + half),
                (center[2] - half, center[2] + half),
            )
        )

    fig = plt.figure(
        figsize=(15, 4.0 * n_rows),
        dpi=300,
    )

    for r, row in enumerate(rows):

        for c, (key, label, color) in enumerate(labels):

            ax = fig.add_subplot(
                n_rows,
                4,
                r * 4 + c + 1,
                projection="3d",
            )

            panel_title = label if r == 0 else ""

            draw_pose(
                ax,
                row["poses"][key],
                skeleton,
                panel_title,
                elev,
                azim,
                color,
            )

            limits = row_limits[r]

            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_zlim(*limits[2])

            if c == 0:
                ax.text2D(
                    -0.16,
                    0.52,
                    row["action"],
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    va="center",
                    ha="center",
                )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.02,
        top=0.96,
        wspace=0.02,
        hspace=0.02,
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    print(f"[SAVE] {output_path}")
    
def uniform_indices(length, count):
    count = max(1, min(int(count), length))
    if count == 1:
        return [length // 2]
    return np.linspace(0, length - 1, count, dtype=int).tolist()


# -------------------------------------------------------------------------
# Dataset runners
# -------------------------------------------------------------------------
def search_h36m_candidates(cli, device):
    """
    H36M 定性可视化自动搜索 + 逐帧诊断。

    不使用严格硬阈值。
    对整个 H36M test set 的所有 frame 进行统计，
    再根据：
        1. Teacher 质量
        2. Student 相对 Teacher 的差异
        3. KD 相对 Student 的恢复
        4. KD 与 Teacher 的接近程度

    自动选择适合论文定性展示的 Top-K 帧。
    """

    ensure_paths("h36m")

    # =========================================================
    # 1. 加载模型
    # =========================================================
    print("\n========== Loading models ==========")

    teacher, t_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["teacher"]["config"],
        CONFIGS["h36m"]["teacher"]["checkpoint"],
        device,
    )

    student, s_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["student"]["config"],
        CONFIGS["h36m"]["student"]["checkpoint"],
        device,
    )

    kd, k_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["kd"]["config"],
        CONFIGS["h36m"]["kd"]["checkpoint"],
        device,
    )

    data_args = t_args

    # =========================================================
    # 2. DataReader
    # =========================================================
    datareader = DataReaderH36M(
        n_frames=data_args.n_frames,
        sample_stride=1,
        data_stride_train=data_args.n_frames // 3,
        data_stride_test=data_args.n_frames,
        dt_root="data/motion3d",
        dt_file=data_args.dt_file,
    )

    # =========================================================
    # 3. Test dataset
    # =========================================================
    dataset = MotionDataset3D(
        data_args,
        data_args.subset_list,
        "test",
    )

    loader = make_loader(
        dataset,
        cli.search_batch_size,
        num_workers=0,
    )

    # =========================================================
    # 4. 统计变量
    # =========================================================

    results = []

    total_frames = 0

    # Student vs Teacher
    student_better_count = 0
    student_worse_count = 0

    # KD vs Student
    kd_better_count = 0
    kd_worse_count = 0

    # 三者关系
    both_positive_count = 0
    student_worse_kd_better_count = 0

    # 平均误差
    sum_teacher_error = 0.0
    sum_student_error = 0.0
    sum_kd_error = 0.0

    # 最大恢复 / 最大退化
    best_recovery = None
    best_degradation = None

    global_idx = 0

    print("\n========== Searching H36M candidates ==========")
    print(f"Total test samples: {len(dataset)}")
    print(f"Batch size: {cli.search_batch_size}")
    if cli.search_max_batches > 0:
        print(f"Max search batches: {cli.search_max_batches}")

    # =========================================================
    # 6. 遍历测试集
    # =========================================================
    for batch_id, (x, y) in enumerate(loader):

        if cli.search_max_batches > 0 and batch_id >= cli.search_max_batches:
            print(
                f"[SEARCH] Reached search-max-batches="
                f"{cli.search_max_batches}; stopping early."
            )
            break

        batch_size = x.shape[0]

        # =====================================================
        # 每个 sample
        # =====================================================
        # Search uses exactly the same single-sample inference path
        # as the final qualitative visualization path.
        for b in range(batch_size):

            sample_idx = global_idx + b
            x_one = x[b]

            # -------------------------------------------------
            # GT
            # -------------------------------------------------
            # Reuse the same GT reconstruction used by run_h36m().
            gt, action, source = h36m_gt_for_sample(
                datareader,
                sample_idx,
            )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            pred_t = h36m_prepare_pred(
                h36m_predict_one(
                    teacher,
                    x_one,
                    t_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            pred_s = h36m_prepare_pred(
                h36m_predict_one(
                    student,
                    x_one,
                    s_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            pred_kd = h36m_prepare_pred(
                h36m_predict_one(
                    kd,
                    x_one,
                    k_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            # -------------------------------------------------
            # Coordinate / scale sanity check
            # -------------------------------------------------
            if sample_idx == 0 and batch_id == 0:
                print("\n========== H36M SEARCH DEBUG ==========")
                print("GT shape:", gt.shape)
                print("Teacher shape:", pred_t.shape)
                print("Student shape:", pred_s.shape)
                print("KD shape:", pred_kd.shape)

                print("\nGT range:")
                print("  min =", float(gt.min()))
                print("  max =", float(gt.max()))

                print("\nTeacher range:")
                print("  min =", float(pred_t.min()))
                print("  max =", float(pred_t.max()))

                print("\nStudent range:")
                print("  min =", float(pred_s.min()))
                print("  max =", float(pred_s.max()))

                print("\nKD range:")
                print("  min =", float(pred_kd.min()))
                print("  max =", float(pred_kd.max()))

                print("\nMPJPE:")
                print("  Teacher =", float(frame_mpjpe(pred_t, gt).mean()))
                print("  Student =", float(frame_mpjpe(pred_s, gt).mean()))
                print("  KD      =", float(frame_mpjpe(pred_kd, gt).mean()))
                print("=======================================\n")

            # -------------------------------------------------
            # Frame-wise MPJPE
            # -------------------------------------------------
            err_t = frame_mpjpe(
                pred_t,
                gt,
            )

            err_s = frame_mpjpe(
                pred_s,
                gt,
            )

            err_kd = frame_mpjpe(
                pred_kd,
                gt,
            )

            # -------------------------------------------------
            # 差异
            # -------------------------------------------------

            # > 0:
            # Student 比 Teacher 差
            degradation = err_s - err_t

            # > 0:
            # KD 比 Student 好
            recovery = err_s - err_kd

            # KD 和 Teacher 的距离
            kd_gap = np.abs(err_kd - err_t)

            # =================================================
            # 更新统计
            # =================================================

            for frame_id in range(len(err_t)):

                teacher_error = float(
                    err_t[frame_id]
                )

                student_error = float(
                    err_s[frame_id]
                )

                kd_error = float(
                    err_kd[frame_id]
                )

                deg = float(
                    degradation[frame_id]
                )

                rec = float(
                    recovery[frame_id]
                )

                gap = float(
                    kd_gap[frame_id]
                )

                # -------------------------------------------------
                # NaN / Inf
                # -------------------------------------------------
                if not (
                    np.isfinite(teacher_error)
                    and np.isfinite(student_error)
                    and np.isfinite(kd_error)
                ):
                    continue

                total_frames += 1

                sum_teacher_error += teacher_error
                sum_student_error += student_error
                sum_kd_error += kd_error

                # -------------------------------------------------
                # Student vs Teacher
                # -------------------------------------------------
                if deg > 0:
                    student_worse_count += 1
                else:
                    student_better_count += 1

                # -------------------------------------------------
                # KD vs Student
                # -------------------------------------------------
                if rec > 0:
                    kd_better_count += 1
                else:
                    kd_worse_count += 1

                # -------------------------------------------------
                # 两个条件同时成立
                # -------------------------------------------------
                if deg > 0 and rec > 0:
                    both_positive_count += 1

                    student_worse_kd_better_count += 1

                # =================================================
                # 记录最大 Student 退化
                # =================================================
                if (
                    best_degradation is None
                    or deg > best_degradation["degradation"]
                ):

                    best_degradation = {
                        "sample_idx": sample_idx,
                        "frame_id": frame_id,
                        "teacher_error": teacher_error,
                        "student_error": student_error,
                        "kd_error": kd_error,
                        "degradation": deg,
                        "recovery": rec,
                        "gap": gap,
                    }

                # =================================================
                # 记录最大 KD 恢复
                # =================================================
                if (
                    best_recovery is None
                    or rec > best_recovery["recovery"]
                ):

                    best_recovery = {
                        "sample_idx": sample_idx,
                        "frame_id": frame_id,
                        "teacher_error": teacher_error,
                        "student_error": student_error,
                        "kd_error": kd_error,
                        "degradation": deg,
                        "recovery": rec,
                        "gap": gap,
                    }

                # =================================================
                # 生成候选
                #
                # 不再硬性要求：
                #     Student > Teacher
                #     KD < Student
                #
                # 而是所有有效帧进入排序。
                # =================================================

                # Teacher 太差的极端帧不适合论文展示
                if teacher_error > cli.max_teacher_error:
                    continue

                # Student / KD 极端异常
                if student_error > cli.max_student_error:
                    continue

                if kd_error > cli.max_kd_error:
                    continue

                # 只保留“误差可控、Student 有退化、KD 有恢复”的帧。
                if deg < cli.min_degradation:
                    continue
                if rec < cli.min_recovery:
                    continue
                if kd_error > cli.max_kd_error:
                    continue

                # -------------------------------------------------
                # 定性评分
                #
                # A: Student 相对 Teacher 的退化
                # B: KD 相对 Student 的恢复
                # C: Teacher 本身的质量
                # D: KD 与 Teacher 的接近程度
                # -------------------------------------------------

                score = (
                    cli.degradation_weight * deg
                    + cli.recovery_weight * rec
                    - cli.teacher_penalty_weight * teacher_error
                    - cli.kd_gap_penalty_weight * gap
                )

                # -------------------------------------------------
                # action/source
                # -------------------------------------------------
                action_value = action[0] if np.ndim(action) else action
                source_value = source[0] if np.ndim(source) else source

                results.append({
                    "score": float(score),

                    "sample_idx": int(sample_idx),

                    "frame_id": int(frame_id),

                    "teacher_error": teacher_error,

                    "student_error": student_error,

                    "kd_error": kd_error,

                    "degradation": deg,

                    "recovery": rec,

                    "gap": gap,

                    "gt": gt[frame_id].copy(),

                    "teacher": pred_t[frame_id].copy(),

                    "student": pred_s[frame_id].copy(),

                    "kd": pred_kd[frame_id].copy(),

                    "action": str(action_value),

                    "source": str(source_value),
                })

        global_idx += batch_size

        if (batch_id + 1) % 20 == 0:

            print(
                f"[SEARCH] batch={batch_id + 1}, "
                f"processed={global_idx}, "
                f"candidates={len(results)}"
            )

    # =========================================================
    # 7. 打印诊断统计
    # =========================================================

    print("\n")
    print("=" * 70)
    print("H36M FRAME-WISE DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Total valid frames: {total_frames}"
    )

    if total_frames > 0:

        print(
            f"Average Teacher MPJPE: "
            f"{sum_teacher_error / total_frames:.3f} mm"
        )

        print(
            f"Average Student MPJPE: "
            f"{sum_student_error / total_frames:.3f} mm"
        )

        print(
            f"Average KD MPJPE: "
            f"{sum_kd_error / total_frames:.3f} mm"
        )

        print()

        print(
            f"Student worse than Teacher: "
            f"{student_worse_count} "
            f"({100.0 * student_worse_count / total_frames:.2f}%)"
        )

        print(
            f"Student better/equal Teacher: "
            f"{student_better_count} "
            f"({100.0 * student_better_count / total_frames:.2f}%)"
        )

        print()

        print(
            f"KD better than Student: "
            f"{kd_better_count} "
            f"({100.0 * kd_better_count / total_frames:.2f}%)"
        )

        print(
            f"KD worse/equal Student: "
            f"{kd_worse_count} "
            f"({100.0 * kd_worse_count / total_frames:.2f}%)"
        )

        print()

        print(
            f"Student worse + KD better: "
            f"{student_worse_kd_better_count} "
            f"({100.0 * student_worse_kd_better_count / total_frames:.2f}%)"
        )

    # =========================================================
    # 8. 最大退化 / 最大恢复
    # =========================================================

    if best_degradation is not None:

        print("\n--- Maximum Student degradation ---")

        print(
            f"sample={best_degradation['sample_idx']}, "
            f"frame={best_degradation['frame_id']}"
        )

        print(
            f"Teacher={best_degradation['teacher_error']:.3f}, "
            f"Student={best_degradation['student_error']:.3f}, "
            f"KD={best_degradation['kd_error']:.3f}"
        )

        print(
            f"Degradation="
            f"{best_degradation['degradation']:.3f}, "
            f"Recovery="
            f"{best_degradation['recovery']:.3f}"
        )

    if best_recovery is not None:

        print("\n--- Maximum KD recovery ---")

        print(
            f"sample={best_recovery['sample_idx']}, "
            f"frame={best_recovery['frame_id']}"
        )

        print(
            f"Teacher={best_recovery['teacher_error']:.3f}, "
            f"Student={best_recovery['student_error']:.3f}, "
            f"KD={best_recovery['kd_error']:.3f}"
        )

        print(
            f"Degradation="
            f"{best_recovery['degradation']:.3f}, "
            f"Recovery="
            f"{best_recovery['recovery']:.3f}"
        )

    # =========================================================
    # 9. 排序
    # =========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # =========================================================
    # 10. 去除同一 sample 的相邻帧
    # =========================================================

    selected = []

    for item in results:

        too_close = False

        for old in selected:

            if (
                item["sample_idx"]
                ==
                old["sample_idx"]
            ):

                distance = abs(
                    item["frame_id"]
                    -
                    old["frame_id"]
                )

                if distance < cli.min_frame_gap:

                    too_close = True
                    break

        if not too_close:

            selected.append(item)

        if len(selected) >= cli.top_k:

            break

    # =========================================================
    # 11. 输出目录
    # =========================================================

    output_dir = (
        Path(cli.output)
        /
        "h36m_search"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # 12. 保存详细结果
    # =========================================================

    txt_path = (
        output_dir
        /
        "search_results.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "H36M qualitative candidate search results\n"
        )

        f.write(
            "=" * 120
            +
            "\n"
        )

        f.write(
            "Rank | Sample | Frame | Score | "
            "Teacher | Student | KD | "
            "Degradation | Recovery | Gap | "
            "Action | Source\n"
        )

        f.write(
            "-" * 120
            +
            "\n"
        )

        for rank, item in enumerate(
            selected,
            1,
        ):

            line = (
                f"[{rank:02d}] "
                f"sample={item['sample_idx']:05d}, "
                f"frame={item['frame_id']:03d}, "
                f"score={item['score']:.2f}, "
                f"Teacher={item['teacher_error']:.2f}, "
                f"Student={item['student_error']:.2f}, "
                f"KD={item['kd_error']:.2f}, "
                f"Degradation={item['degradation']:.2f}, "
                f"Recovery={item['recovery']:.2f}, "
                f"Gap={item['gap']:.2f}, "
                f"Action={item['action']}, "
                f"Source={item['source']}\n"
            )

            print(line.strip())

            f.write(line)

    # =========================================================
    # 13. 保存 Top-K 图
    # =========================================================

    for rank, item in enumerate(
        selected,
        1,
    ):

        poses = {
            "gt": item["gt"],
            "teacher": item["teacher"],
            "student": item["student"],
            "kd": item["kd"],
        }

        title = (
            f"H36M | "
            f"Sample {item['sample_idx']} | "
            f"Frame {item['frame_id']} | "
            f"T {item['teacher_error']:.1f} | "
            f"S {item['student_error']:.1f} | "
            f"KD {item['kd_error']:.1f}"
        )

        out_path = (
            output_dir
            /
            f"top_{rank:02d}_"
            f"sample_{item['sample_idx']:05d}_"
            f"frame_{item['frame_id']:03d}.png"
        )

        render_comparison(
            poses,
            H36M_SKELETON,
            str(out_path),
            title,
            cli.elev,
            cli.azim,
        )

    # =========================================================
    # 14. 最终信息
    # =========================================================

    print(
        f"\n[SAVE] {txt_path}"
    )

    print(
        "\n========== Search finished =========="
    )

    print(
        f"Valid candidates: {len(results)}"
    )

    print(
        f"Selected candidates: {len(selected)}"
    )

    print(
        f"Results saved to: {output_dir}"
    )
    
    
def run_h36m(cli, device):
    ensure_paths("h36m")

    teacher, t_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["teacher"]["config"],
        CONFIGS["h36m"]["teacher"]["checkpoint"],
        device,
    )
    student, s_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["student"]["config"],
        CONFIGS["h36m"]["student"]["checkpoint"],
        device,
    )
    kd, k_args = load_checkpoint_from_config(
        CONFIGS["h36m"]["kd"]["config"],
        CONFIGS["h36m"]["kd"]["checkpoint"],
        device,
    )

    data_args = t_args
    datareader = DataReaderH36M(
        n_frames=data_args.n_frames,
        sample_stride=1,
        data_stride_train=data_args.n_frames // 3,
        data_stride_test=data_args.n_frames,
        dt_root="data/motion3d",
        dt_file=data_args.dt_file,
    )

    # =========================================================
    # Final mode:
    # explicitly pair each sample with one frame.
    #
    # Example:
    #   samples = 1590,1427,703
    #   frames  = 30,34,106
    #
    # -> (1590,30), (1427,34), (703,106)
    # =========================================================
    if cli.mode == "final":
        if len(cli.frames) not in (1, len(cli.sample_indices)):
            raise ValueError(
                "In final mode, --frames must contain either one frame "
                "or exactly one frame for each --sample-indices value.\n"
                "Example:\n"
                "  --sample-indices \"1590,1427,703\" "
                "--frames \"30,34,106\""
            )

        samples = collect_h36m_inputs(
            data_args,
            cli.sample_indices,
        )

        rows = []

        for pos, sample_idx in enumerate(cli.sample_indices):
            x = samples[sample_idx]["x"]

            gt, action, source = h36m_gt_for_sample(
                datareader,
                sample_idx,
            )

            pred_t = h36m_prepare_pred(
                h36m_predict_one(
                    teacher,
                    x,
                    t_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            pred_s = h36m_prepare_pred(
                h36m_predict_one(
                    student,
                    x,
                    s_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            pred_kd = h36m_prepare_pred(
                h36m_predict_one(
                    kd,
                    x,
                    k_args,
                    device,
                ),
                datareader,
                sample_idx,
            )

            frame = (
                cli.frames[0]
                if len(cli.frames) == 1
                else cli.frames[pos]
            )

            max_frame = min(
                len(gt),
                len(pred_t),
                len(pred_s),
                len(pred_kd),
            ) - 1

            if frame < 0 or frame > max_frame:
                raise IndexError(
                    f"H36M frame {frame} out of range for sample "
                    f"{sample_idx}; valid range is 0..{max_frame}."
                )

            action_value = (
                action[0] if np.ndim(action) else action
            )
            source_value = (
                source[0] if np.ndim(source) else source
            )

            rows.append({
                "sample_idx": int(sample_idx),
                "frame": int(frame),
                "action": str(action_value),
                "source": str(source_value),
                "poses": {
                    "gt": gt[frame],
                    "teacher": pred_t[frame],
                    "student": pred_s[frame],
                    "kd": pred_kd[frame],
                },
            })

            print(
                f"[FINAL] sample={sample_idx}, "
                f"frame={frame}, action={action_value}"
            )

        output_path = os.path.join(
            cli.output,
            "h36m_qualitative_3x4.svg",
        )

        render_qualitative_grid(
            rows,
            H36M_SKELETON,
            output_path,
            cli.elev,
            cli.azim,
        )

        print(
            f"[DONE] Final H36M 3x4 figure: {output_path}"
        )
        return

    # =========================================================
    # Candidate mode:
    # preserve the original single-sample/frame behavior.
    # =========================================================
    samples = collect_h36m_inputs(
        data_args,
        cli.sample_indices,
    )

    for sample_idx in sorted(samples):
        x = samples[sample_idx]["x"]

        gt, action, source = h36m_gt_for_sample(
            datareader,
            sample_idx,
        )

        pred_t = h36m_prepare_pred(
            h36m_predict_one(
                teacher,
                x,
                t_args,
                device,
            ),
            datareader,
            sample_idx,
        )

        pred_s = h36m_prepare_pred(
            h36m_predict_one(
                student,
                x,
                s_args,
                device,
            ),
            datareader,
            sample_idx,
        )

        pred_kd = h36m_prepare_pred(
            h36m_predict_one(
                kd,
                x,
                k_args,
                device,
            ),
            datareader,
            sample_idx,
        )

        if cli.mode == "candidate":
            frame_ids = uniform_indices(
                pred_t.shape[0],
                cli.num_candidates,
            )
        else:
            frame_ids = (
                cli.frames
                or [pred_t.shape[0] // 2]
            )

        action_name = (
            str(action[0])
            if np.ndim(action)
            else str(action)
        )
        source_name = (
            str(source[0])
            if np.ndim(source)
            else str(source)
        )

        for frame_id in frame_ids:
            if frame_id < 0 or frame_id >= pred_t.shape[0]:
                print(
                    f"[WARN] H36M frame {frame_id} "
                    f"out of range; skipped."
                )
                continue

            poses = {
                "gt": gt[frame_id],
                "teacher": pred_t[frame_id],
                "student": pred_s[frame_id],
                "kd": pred_kd[frame_id],
            }

            out = os.path.join(
                cli.output,
                "h36m",
                f"sample_{sample_idx:05d}",
                f"frame_{frame_id:03d}.png",
            )

            render_comparison(
                poses,
                H36M_SKELETON,
                out,
                (
                    f"Human3.6M | sample={sample_idx} | "
                    f"frame={frame_id} | {action_name} | "
                    f"{source_name}"
                ),
                cli.elev,
                cli.azim,
            )

def search_mpi_candidates(cli, device):
    """
    MPI-INF-3DHP qualitative candidate search.

    MPI 每个 test sample 对应一个 243-frame temporal window，
    模型最终只输出中心帧，因此这里按照 sample 搜索，
    而不是按照 frame 搜索。

    排序依据：
        1. Teacher MPJPE 不宜过大
        2. Student 相比 Teacher 有明显退化
        3. KD 相比 Student 有明显恢复
        4. KD 与 Teacher 保持较近
    """

    ensure_paths("mpi")

    # ---------------------------------------------------------
    # 1. Load models
    # ---------------------------------------------------------
    teacher, t_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["teacher"]["config"],
        CONFIGS["mpi"]["teacher"]["checkpoint"],
        device,
    )

    student, s_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["student"]["config"],
        CONFIGS["mpi"]["student"]["checkpoint"],
        device,
    )

    kd, k_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["kd"]["config"],
        CONFIGS["mpi"]["kd"]["checkpoint"],
        device,
    )

    # ---------------------------------------------------------
    # 2. Collect all MPI test samples
    # ---------------------------------------------------------
    dataset = Fusion(t_args, train=False)

    batch_size = getattr(
        t_args,
        "test_batch_size",
        getattr(t_args, "batch_size", 8),
    )

    loader = make_loader(
        dataset,
        batch_size,
        num_workers=0,
    )

    results = []

    global_idx = 0

    print("\n========== MPI qualitative search ==========")

    for batch_id, data in enumerate(loader):

        _, gt_3d, input_2d, seq, scale, _ = data

        batch_size_actual = input_2d.shape[0]

        for b in range(batch_size_actual):

            sample_idx = global_idx + b

            sample = {
                "input_2d": input_2d[b].clone(),
                "gt_3d": gt_3d[b].clone(),
                "seq": seq[b],
                "scale": (
                    scale[b].clone()
                    if torch.is_tensor(scale[b])
                    else scale[b]
                ),
            }

            # -------------------------------------------------
            # GT
            # -------------------------------------------------
            # -------------------------------------------------
            # GT
            # -------------------------------------------------
            gt = mpi_gt_for_sample(
                sample,
                t_args,
            )

            # -------------------------------------------------
            # Teacher
            # -------------------------------------------------
            pred_t = mpi_predict_one(
                teacher,
                sample,
                t_args,
                device,
            )

            # -------------------------------------------------
            # Student
            # -------------------------------------------------
            pred_s = mpi_predict_one(
                student,
                sample,
                s_args,
                device,
            )

            # -------------------------------------------------
            # KD Student
            # -------------------------------------------------
            pred_kd = mpi_predict_one(
                kd,
                sample,
                k_args,
                device,
            )

            # =================================================
            # Debug: sample 0
            # =================================================
            if sample_idx == 0:
                print("\n========== MPI SEARCH DEBUG ==========")

                raw_gt = sample["gt_3d"]

                if torch.is_tensor(raw_gt):
                    raw_gt_np = raw_gt.detach().cpu().numpy()
                else:
                    raw_gt_np = np.asarray(raw_gt)

                print("raw GT shape:", raw_gt_np.shape)
                print(
                    "raw GT range:",
                    raw_gt_np.min(),
                    raw_gt_np.max(),
                )

                print("GT shape:", gt.shape)
                print(
                    "GT range:",
                    gt.min(),
                    gt.max(),
                )

                print("scale:", sample["scale"])
                print("seq:", sample["seq"])

                print("Teacher shape:", pred_t.shape)
                print("Student shape:", pred_s.shape)
                print("KD shape:", pred_kd.shape)

                print(
                    "Teacher MPJPE:",
                    np.linalg.norm(
                        pred_t - gt,
                        axis=-1
                    ).mean()
                )

                print(
                    "Student MPJPE:",
                    np.linalg.norm(
                        pred_s - gt,
                        axis=-1
                    ).mean()
                )

                print(
                    "KD MPJPE:",
                    np.linalg.norm(
                        pred_kd - gt,
                        axis=-1
                    ).mean()
                )

                print("======================================\n")

            # -------------------------------------------------
            # MPJPE
            # -------------------------------------------------
            teacher_error = float(
                np.linalg.norm(
                    pred_t - gt,
                    axis=-1
                ).mean()
            )

            student_error = float(
                np.linalg.norm(
                    pred_s - gt,
                    axis=-1
                ).mean()
            )

            kd_error = float(
                np.linalg.norm(
                    pred_kd - gt,
                    axis=-1
                ).mean()
            )

            # Student 相比 Teacher 的退化
            degradation = student_error - teacher_error

            # KD 相比 Student 的恢复
            recovery = student_error - kd_error

            # KD 与 Teacher 的误差距离
            gap = abs(kd_error - teacher_error)

            # -------------------------------------------------
            # Candidate score
            # -------------------------------------------------
            #
            # 希望：
            #   Teacher error 小
            #   degradation 大
            #   recovery 大
            #   KD gap 小
            #
            score = (
                1.0 * degradation
                + 1.5 * recovery
                - 0.5 * teacher_error
                - 0.5 * gap
            )

            results.append({
                "sample_idx": sample_idx,
                "seq": str(seq[b]),
                "gt": gt,
                "teacher": pred_t,
                "student": pred_s,
                "kd": pred_kd,
                "teacher_error": teacher_error,
                "student_error": student_error,
                "kd_error": kd_error,
                "degradation": degradation,
                "recovery": recovery,
                "gap": gap,
                "score": score,
            })

        global_idx += batch_size_actual

        if (batch_id + 1) % 20 == 0:
            print(
                f"[SEARCH] processed {global_idx} MPI samples"
            )

    # ---------------------------------------------------------
    # 3. Basic statistics
    # ---------------------------------------------------------
    print("\n========== MPI search statistics ==========")

    print(
        f"Total samples: {len(results)}"
    )

    if not results:
        raise RuntimeError(
            "No MPI samples were processed."
        )

    teacher_mean = np.mean(
        [x["teacher_error"] for x in results]
    )

    student_mean = np.mean(
        [x["student_error"] for x in results]
    )

    kd_mean = np.mean(
        [x["kd_error"] for x in results]
    )

    print(
        f"Average Teacher MPJPE: {teacher_mean:.3f} mm"
    )

    print(
        f"Average Student MPJPE: {student_mean:.3f} mm"
    )

    print(
        f"Average KD MPJPE: {kd_mean:.3f} mm"
    )

    # ---------------------------------------------------------
    # 4. Filter reasonable qualitative candidates
    # ---------------------------------------------------------
    candidates = [
        x for x in results
        if (
            x["degradation"] > 3.0
            and x["recovery"] > 3.0
            and x["teacher_error"] < 60.0
        )
    ]

    # 如果严格筛选后太少，则退化为全部排序
    if len(candidates) < cli.top_k:
        print(
            "[WARN] Too few filtered candidates; "
            "using all samples for ranking."
        )
        candidates = results

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    selected = candidates[:cli.top_k]

    # ---------------------------------------------------------
    # 5. Save ranking results
    # ---------------------------------------------------------
    output_dir = (
        Path(cli.output)
        / "mpi_search"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_path = (
        output_dir
        / "search_results.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "MPI-INF-3DHP qualitative candidate search results\n"
        )

        f.write("=" * 120 + "\n")

        f.write(
            "Rank | Sample | Score | "
            "Teacher | Student | KD | "
            "Degradation | Recovery | Gap | Sequence\n"
        )

        f.write("-" * 120 + "\n")

        for rank, item in enumerate(
            selected,
            1,
        ):

            line = (
                f"[{rank:02d}] "
                f"sample={item['sample_idx']:05d}, "
                f"score={item['score']:.2f}, "
                f"Teacher={item['teacher_error']:.2f}, "
                f"Student={item['student_error']:.2f}, "
                f"KD={item['kd_error']:.2f}, "
                f"Degradation={item['degradation']:.2f}, "
                f"Recovery={item['recovery']:.2f}, "
                f"Gap={item['gap']:.2f}, "
                f"Sequence={item['seq']}\n"
            )

            print(line.strip())
            f.write(line)

    # ---------------------------------------------------------
    # 6. Save Top-K candidate images
    # ---------------------------------------------------------
    for rank, item in enumerate(
        selected,
        1,
    ):

        poses = {
            "gt": item["gt"],
            "teacher": item["teacher"],
            "student": item["student"],
            "kd": item["kd"],
        }

        title = (
            f"MPI | Sample {item['sample_idx']} | "
            f"T {item['teacher_error']:.1f} | "
            f"S {item['student_error']:.1f} | "
            f"KD {item['kd_error']:.1f}"
        )

        out_path = (
            output_dir
            / f"top_{rank:02d}_"
            f"sample_{item['sample_idx']:05d}.png"
        )

        render_comparison(
            poses,
            MPI_SKELETON,
            str(out_path),
            title,
            cli.elev,
            cli.azim,
        )

    print(f"\n[SAVE] {txt_path}")

    print(
        f"[SAVE] MPI candidates: {output_dir}"
    )

    print(
        f"[DONE] Selected {len(selected)} candidates."
    )
    
def run_mpi(cli, device):
    ensure_paths("mpi")

    teacher, t_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["teacher"]["config"],
        CONFIGS["mpi"]["teacher"]["checkpoint"],
        device,
    )

    student, s_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["student"]["config"],
        CONFIGS["mpi"]["student"]["checkpoint"],
        device,
    )

    kd, k_args = load_checkpoint_from_config(
        CONFIGS["mpi"]["kd"]["config"],
        CONFIGS["mpi"]["kd"]["checkpoint"],
        device,
    )

    # =========================================================
    # Final mode:
    # 3 MPI samples -> one 3x4 figure
    #
    # Row 1: sample 02056
    # Row 2: sample 00871
    # Row 3: sample 01092
    #
    # Columns:
    # Ground Truth | Teacher | Student | KD Student
    # =========================================================
    if cli.mode == "final":

        samples = collect_mpi_inputs(
            t_args,
            cli.sample_indices,
        )

        rows = []

        for sample_idx in cli.sample_indices:

            if sample_idx not in samples:
                raise IndexError(
                    f"MPI sample {sample_idx} was not found."
                )

            sample = samples[sample_idx]

            # -------------------------------------------------
            # Ground Truth
            # -------------------------------------------------
            gt = mpi_gt_for_sample(
                sample,
                t_args,
            )

            # -------------------------------------------------
            # Teacher
            # -------------------------------------------------
            pred_t = mpi_predict_one(
                teacher,
                sample,
                t_args,
                device,
            )

            # -------------------------------------------------
            # Student
            # -------------------------------------------------
            pred_s = mpi_predict_one(
                student,
                sample,
                s_args,
                device,
            )

            # -------------------------------------------------
            # KD Student
            # -------------------------------------------------
            pred_kd = mpi_predict_one(
                kd,
                sample,
                k_args,
                device,
            )

            # -------------------------------------------------
            # Calculate MPJPE for displaying in row label
            # -------------------------------------------------
            teacher_error = float(
                np.linalg.norm(
                    pred_t - gt,
                    axis=-1,
                ).mean()
            )

            student_error = float(
                np.linalg.norm(
                    pred_s - gt,
                    axis=-1,
                ).mean()
            )

            kd_error = float(
                np.linalg.norm(
                    pred_kd - gt,
                    axis=-1,
                ).mean()
            )

            # Sequence name
            seq_name = str(sample["seq"])

            rows.append({
                "sample_idx": int(sample_idx),
                "action": seq_name,
                "source": seq_name,
                "teacher_error": teacher_error,
                "student_error": student_error,
                "kd_error": kd_error,
                "poses": {
                    "gt": gt,
                    "teacher": pred_t,
                    "student": pred_s,
                    "kd": pred_kd,
                },
            })

            print(
                f"[FINAL] sample={sample_idx}, "
                f"sequence={seq_name}, "
                f"Teacher={teacher_error:.2f}, "
                f"Student={student_error:.2f}, "
                f"KD={kd_error:.2f}"
            )
        # =====================================================
        # Render ONE 3x4 figure
        # =====================================================
        output_path = os.path.join(
            cli.output,
            "mpi_qualitative_3x4.svg",
        )

        render_qualitative_grid(
            rows,
            MPI_SKELETON,
            output_path,
            cli.elev,
            cli.azim,
        )

        print(
            f"[DONE] Final MPI 3x4 figure: {output_path}"
        )

        return

    # =========================================================
    # Candidate mode:
    # preserve original single-sample behavior
    # =========================================================

    samples = collect_mpi_inputs(
        t_args,
        cli.sample_indices,
    )

    for sample_idx in sorted(samples):

        sample = samples[sample_idx]

        gt = mpi_gt_for_sample(
            sample,
            t_args,
        )

        pred_t = mpi_predict_one(
            teacher,
            sample,
            t_args,
            device,
        )

        pred_s = mpi_predict_one(
            student,
            sample,
            s_args,
            device,
        )

        pred_kd = mpi_predict_one(
            kd,
            sample,
            k_args,
            device,
        )

        poses = {
            "gt": gt,
            "teacher": pred_t,
            "student": pred_s,
            "kd": pred_kd,
        }

        out = os.path.join(
            cli.output,
            "mpi",
            f"sample_{sample_idx:05d}.png",
        )

        render_comparison(
            poses,
            MPI_SKELETON,
            out,
            f"MPI-INF-3DHP | test sample={sample_idx} | center frame",
            cli.elev,
            cli.azim,
        )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["h36m", "mpi", "both"], default="both")
    p.add_argument(
    "--mode",
    choices=["candidate", "search", "final"],
    default="candidate",
    help="candidate=uniform frames; search=automatic qualitative search; final=paired sample/frame 3x4 figure."
)
    p.add_argument(
        "--sample-indices",
        type=str,
        default="0",
        help="Global test sample indices, e.g. 0,10,20",
    )
    p.add_argument(
        "--frames",
        type=str,
        default="121",
        help="H36M local frame indices, e.g. 60,90,121,150,180",
    )
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--output", type=str, default="visualization/qualitative")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument("--azim", type=float, default=-70.0)
    p.add_argument(
    "--search-batch-size",
    type=int,
    default=4,
    help="Batch size used during H36M automatic search."
)

    p.add_argument(
        "--search-max-batches",
        type=int,
        default=0,
        help="Maximum H36M search batches; 0 means the full test set.",
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of qualitative candidates to save."
    )

    p.add_argument(
        "--max-teacher-error",
        type=float,
        default=60.0,
        help="Maximum Teacher MPJPE allowed for a candidate."
    )

    p.add_argument("--max-student-error", type=float, default=100.0,
                   help="Maximum Student MPJPE allowed for a qualitative frame.")
    p.add_argument("--max-kd-error", type=float, default=80.0,
                   help="Maximum KD Student MPJPE allowed for a qualitative frame.")
    p.add_argument("--teacher-penalty-weight", type=float, default=0.10)
    p.add_argument("--kd-gap-penalty-weight", type=float, default=0.20)

    p.add_argument(
        "--min-degradation",
        type=float,
        default=5.0,
        help="Minimum Student-Teacher MPJPE gap."
    )

    p.add_argument(
        "--min-recovery",
        type=float,
        default=3.0,
        help="Minimum Student-KD MPJPE improvement."
    )

    p.add_argument(
        "--degradation-weight",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--recovery-weight",
        type=float,
        default=1.5
    )

    p.add_argument(
        "--min-frame-gap",
        type=int,
        default=15,
        help="Minimum frame distance between selected candidates from the same sample."
    )
    return p


def main():
    cli = build_parser().parse_args()

    cli.sample_indices = parse_int_list(
        cli.sample_indices
    )

    cli.frames = parse_int_list(
        cli.frames
    )

    if not cli.sample_indices:
        raise ValueError(
            "--sample-indices must contain at least one index."
        )

    if cli.device == "cuda" and not torch.cuda.is_available():
        print(
            "[WARN] CUDA requested but is not available. "
            "Falling back to CPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(cli.device)

    print(f"[INFO] Device: {device}")

    # =========================================================
    # H36M
    # =========================================================
    if cli.dataset in ("h36m", "both"):

        if cli.mode == "search":
            search_h36m_candidates(
                cli,
                device
            )
        else:
            run_h36m(
                cli,
                device
            )

    # =========================================================
    # MPI
    # =========================================================
    if cli.dataset in ("mpi", "both"):
        if cli.mode == "search":
            search_mpi_candidates(cli, device)
        else:
            run_mpi(cli, device)


if __name__ == "__main__":
    main()