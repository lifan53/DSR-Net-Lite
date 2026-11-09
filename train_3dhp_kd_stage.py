import argparse
import os
import pkg_resources

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import wandb

from torch import nn
from torch import optim
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

from loss.pose3d import (
    loss_mpjpe,
    n_mpjpe,
    loss_velocity,
    loss_limb_var,
    loss_limb_gt,
    loss_angle,
    loss_angle_velocity
)

from utils.data import denormalize

from data.reader.motion_dataset import (
    MPI3DHP,
    Fusion
)

from utils.tools import (
    set_random_seed,
    get_config,
    print_args,
    create_directory_if_not_exists,
    count_param_numbers
)

from utils.learning import (
    load_model,
    AverageMeter,
    decay_lr_exponentially
)

from utils.utils_3dhp import *

from model.MotionAGFormer_DCF3_CKA import (
    MotionAGFormer_DCF3_Pruned
)


# ============================================================
# Fixed Experiment Definition
# ============================================================

# Teacher:
#   16 layers
#   128 dimensions
#
# Student:
#   10 layers
#   112 dimensions
#
# Teacher pruning:
#   delete L5-L10
#
# Stage-aware KD:
#
#   Teacher L2  -> Student L2
#   Teacher L4  -> Student L4
#   Teacher L11 -> Student L5
#
# Output KD:
#
#   Teacher output -> Student output
#
# ============================================================


TEACHER_DIM = 128
STUDENT_DIM = 112

TEACHER_LAYERS = 16
STUDENT_LAYERS = 10


# Teacher original layer numbers
DISTILL_TEACHER_LAYERS = [
    2,
    4,
    11
]


# Student corresponding layer numbers
DISTILL_STUDENT_LAYERS = [
    2,
    4,
    5
]


# ------------------------------------------------------------
# Alignment
#
# (student layer, teacher layer)
# ------------------------------------------------------------

ALIGN_PAIRS = [
    (2, 2),
    (4, 4),
    (5, 11)
]


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        required=True
    )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    parser.add_argument(
        "--teacher-config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        required=True
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        metavar="PATH"
    )

    parser.add_argument(
        "--checkpoint-file",
        type=str
    )

    parser.add_argument(
        "--new-checkpoint",
        type=str,
        default="checkpoint"
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    parser.add_argument(
        "-sd",
        "--seed",
        default=42,
        type=int
    )

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    parser.add_argument(
        "--num-cpus",
        default=16,
        type=int
    )

    # --------------------------------------------------------
    # Stage KD weight
    # --------------------------------------------------------

    parser.add_argument(
        "--beta-kd",
        default=0.005,
        type=float,
        help="weight of stage-aware intermediate feature KD"
    )

    # --------------------------------------------------------
    # Output KD weight
    # --------------------------------------------------------

    parser.add_argument(
        "--alpha-kd",
        default=0.005,
        type=float,
        help="weight of output-level knowledge distillation"
    )

    # --------------------------------------------------------
    # WandB
    # --------------------------------------------------------

    parser.add_argument(
        "--use-wandb",
        action="store_true"
    )

    parser.add_argument(
        "--wandb-name",
        default=None,
        type=str
    )

    parser.add_argument(
        "--wandb-run-id",
        default=None,
        type=str
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--eval-only",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# Clean State Dict
# ============================================================

def clean_state_dict(state_dict):

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        if k.startswith("module."):
            k = k[7:]

        new_state_dict[k] = v

    return new_state_dict


# ============================================================
# Load Checkpoint Model
# ============================================================

def load_checkpoint_model(
        model,
        checkpoint_path,
        strict=True
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state_dict = clean_state_dict(
        checkpoint["model"]
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=strict
    )

    return checkpoint, missing, unexpected


# ============================================================
# Build Student
# ============================================================

def build_student(
        args,
        opts,
        device
):

    print()
    print("=" * 60)
    print("[INFO] Building Student model")
    print("=" * 60)

    model_student = load_model(args)

    print(
        "[INFO] Student initialization:"
    )

    print(
        opts.init_checkpoint
    )

    checkpoint = torch.load(
        opts.init_checkpoint,
        map_location="cpu"
    )

    student_state = clean_state_dict(
        checkpoint["model"]
    )

    missing, unexpected = model_student.load_state_dict(
        student_state,
        strict=True
    )

    print(
        "[INFO] Student initialization loaded!"
    )

    print(
        f"[INFO] Student missing keys: {missing}"
    )

    print(
        f"[INFO] Student unexpected keys: {unexpected}"
    )

    if torch.cuda.is_available():

        model_student = torch.nn.DataParallel(
            model_student
        )

    model_student = model_student.to(device)

    return model_student


# ============================================================
# Build Teacher
# ============================================================

def build_teacher(
        opts,
        device
):

    print()
    print("=" * 60)
    print("[INFO] Building Teacher model")
    print("=" * 60)

    teacher_args = get_config(
        opts.teacher_config
    )

    model_teacher = load_model(
        teacher_args
    )

    print(
        "[INFO] Teacher checkpoint:"
    )

    print(
        opts.teacher_checkpoint
    )

    checkpoint = torch.load(
        opts.teacher_checkpoint,
        map_location="cpu"
    )

    teacher_state = clean_state_dict(
        checkpoint["model"]
    )

    missing, unexpected = model_teacher.load_state_dict(
        teacher_state,
        strict=True
    )

    print(
        "[INFO] Teacher checkpoint loaded!"
    )

    print(
        f"[INFO] Teacher missing keys: {missing}"
    )

    print(
        f"[INFO] Teacher unexpected keys: {unexpected}"
    )

    if torch.cuda.is_available():

        model_teacher = torch.nn.DataParallel(
            model_teacher
        )

    model_teacher = model_teacher.to(device)

    model_teacher.eval()

    # Teacher does not participate in optimization
    for p in model_teacher.parameters():

        p.requires_grad = False

    return model_teacher


# ============================================================
# Build Projection Layers
# ============================================================

def build_projection_layers(
        device
):

    print()
    print("=" * 60)
    print("[INFO] Building Stage KD projection layers")
    print("=" * 60)

    print(
        f"[INFO] Student feature dimension: "
        f"{STUDENT_DIM}"
    )

    print(
        f"[INFO] Teacher feature dimension: "
        f"{TEACHER_DIM}"
    )

    print(
        f"[INFO] Alignment pairs: "
        f"{ALIGN_PAIRS}"
    )

    proj_layers = nn.ModuleDict()

    for student_layer, teacher_layer in ALIGN_PAIRS:

        proj_layers[str(student_layer)] = nn.Linear(
            STUDENT_DIM,
            TEACHER_DIM
        )

        print(
            f"[INFO] Projection: "
            f"Student L{student_layer} "
            f"{STUDENT_DIM} -> "
            f"Teacher L{teacher_layer} "
            f"{TEACHER_DIM}"
        )

    proj_layers = proj_layers.to(device)

    return proj_layers


# ============================================================
# Input Augmentation
# ============================================================

def input_augmentation(
        input_2D,
        model,
        joints_left,
        joints_right
):

    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip = model(
        input_2D_flip
    )

    output_3D_flip[..., 0] *= -1

    output_3D_flip[
        ...,
        joints_left + joints_right,
        :
    ] = output_3D_flip[
        ...,
        joints_right + joints_left,
        :
    ]

    output_3D_non_flip = model(
        input_2D_non_flip
    )

    output_3D = (
        output_3D_non_flip
        +
        output_3D_flip
    ) / 2

    return (
        input_2D_non_flip,
        output_3D
    )


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
        args,
        model_student,
        model_teacher,
        proj_layers,
        train_loader,
        optimizer,
        device,
        losses,
        beta_kd,
        alpha_kd
):

    model_student.train()

    model_teacher.eval()

    # --------------------------------------------------------
    # Student:
    #
    #   L2
    #   L4
    #   L5
    #
    # Teacher:
    #
    #   L2
    #   L4
    #   L11
    #
    # --------------------------------------------------------

    student_layers = [
        s for s, t in ALIGN_PAIRS
    ]

    teacher_layers = [
        t for s, t in ALIGN_PAIRS
    ]

    for batch_idx, (x, y) in enumerate(
        tqdm(train_loader)
    ):

        x = x.to(device)

        y = y.to(device)

        batch_size = x.shape[0]

        # ====================================================
        # Teacher Forward
        # ====================================================

        with torch.no_grad():

            pred_teacher, teacher_feats = model_teacher(
                x,
                return_intermediates=teacher_layers
            )

        # ====================================================
        # Student Forward
        # ====================================================

        pred_student, student_feats = model_student(
            x,
            return_intermediates=student_layers
        )

        # ====================================================
        # GT Losses
        # ====================================================

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(
            pred_student,
            y
        )

        loss_3d_scale = n_mpjpe(
            pred_student,
            y
        )

        loss_3d_velocity = loss_velocity(
            pred_student,
            y
        )

        loss_lv = loss_limb_var(
            pred_student
        )

        loss_lg = loss_limb_gt(
            pred_student,
            y
        )

        loss_a = loss_angle(
            pred_student,
            y
        )

        loss_av = loss_angle_velocity(
            pred_student,
            y
        )

        # ====================================================
        # Stage-aware Intermediate KD
        #
        # Teacher:
        #   L2
        #   L4
        #   L11
        #
        # Student:
        #   L2
        #   L4
        #   L5
        #
        # ====================================================

        mid_losses = []

        for student_layer, teacher_layer in ALIGN_PAIRS:

            s_feat = student_feats[
                student_layer
            ]

            t_feat = teacher_feats[
                teacher_layer
            ].detach()

            # ------------------------------------------------
            # Student 112 -> Teacher 128
            # ------------------------------------------------

            s_feat_proj = proj_layers[
                str(student_layer)
            ](
                s_feat
            )

            loss_stage = F.mse_loss(
                s_feat_proj,
                t_feat
            )

            mid_losses.append(
                loss_stage
            )

        loss_kd_mid = (
            sum(mid_losses)
            /
            len(mid_losses)
        )

        # ====================================================
        # Output KD
        #
        # Teacher output -> Student output
        # ====================================================

        pred_teacher_detach = pred_teacher.detach()

        loss_kd_out = F.mse_loss(
            pred_student,
            pred_teacher_detach
        )

        # ====================================================
        # Total Loss
        # ====================================================

        loss_total = (

            # ------------------------------------------------
            # Ground Truth
            # ------------------------------------------------

            loss_3d_pos

            +

            args.lambda_scale
            *
            loss_3d_scale

            +

            args.lambda_3d_velocity
            *
            loss_3d_velocity

            +

            args.lambda_lv
            *
            loss_lv

            +

            args.lambda_lg
            *
            loss_lg

            +

            args.lambda_a
            *
            loss_a

            +

            args.lambda_av
            *
            loss_av

            # ------------------------------------------------
            # Stage-aware KD
            # ------------------------------------------------

            +

            beta_kd
            *
            loss_kd_mid

            # ------------------------------------------------
            # Output KD
            # ------------------------------------------------

            +

            alpha_kd
            *
            loss_kd_out
        )

        # ====================================================
        # Logging
        # ====================================================

        losses["gt"].update(
            loss_3d_pos.item(),
            batch_size
        )

        losses["kd_mid"].update(
            loss_kd_mid.item(),
            batch_size
        )

        losses["kd_out"].update(
            loss_kd_out.item(),
            batch_size
        )

        losses["total"].update(
            loss_total.item(),
            batch_size
        )

        # ====================================================
        # First Batch Debug
        # ====================================================

        if batch_idx == 0:

            print()

            print(
                "[DEBUG] First batch:"
            )

            print(
                f"        GT MPJPE      = "
                f"{loss_3d_pos.item():.6f}"
            )

            print(
                f"        Stage KD      = "
                f"{loss_kd_mid.item():.6f}"
            )

            print(
                f"        beta          = "
                f"{beta_kd:.6f}"
            )

            print(
                f"        beta * Stage  = "
                f"{beta_kd * loss_kd_mid.item():.6f}"
            )

            print(
                f"        Output KD     = "
                f"{loss_kd_out.item():.6f}"
            )

            print(
                f"        alpha         = "
                f"{alpha_kd:.6f}"
            )

            print(
                f"        alpha * Out   = "
                f"{alpha_kd * loss_kd_out.item():.6f}"
            )

            print(
                f"        Total         = "
                f"{loss_total.item():.6f}"
            )

        # ====================================================
        # Backward
        # ====================================================

        loss_total.backward()

        optimizer.step()


# ============================================================
# Evaluation
# ============================================================

def evaluate(
        model,
        test_loader,
        n_frames
):

    model.eval()

    eval_joints = [
        0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13
    ]

    joints_left = [
        5, 6, 7, 11, 12, 13
    ]

    joints_right = [
        2, 3, 4, 8, 9, 10
    ]

    data_inference = {}

    error_sum_test = AccumLoss()

    pck_list = []

    auc_list = []

    with torch.no_grad():

        for data in tqdm(
            test_loader,
            0
        ):

            (
                batch_cam,
                gt_3D,
                input_2D,
                seq,
                scale,
                bb_box
            ) = data

            [
                input_2D,
                gt_3D,
                batch_cam,
                scale,
                bb_box
            ] = get_variable(
                "test",
                [
                    input_2D,
                    gt_3D,
                    batch_cam,
                    scale,
                    bb_box
                ]
            )

            N = input_2D.size(0)

            # ------------------------------------------------
            # Ground Truth
            # ------------------------------------------------

            out_target = gt_3D.clone().view(
                N,
                -1,
                17,
                3
            )

            out_target[:, :, 14] = 0

            # ------------------------------------------------
            # Test Augmentation
            # ------------------------------------------------

            input_2D, output_3D = input_augmentation(
                input_2D,
                model,
                joints_left,
                joints_right
            )

            # ------------------------------------------------
            # Scale Recovery
            # ------------------------------------------------

            output_3D = (
                output_3D
                *
                scale.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(
                    1,
                    output_3D.size(1),
                    17,
                    3
                )
            )

            # ------------------------------------------------
            # Center Frame
            # ------------------------------------------------

            pad = (
                n_frames - 1
            ) // 2

            pred_out = (
                output_3D[:, pad]
                .unsqueeze(1)
            )

            # ------------------------------------------------
            # Root
            # ------------------------------------------------

            pred_out[..., 14, :] = 0

            pred_out = denormalize(
                pred_out,
                seq
            )

            # ------------------------------------------------
            # Root Relative
            # ------------------------------------------------

            pred_out_rel = (
                pred_out
                -
                pred_out[..., 14:15, :]
            )

            out_target_rel = (
                out_target
                -
                out_target[..., 14:15, :]
            )

            # ------------------------------------------------
            # MPJPE
            # ------------------------------------------------

            joint_error_test = mpjpe_cal(
                pred_out_rel,
                out_target_rel
            ).item()

            error_sum_test.update(
                joint_error_test * N,
                N
            )

            # ------------------------------------------------
            # PCK
            # ------------------------------------------------

            dists = torch.norm(
                pred_out_rel
                -
                out_target_rel,
                dim=-1
            )

            dists_eval = dists[
                :,
                :,
                eval_joints
            ]

            pck = (
                dists_eval < 150.0
            ).float().mean() * 100

            pck_list.append(
                pck.item()
            )

            # ------------------------------------------------
            # AUC
            # ------------------------------------------------

            thresholds = torch.linspace(
                0,
                150,
                31
            ).to(dists.device)

            auc_pcks = []

            for t in thresholds:

                auc_pcks.append(
                    (
                        dists_eval < t
                    ).float()
                    .mean()
                    .item()
                )

            auc = (
                np.mean(auc_pcks)
                *
                100
            )

            auc_list.append(
                auc
            )

            # ------------------------------------------------
            # Save Inference
            # ------------------------------------------------

            inference_out = (
                pred_out_rel
                +
                out_target[
                    ...,
                    14:15,
                    :
                ]
            )

            for seq_cnt in range(
                len(seq)
            ):

                seq_name = seq[
                    seq_cnt
                ]

                current = (
                    inference_out[
                        seq_cnt
                    ]
                    .permute(
                        2,
                        1,
                        0
                    )
                    .cpu()
                    .numpy()
                )

                if seq_name in data_inference:

                    data_inference[
                        seq_name
                    ] = np.concatenate(
                        (
                            data_inference[
                                seq_name
                            ],
                            current
                        ),
                        axis=2
                    )

                else:

                    data_inference[
                        seq_name
                    ] = current

    # --------------------------------------------------------
    # Format MAT
    # --------------------------------------------------------

    for seq_name in data_inference.keys():

        data_inference[
            seq_name
        ] = data_inference[
            seq_name
        ][:, :, None, :]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print("=" * 40)

    print(
        "Evaluation Results "
        "(MPI-INF-3DHP):"
    )

    print(
        f"Protocol #1 (MPJPE): "
        f"{error_sum_test.avg:.2f} mm"
    )

    print(
        f"PCK (Threshold 150mm): "
        f"{np.mean(pck_list):.2f}%"
    )

    print(
        f"AUC (0-150mm Range): "
        f"{np.mean(auc_list):.2f}"
    )

    print("=" * 40)

    return (
        error_sum_test.avg,
        data_inference,
        np.mean(pck_list),
        np.mean(auc_list)
    )


# ============================================================
# Save Checkpoint
# ============================================================

def save_checkpoint(
        checkpoint_path,
        epoch,
        lr,
        optimizer,
        model,
        proj_layers,
        min_mpjpe,
        wandb_id
):

    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "proj_layers": proj_layers.state_dict(),
            "min_mpjpe": min_mpjpe,
            "wandb_id": wandb_id
        },
        checkpoint_path
    )


# ============================================================
# Save Inference
# ============================================================

def save_data_inference(
        path,
        data_inference,
        latest
):

    if latest:

        mat_path = os.path.join(
            path,
            "inference_data.mat"
        )

    else:

        mat_path = os.path.join(
            path,
            "inference_data_best.mat"
        )

    scio.savemat(
        mat_path,
        data_inference
    )


# ============================================================
# Main Training Function
# ============================================================

def train(
        args,
        opts
):

    print_args(args)

    create_directory_if_not_exists(
        opts.new_checkpoint
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Experiment Information
    # ========================================================

    print()

    print("=" * 60)

    print(
        "MPI-INF-3DHP "
        "Stage-aware Knowledge Distillation"
    )

    print("=" * 60)

    print(
        f"[INFO] Student config: "
        f"{opts.config}"
    )

    print(
        f"[INFO] Student checkpoint: "
        f"{opts.init_checkpoint}"
    )

    print(
        f"[INFO] Teacher config: "
        f"{opts.teacher_config}"
    )

    print(
        f"[INFO] Teacher checkpoint: "
        f"{opts.teacher_checkpoint}"
    )

    print(
        f"[INFO] beta_kd: "
        f"{opts.beta_kd}"
    )

    print(
        f"[INFO] alpha_kd: "
        f"{opts.alpha_kd}"
    )

    print(
        f"[INFO] Teacher layers: "
        f"{DISTILL_TEACHER_LAYERS}"
    )

    print(
        f"[INFO] Student layers: "
        f"{DISTILL_STUDENT_LAYERS}"
    )

    print(
        f"[INFO] Alignment: "
        f"{ALIGN_PAIRS}"
    )

    print(
        "[INFO] Output KD: "
        "Teacher output -> Student output"
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = MPI3DHP(
        args,
        train=True
    )

    test_dataset = Fusion(
        args,
        train=False
    )

    common_loader_params = {

        "num_workers":
            opts.num_cpus - 1,

        "pin_memory":
            True,

        "prefetch_factor":
            (opts.num_cpus - 1) // 3,

        "persistent_workers":
            True
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        **common_loader_params
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        **common_loader_params
    )

    # ========================================================
    # Build Student
    # ========================================================

    model_student = build_student(
        args,
        opts,
        device
    )

    # ========================================================
    # Build Teacher
    # ========================================================

    model_teacher = build_teacher(
        opts,
        device
    )

    # ========================================================
    # Projection Layers
    # ========================================================

    proj_layers = build_projection_layers(
        device
    )

    # ========================================================
    # Parameter Count
    # ========================================================

    student_params = count_param_numbers(
        model_student
    )

    teacher_params = count_param_numbers(
        model_teacher
    )

    projection_params = sum(
        p.numel()
        for p in proj_layers.parameters()
    )

    print()

    print(
        f"[INFO] Student parameters: "
        f"{student_params:,}"
    )

    print(
        f"[INFO] Teacher parameters: "
        f"{teacher_params:,}"
    )

    print(
        f"[INFO] Projection parameters: "
        f"{projection_params:,}"
    )

    # ========================================================
    # Optimizer
    # ========================================================

    all_params = (
        list(
            filter(
                lambda p: p.requires_grad,
                model_student.parameters()
            )
        )
        +
        list(
            proj_layers.parameters()
        )
    )

    lr = args.learning_rate

    optimizer = optim.AdamW(
        all_params,
        lr=lr,
        weight_decay=args.weight_decay
    )

    lr_decay = args.lr_decay

    epoch_start = 0

    min_mpjpe = float("inf")

    # ========================================================
    # WandB
    # ========================================================

    wandb_id = (
        opts.wandb_run_id
        if opts.wandb_run_id is not None
        else wandb.util.generate_id()
    )

    if opts.use_wandb:

        print(
            f"[INFO] WandB Run ID: "
            f"{wandb_id}"
        )

        wandb.init(
            id=wandb_id,
            name=opts.wandb_name,
            project="MotionMetaFormer",
            resume="allow",
            settings=wandb.Settings(
                start_method="fork"
            )
        )

        wandb.config.update({

            "kd_type":
                "stage_aware_plus_output",

            "beta_kd":
                opts.beta_kd,

            "alpha_kd":
                opts.alpha_kd,

            "teacher_layers":
                DISTILL_TEACHER_LAYERS,

            "student_layers":
                DISTILL_STUDENT_LAYERS,

            "align_pairs":
                ALIGN_PAIRS,

            "student_dim":
                STUDENT_DIM,

            "teacher_dim":
                TEACHER_DIM,

            "student_layers_total":
                STUDENT_LAYERS,

            "teacher_layers_total":
                TEACHER_LAYERS
        })

    # ========================================================
    # Checkpoint Paths
    # ========================================================

    ckpt_latest = os.path.join(
        opts.new_checkpoint,
        "latest_epoch.pth.tr"
    )

    ckpt_best = os.path.join(
        opts.new_checkpoint,
        "best_epoch.pth.tr"
    )

    # ========================================================
    # Resume
    # ========================================================

    if opts.resume:

        checkpoint_path = os.path.join(
            opts.checkpoint,
            opts.checkpoint_file
        )

        print(
            f"[INFO] Resume checkpoint: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        model_student.load_state_dict(
            clean_state_dict(
                checkpoint["model"]
            ),
            strict=True
        )

        proj_layers.load_state_dict(
            checkpoint["proj_layers"],
            strict=True
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        epoch_start = checkpoint[
            "epoch"
        ]

        lr = checkpoint[
            "lr"
        ]

        min_mpjpe = checkpoint[
            "min_mpjpe"
        ]

    # ========================================================
    # Eval Only
    # ========================================================

    if opts.eval_only:

        print()

        print(
            "=" * 60
        )

        print(
            "[INFO] Evaluation only"
        )

        print(
            "=" * 60
        )

        (
            mpjpe,
            data_inference,
            pck,
            auc
        ) = evaluate(
            model_student,
            test_loader,
            args.n_frames
        )

        print(
            f"[INFO] MPJPE: {mpjpe:.2f} mm"
        )

        print(
            f"[INFO] PCK: {pck:.2f}%"
        )

        print(
            f"[INFO] AUC: {auc:.2f}"
        )

        return

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        epoch_start,
        args.epochs
    ):

        print()

        print(
            f"[INFO] Epoch "
            f"{epoch + 1}/{args.epochs}"
        )

        losses = {

            "gt":
                AverageMeter(),

            "kd_mid":
                AverageMeter(),

            "kd_out":
                AverageMeter(),

            "total":
                AverageMeter()
        }

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_one_epoch(
            args,
            model_student,
            model_teacher,
            proj_layers,
            train_loader,
            optimizer,
            device,
            losses,
            opts.beta_kd,
            opts.alpha_kd
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        with torch.no_grad():

            (
                mpjpe,
                data_inference,
                pck,
                auc
            ) = evaluate(
                model_student,
                test_loader,
                args.n_frames
            )

        # ----------------------------------------------------
        # Best Model
        # ----------------------------------------------------

        if mpjpe < min_mpjpe:

            min_mpjpe = mpjpe

            save_checkpoint(
                ckpt_best,
                epoch,
                lr,
                optimizer,
                model_student,
                proj_layers,
                min_mpjpe,
                wandb_id
            )

            save_data_inference(
                opts.new_checkpoint,
                data_inference,
                latest=False
            )

            print()

            print(
                f"[INFO] New best MPJPE: "
                f"{min_mpjpe:.2f} mm"
            )

        # ----------------------------------------------------
        # Latest Model
        # ----------------------------------------------------

        save_checkpoint(
            ckpt_latest,
            epoch,
            lr,
            optimizer,
            model_student,
            proj_layers,
            min_mpjpe,
            wandb_id
        )

        save_data_inference(
            opts.new_checkpoint,
            data_inference,
            latest=True
        )

        # ----------------------------------------------------
        # WandB
        # ----------------------------------------------------

        if opts.use_wandb:

            wandb.log({

                "lr":
                    lr,

                "train/loss_gt":
                    losses["gt"].avg,

                "train/loss_kd_mid":
                    losses["kd_mid"].avg,

                "train/loss_kd_out":
                    losses["kd_out"].avg,

                "train/loss_total":
                    losses["total"].avg,

                "eval/mpjpe":
                    mpjpe,

                "eval/pck":
                    pck,

                "eval/auc":
                    auc,

                "eval/min_mpjpe":
                    min_mpjpe
            }, step=epoch + 1)

        # ----------------------------------------------------
        # LR Decay
        # ----------------------------------------------------

        lr = decay_lr_exponentially(
            lr,
            lr_decay,
            optimizer
        )


# ============================================================
# Main
# ============================================================

def main():

    opts = parse_args()

    set_random_seed(
        opts.seed
    )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    args = get_config(
        opts.config
    )

    train(
        args,
        opts
    )


if __name__ == "__main__":

    main()