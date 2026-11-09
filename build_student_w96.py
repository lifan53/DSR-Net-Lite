import os
import torch
import argparse
from collections import OrderedDict

from model.MotionAGFormer_DCF3_CKA import MotionAGFormer_DCF3_Pruned
from utils.tools import get_config


def copy_with_crop(src_tensor, dst_tensor):
    """
    将teacher权重裁剪后复制到student
    支持:
    128 -> 96
    """

    if src_tensor.shape == dst_tensor.shape:
        dst_tensor.copy_(src_tensor)
        return

    slices = tuple(
        slice(0, min(s, d))
        for s, d in zip(src_tensor.shape, dst_tensor.shape)
    )

    dst_tensor[slices] = src_tensor[slices]


def build_student_w96(args_model, teacher_ckpt, save_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Building Student-2 (10 Layers, Width=96)")
    print("=" * 60)

    # --------------------------------------------------
    # Load Student-1 checkpoint
    # --------------------------------------------------
    ckpt = torch.load(teacher_ckpt, map_location=device)

    teacher_state = OrderedDict()

    for k, v in ckpt["model"].items():
        name = k[7:] if k.startswith("module.") else k
        teacher_state[name] = v

    print(f"Loaded checkpoint: {teacher_ckpt}")

    # --------------------------------------------------
    # Build Student-2
    # --------------------------------------------------
    student = MotionAGFormer_DCF3_Pruned(
        n_layers=10,
        dim_in=args_model.dim_in,
        dim_feat=96,
        dim_rep=args_model.dim_rep,
        dim_out=args_model.dim_out,
        mlp_ratio=args_model.mlp_ratio,
        num_heads=args_model.num_heads,
        n_frames=args_model.n_frames,
        use_layer_scale=args_model.use_layer_scale,
        use_adaptive_fusion=args_model.use_adaptive_fusion,
        use_temporal_similarity=args_model.use_temporal_similarity,
        use_dynamic_spatial_similarity=args_model.use_dynamic_spatial_similarity,
        use_mlp_temporal_similarity=args_model.use_mlp_temporal_similarity,
    )

    student_state = student.state_dict()

    loaded_count = 0

    # --------------------------------------------------
    # Weight Transfer
    # --------------------------------------------------
    for k in student_state.keys():

        if k not in teacher_state:
            continue

        try:
            copy_with_crop(
                teacher_state[k],
                student_state[k]
            )

            loaded_count += 1

        except Exception as e:
            print(f"Skip: {k}")
            print(e)

    missing, unexpected = student.load_state_dict(
        student_state,
        strict=False
    )

    print()
    print(f"Loaded tensors: {loaded_count}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    total_params = sum(
        p.numel()
        for p in student.parameters()
    )

    print(f"Student Params: {total_params:,}")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    save_dir = os.path.dirname(save_path)

    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model": student.state_dict()
        },
        save_path
    )

    print()
    print(f"Saved to: {save_path}")

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------
    student = student.to(device)

    x = torch.randn(
        1,
        args_model.n_frames,
        args_model.num_joints,
        3
    ).to(device)

    student.eval()

    with torch.no_grad():
        out = student(x)

    print()
    print(f"Output shape: {out.shape}")
    print("Build Success!")

    return student


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/h36m/MotionAGFormer_DCF3_CKA_student.yaml"
    )

    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        required=True,
        help="Student-1 best checkpoint"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoint/student_L6L9_w96/init.pth.tr"
    )

    cli = parser.parse_args()

    args_model = get_config(cli.config)

    build_student_w96(
        args_model,
        cli.teacher_ckpt,
        cli.save_path
    )