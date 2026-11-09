import torch
import argparse
from collections import OrderedDict
from model.MotionAGFormer_DCF3_CKA import MotionAGFormer_DCF3_Pruned
from utils.tools import get_config


def build_student(args_model, teacher_ckpt, skip_layers, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. 加载teacher权重 ──────────────────────────────
    ckpt = torch.load(teacher_ckpt, map_location=device)
    teacher_state = OrderedDict()
    for k, v in ckpt['model'].items():
        name = k[7:] if k.startswith('module.') else k
        teacher_state[name] = v

    # ── 2. 确定保留层并重新编号 ─────────────────────────
    teacher_n_layers = args_model.n_layers  # yaml里的n_layers，即teacher层数
    keep_indices = [i for i in range(teacher_n_layers) if i not in skip_layers]
    student_n_layers = len(keep_indices)
    print(f"Teacher层数: {teacher_n_layers}")
    print(f"跳过层: {sorted(skip_layers)}")
    print(f"保留层: {keep_indices}")
    print(f"Student层数: {student_n_layers}")

    # ── 3. 搬运权重 ─────────────────────────────────────
    student_state = OrderedDict()

    # 非层参数直接复制
    for k, v in teacher_state.items():
        if not k.startswith('layers.'):
            student_state[k] = v

    # 层参数重新编号
    for new_idx, old_idx in enumerate(keep_indices):
        for k, v in teacher_state.items():
            if k.startswith(f'layers.{old_idx}.'):
                new_key = k.replace(f'layers.{old_idx}.', f'layers.{new_idx}.', 1)
                student_state[new_key] = v

    # ── 4. 建student并加载 ──────────────────────────────
    student = MotionAGFormer_DCF3_Pruned(
        n_layers=student_n_layers,       # 由skip_layers自动推算
        dim_in=args_model.dim_in,        # 以下全从配置文件读
        dim_feat=args_model.dim_feat,
        dim_rep=args_model.dim_rep,
        mlp_ratio=args_model.mlp_ratio,
        num_heads=args_model.num_heads,
        n_frames=args_model.n_frames,
        use_layer_scale=args_model.use_layer_scale,
        use_adaptive_fusion=args_model.use_adaptive_fusion,
        use_temporal_similarity=args_model.use_temporal_similarity,
        use_dynamic_spatial_similarity=args_model.use_dynamic_spatial_similarity,
        use_mlp_temporal_similarity=args_model.use_mlp_temporal_similarity,
    )

    missing, unexpected = student.load_state_dict(student_state, strict=True)
    print(f"Missing keys:    {missing}")     # 期望 []
    print(f"Unexpected keys: {unexpected}")  # 期望 []

    total = sum(p.numel() for p in student.parameters())
    print(f"Student参数量: {total:,}")

    # ── 5. 保存 ─────────────────────────────────────────
    torch.save({'model': student.state_dict()}, save_path)
    print(f"已保存至: {save_path}")

    return student


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/h36m/MotionAGFormer_DCF3.yaml')
    parser.add_argument(
    '--teacher_ckpt',
    type=str,
    required=True,
    help='teacher checkpoint path'
        )
    parser.add_argument('--skip_layers', type=int, nargs='+', default=[6, 9],
                        help='要删除的teacher层编号，例如 --skip_layers 6 9')
    parser.add_argument('--save_path', type=str,
                        default='checkpoint/student_noncontiguous/init.pth.tr')
    cli = parser.parse_args()

    args_model = get_config(cli.config)
    build_student(
    args_model,
    cli.teacher_ckpt,
    set(cli.skip_layers),
    cli.save_path
    )
    # 验证
    import os
    ckpt = torch.load(cli.save_path, map_location='cpu')
    print("验证保存的checkpoint keys示例:")
    for k in list(ckpt['model'].keys())[:5]:
        print(f"  {k}")