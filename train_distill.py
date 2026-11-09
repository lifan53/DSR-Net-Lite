import argparse
import os
import numpy as np
import torch
import wandb
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader
from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.vis import plot_evaluation_curves

STUDENT_DIM = 96
TEACHER_DIM = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher-config", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH')
    parser.add_argument('--new-checkpoint', type=str, default='checkpoint')
    parser.add_argument('--checkpoint-file', type=str)
    parser.add_argument('-sd', '--seed', default=42, type=int)
    parser.add_argument('--num-cpus', default=16, type=int)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--alpha-kd', default=0.0, type=float,
                        help='输出蒸馏loss权重，0=不用输出蒸馏')
    parser.add_argument('--beta-kd', default=0.0, type=float,
                        help='中间层蒸馏loss权重，0=不用中间层蒸馏')
    parser.add_argument('--distill-layers', type=int, nargs='+',
                        default=[3, 5, 7],
                        help='student中间层蒸馏位置（同时也是teacher对齐层），默认[3,5,7]')
    return parser.parse_args()


def train_one_epoch(args, model_student, model_teacher, proj_layers,
                    align_pairs, train_loader, optimizer, device,
                    losses, alpha_kd, beta_kd):
    model_student.train()
    model_teacher.eval()

    use_out = alpha_kd > 0
    use_mid = beta_kd > 0
    mid_layers = [s for s, t in align_pairs] if use_mid else None

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]

            # teacher推理
            if use_mid:
                pred_teacher, t_feats = model_teacher(
                    x, return_intermediates=mid_layers)
            else:
                pred_teacher = model_teacher(x)

        # student推理
        if use_mid:
            pred_student, s_feats = model_student(
                x, return_intermediates=mid_layers)
        else:
            pred_student = model_student(x)

        optimizer.zero_grad()

        # GT loss
        loss_gt = loss_mpjpe(pred_student, y)
        loss_3d_scale = n_mpjpe(pred_student, y)
        loss_3d_velocity = loss_velocity(pred_student, y)
        loss_lv = loss_limb_var(pred_student)
        loss_lg = loss_limb_gt(pred_student, y)
        loss_a = loss_angle(pred_student, y)
        loss_av = loss_angle_velocity(pred_student, y)

        # 输出蒸馏loss
        loss_kd_out = torch.tensor(0.0, device=device)
        if use_out:
            loss_kd_out = F.mse_loss(pred_student, pred_teacher.detach())

        # 中间层蒸馏loss
        loss_kd_mid = torch.tensor(0.0, device=device)
        if use_mid:
            mid_losses = []
            for s_idx, t_idx in align_pairs:
                s_feat = s_feats[s_idx]           # [B, T, J, 96]
                t_feat = t_feats[t_idx].detach()  # [B, T, J, 128]
                s_feat_proj = proj_layers[str(s_idx)](s_feat)  # 96->128
                mid_losses.append(F.mse_loss(s_feat_proj, t_feat))
            loss_kd_mid = sum(mid_losses) / len(mid_losses)

        # 第一个batch打印loss数量级
        if batch_idx == 0:
            print(f"[DEBUG] loss_gt={loss_gt.item():.6f}  "
                  f"alpha*kd_out={alpha_kd * loss_kd_out.item():.6f}  "
                  f"beta*kd_mid={beta_kd * loss_kd_mid.item():.6f}")

        loss_total = (loss_gt
                      + args.lambda_scale * loss_3d_scale
                      + args.lambda_3d_velocity * loss_3d_velocity
                      + args.lambda_lv * loss_lv
                      + args.lambda_lg * loss_lg
                      + args.lambda_a * loss_a
                      + args.lambda_av * loss_av
                      + alpha_kd * loss_kd_out
                      + beta_kd * loss_kd_mid)

        losses['gt'].update(loss_gt.item(), x.shape[0])
        losses['kd_out'].update(loss_kd_out.item(), x.shape[0])
        losses['kd_mid'].update(loss_kd_mid.item(), x.shape[0])
        losses['total'].update(loss_total.item(), x.shape[0])

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device):
    print("[INFO] Evaluation")
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            if args.flip:
                batch_input_flip = flip_data(x)
                pred1 = model(x)
                pred_flip = model(batch_input_flip)
                pred2 = flip_data(pred_flip)
                predicted_3d_pos = (pred1 + pred2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0
            else:
                y[:, 0, 0, 2] = 0
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])
    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]

    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(actions))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']

    for idx in range(len(action_clips)):
        if source_clips[idx][0][:-6] in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred = (pred * factor) - (pred * factor)[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        err2 = calculate_p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    for i in range(num_test_frames):
        if e1_all[i] > 0:
            results[actions[i]].append(e1_all[i] / oc[i])
            results_procrustes[actions[i]].append(e2_all[i] / oc[i])

    final_result = [np.mean(results[a]) for a in action_names]
    final_result_procrustes = [np.mean(results_procrustes[a])
                               for a in action_names]
    e1 = np.mean(final_result)
    e2 = np.mean(final_result_procrustes)
    print(f'Protocol #1 Error (MPJPE): {e1} mm')
    print(f'Protocol #2 Error (P-MPJPE): {e2} mm')
    return e1, e2


def save_checkpoint(path, epoch, lr, optimizer, model,
                    min_mpjpe, wandb_id, mpjpe, p_mpjpe):
    torch.save({
        'epoch': epoch + 1, 'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
        'mpjpe': mpjpe, 'p_mpjpe': p_mpjpe,
    }, path)


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构建对齐关系（student层编号 = teacher层编号，因为删的是尾部）
    align_pairs = [(idx, idx) for idx in opts.distill_layers]
    print(f"[INFO] 蒸馏策略: alpha_kd={opts.alpha_kd}, "
          f"beta_kd={opts.beta_kd}, "
          f"distill_layers={opts.distill_layers}")
    print(f"[INFO] 对齐层对: {align_pairs}")

    datareader = DataReaderH36M(
        n_frames=args.n_frames, sample_stride=1,
        data_stride_train=args.n_frames // 3,
        data_stride_test=args.n_frames,
        dt_root='data/motion3d', dt_file=args.dt_file)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True,
                              **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             **common_loader_params)

    # 加载student
    model_student = load_model(args)
    if opts.checkpoint and opts.checkpoint_file:
        ckpt_path = os.path.join(opts.checkpoint, opts.checkpoint_file)
        ckpt = torch.load(ckpt_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model_student.load_state_dict(new_state_dict, strict=True)
        print(f"[INFO] 学生模型权重加载成功：{ckpt_path}")
    else:
        print("[INFO] 学生模型随机初始化")
    if torch.cuda.is_available():
        model_student = torch.nn.DataParallel(model_student)
    model_student.to(device)

    # 加载teacher
    teacher_args = get_config(opts.teacher_config)
    model_teacher = load_model(teacher_args)
    teacher_ckpt = torch.load(opts.teacher_checkpoint, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in teacher_ckpt['model'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model_teacher.load_state_dict(new_state_dict, strict=False)
    if torch.cuda.is_available():
        model_teacher = torch.nn.DataParallel(model_teacher)
    model_teacher.to(device)
    model_teacher.eval()
    print("[INFO] 教师模型加载成功")

    # 中间层投影器（student 96维 -> teacher 128维）
    # 只有beta_kd>0时才有意义，但始终建立以保持代码结构一致
    proj_layers = nn.ModuleDict({
        str(s_idx): nn.Linear(STUDENT_DIM, TEACHER_DIM)
        for s_idx, _ in align_pairs
    }).to(device)
    proj_param_count = sum(p.numel() for p in proj_layers.parameters())
    print(f"[INFO] 投影层参数量: {proj_param_count:,}  "
          f"（仅beta_kd>0时参与训练）")

    n_params = count_param_numbers(model_student)
    print(f"[INFO] 学生模型参数量: {n_params:,}")

    # 优化器：student参数 + 投影层参数
    all_params = (list(filter(lambda p: p.requires_grad,
                              model_student.parameters()))
                  + list(proj_layers.parameters()))
    lr = args.learning_rate
    optimizer = optim.AdamW(all_params, lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')
    wandb_id = wandb.util.generate_id()
    history_mpjpe, history_p_mpjpe = [], []

    if opts.use_wandb:
        wandb.init(id=wandb_id, name=opts.wandb_name,
                   project='MotionMetaFormer', resume="allow",
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update({
            'alpha_kd': opts.alpha_kd,
            'beta_kd': opts.beta_kd,
            'distill_layers': opts.distill_layers,
        })

    ckpt_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    ckpt_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        print(f"[INFO] epoch {epoch}")
        losses = {name: AverageMeter()
                  for name in ['gt', 'kd_out', 'kd_mid', 'total']}

        train_one_epoch(args, model_student, model_teacher, proj_layers,
                        align_pairs, train_loader, optimizer, device,
                        losses, opts.alpha_kd, opts.beta_kd)

        mpjpe, p_mpjpe = evaluate(args, model_student,
                                   test_loader, datareader, device)
        history_mpjpe.append(mpjpe)
        history_p_mpjpe.append(p_mpjpe)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(ckpt_best, epoch, lr, optimizer,
                            model_student, min_mpjpe,
                            wandb_id, mpjpe, p_mpjpe)
        save_checkpoint(ckpt_latest, epoch, lr, optimizer,
                        model_student, min_mpjpe,
                        wandb_id, mpjpe, p_mpjpe)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            plot_evaluation_curves(opts.new_checkpoint,
                                   history_mpjpe, history_p_mpjpe)

        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_gt': losses['gt'].avg,
                'train/loss_kd_out': losses['kd_out'].avg,
                'train/loss_kd_mid': losses['kd_mid'].avg,
                'train/loss_total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval/min_mpjpe': min_mpjpe,
            }, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train(args, opts)


if __name__ == '__main__':
    main()