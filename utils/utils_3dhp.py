import os
from torch.autograd import Variable
import torch
import numpy as np

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape, f"Predicted shape is {predicted.shape} while target is {target.shape}"
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_p1, wandb_id, last=True):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    file_name = 'last.pth.tr' if last else 'best.pth.tr'
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_p1': min_p1,
        'wandb_id': wandb_id,
    }, os.path.join(checkpoint_path, file_name))
def compute_pck_auc(pred, gt):
    """
    计算 PCK 和 AUC
    pred, gt: numpy 数组, 形状为 (N, 14, 3) 或 (N, 17, 3), 单位 mm
    """
    # 计算每个关节点的欧氏距离 (N, J)
    distances = np.sqrt(np.sum((pred - gt)**2, axis=-1))
    
    # PCK: 阈值 150mm
    pck = np.mean(distances < 150.0) * 100
    
    # AUC: 在 0-150mm 之间取多个阈值计算 PCK 的平均值
    thresholds = np.linspace(0, 150, 151) # 0, 1, 2...150
    pck_curve = []
    for t in thresholds:
        pck_curve.append(np.mean(distances < t))
    
    auc = np.mean(pck_curve) * 100
    
    return pck, auc