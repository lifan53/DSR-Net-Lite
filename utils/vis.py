# utils/vis.py

import os
import matplotlib.pyplot as plt
import numpy as np
from data.const import H36M_JOINT_TO_LABEL 
from utils.tools import create_directory_if_not_exists 


def plot_evaluation_curves(checkpoint_dir, history_mpjpe, history_p_mpjpe):
    
    epochs = range(1, len(history_mpjpe) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_mpjpe, label='MPJPE (Protocol #1)', marker='o', linestyle='-')
    plt.plot(epochs, history_p_mpjpe, label='P-MPJPE (Protocol #2)', marker='x', linestyle='--')
    
    plt.title('Evaluation Error vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error (mm)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    create_directory_if_not_exists(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, 'evaluation_error_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved Evaluation Curve to {save_path}")
    


def plot_joint_error_bar(checkpoint_dir, joints_error, num_joints, current_epoch):
    
    joint_labels = [H36M_JOINT_TO_LABEL[i] for i in range(num_joints)]
    
    plt.figure(figsize=(14, 7))
    plt.bar(joint_labels, joints_error, color='teal')
    
    plt.title(f'Joint Error (JPE) at Epoch {current_epoch}')
    plt.xlabel('Joint Label')
    plt.ylabel('Error (mm)')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    create_directory_if_not_exists(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, f'joint_error_epoch_{current_epoch}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved Joint Error Bar to {save_path}")
    