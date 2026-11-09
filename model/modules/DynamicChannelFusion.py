import torch
from torch import nn
import torch.nn.functional as F

class DynamicChannelFusion(nn.Module):
   
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.C = channels
        
        reduced_c = self.C * 2 // reduction_ratio 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.C * 2, reduced_c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_c, self.C * 2, bias=False)
        )

    def forward(self, X_G, X_M):
        
        X_G_sq = X_G.mean(dim=(1, 2))
        X_M_sq = X_M.mean(dim=(1, 2))

        X_C = torch.cat([X_G_sq, X_M_sq], dim=-1)
        
        W_raw = self.mlp(X_C) 
        
        W_alpha = W_raw[:, :self.C].sigmoid() 
        W_beta = W_raw[:, self.C:].sigmoid()  
       
        W_alpha = W_alpha.unsqueeze(1).unsqueeze(1) 
        W_beta = W_beta.unsqueeze(1).unsqueeze(1)   
        Output = W_alpha * X_G + W_beta * X_M
        
        return Output