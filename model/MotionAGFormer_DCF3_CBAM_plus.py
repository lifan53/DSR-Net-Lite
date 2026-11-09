# DSR-Net模型
from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath

from model.modules.attention import Attention
from model.modules.dyngraph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN
from model.modules.DynamicChannelFusion import DynamicChannelFusion

from model.modules.cbam_attention_plus import CBAM_Plus

class MixFormerBlock_DCF3(nn.Module):
    """
    Implementation of AGFormer block with improvements.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', use_temporal_similarity=True, temporal_connection_len=1, neighbour_num=4, n_frames=243,use_dynamic_spatial_similarity=False, use_mlp_temporal_similarity=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.norm1 = nn.LayerNorm(dim)
        self.mixer1 = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        self.mixer2 = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len,
                         use_dynamic_spatial_similarity=use_dynamic_spatial_similarity, 
                            use_mlp_temporal_similarity=use_mlp_temporal_similarity)
        self.mixer3 = MultiScaleTCN(in_channels=dim, out_channels=dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        
        self.lambda_scalar = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32), 
            requires_grad=True
        )
        
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        
    

    def forward(self, x):
        """
        Forward pass of the MixFormer block.
        
        x: Tensor with shape [B, T, J, C]
        """
        
        
        x1 = x
        x2 = x

        if self.use_layer_scale:
            x1 = self.norm1(x)
            x2 = self.norm1(x)
            
            x1 = x1 + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer1(x1))
            x2 = x2 + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer2(x2))
          
        
            lambda_val = torch.sigmoid(self.lambda_scalar)
            x = lambda_val * x1 + (1 - lambda_val) * x2

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
          
            x = x + self.drop_path(self.mixer1(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x




class MotionAGFormerBlock_DCF3(nn.Module):
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,use_dynamic_spatial_similarity=False, use_mlp_temporal_similarity=False):
        super().__init__()
        self.hierarchical = hierarchical
        dim_branch = dim // 2 if hierarchical else dim
        
        self.cbam = CBAM_Plus(dim_branch)

        # ST Attention branch
        self.att_spatial = MixFormerBlock_DCF3(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                             qk_scale, use_layer_scale, layer_scale_init_value,
                                             mode='spatial',
                                             use_temporal_similarity=use_temporal_similarity,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames,
                                            use_dynamic_spatial_similarity=use_dynamic_spatial_similarity,
                                             use_mlp_temporal_similarity=use_mlp_temporal_similarity)
        self.att_temporal =MixFormerBlock_DCF3(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                             qk_scale, use_layer_scale, layer_scale_init_value,
                                             mode='temporal',
                                             use_temporal_similarity=use_temporal_similarity,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames,
                                               use_dynamic_spatial_similarity=use_dynamic_spatial_similarity,
                                             use_mlp_temporal_similarity=use_mlp_temporal_similarity)
        # ST Graph branch
        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=17,
                                     mode='spatial',
                                    use_dynamic_spatial_similarity=use_dynamic_spatial_similarity)
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                         use_mlp_temporal_similarity=use_mlp_temporal_similarity)
        else:
            self.graph_spatial = MixFormerBlock_DCF3(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                             qk_scale, use_layer_scale, layer_scale_init_value,
                                             mode='spatial',
                                             use_temporal_similarity=use_temporal_similarity,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames,
                                             use_dynamic_spatial_similarity=use_dynamic_spatial_similarity,
                                            use_mlp_temporal_similarity=use_mlp_temporal_similarity)
            self.graph_temporal = MixFormerBlock_DCF3(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                             qk_scale, use_layer_scale, layer_scale_init_value,
                                             mode='temporal',
                                             use_temporal_similarity=use_temporal_similarity,
                                             neighbour_num=neighbour_num,
                                             n_frames=n_frames,
                                            use_dynamic_spatial_similarity=use_dynamic_spatial_similarity,
                                            use_mlp_temporal_similarity=use_mlp_temporal_similarity)
        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.dynamic_fusion = DynamicChannelFusion(dim)

    def _init_fusion(self):
        pass

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
    
            x_attn_temp = self.att_spatial(x)
            
            res_input_a = x_attn_temp.permute(0, 3, 1, 2)    # [B, C, T, J]
    
            x_attn_combined = self.cbam(res_input_a).permute(0, 2, 3, 1) # [B, T, J, C]
            
            x_attn = self.att_temporal(x_attn_combined)

            x_graph_temp = self.graph_temporal(x)
           
            res_input_g = x_graph_temp.permute(0, 3, 1, 2)
            
            x_graph_combined = self.cbam(res_input_g).permute(0, 2, 3, 1)
            
            x_graph = self.graph_spatial(x_graph_combined)
            
            
        

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            x = self.dynamic_fusion(x_graph, x_attn)
        else:
            x = (x_attn + x_graph) * 0.5

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,use_dynamic_spatial_similarity=False, use_mlp_temporal_similarity=False):
    """
    generates MotionAGFormer layers
    """
    layers = []
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
    for i in range(n_layers):
        layers.append(MotionAGFormerBlock_DCF3(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=dpr[i],
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                         use_dynamic_spatial_similarity=use_dynamic_spatial_similarity, # <--- 传递
                                     use_mlp_temporal_similarity=use_mlp_temporal_similarity))
    layers = nn.Sequential(*layers)

    return layers


class MotionAGFormer_DCF3(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243,use_dynamic_spatial_similarity=False, use_mlp_temporal_similarity=False):
        """
        :param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_rep: Motion representation dimension
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param use_tcn: If true, uses MS-TCN for temporal part of the graph branch.
        :param graph_only: Uses GCN instead of GraphFormer in the graph branch.
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames,
                                   use_dynamic_spatial_similarity=use_dynamic_spatial_similarity, # <--- 传递
                                     use_mlp_temporal_similarity=use_mlp_temporal_similarity)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False, return_intermediates=None):
        x = self.joints_embed(x)
        x = x + self.pos_embed

        intermediates = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediates is not None and idx in return_intermediates:
                intermediates[idx] = x

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x
        x = self.head(x)

        if return_intermediates is not None:
            return x, intermediates
        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    
    b, t, j, c_in = 4, 243, 17, 3 
    dim_feat = 128  
    random_x = torch.randn((b, t, j, c_in)).to('cuda')

    
    model = MotionAGFormer_DCF3(
        n_layers=12,           
        dim_in=c_in,          
        dim_feat=dim_feat,    
        mlp_ratio=4, 
        hierarchical=False,
        use_tcn=False, 
        graph_only=False, 
        n_frames=t,
        use_dynamic_spatial_similarity=True, 
        use_mlp_temporal_similarity=True
    ).to('cuda')
    
    model.eval()

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter #: {model_params:,}")

    with torch.no_grad():
        out = model(random_x)
    print(f"Output shape: {out.shape}") 

    try:
        flops = profile_macs(model, random_x)
        print(f"Model FLOPS #: {flops:,}")
    except Exception as e:
        print("FLOPs calculation failed, but model runs!")

    import time
    for _ in range(10): _ = model(random_x)  
    start_time = time.time()
    num_iters = 100
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(random_x)
    fps = num_iters / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    assert out.shape == (b, t, j, 3), f"Output shape error! Got {out.shape}"
    print("Test Passed!")

if __name__ == '__main__':
    _test()
