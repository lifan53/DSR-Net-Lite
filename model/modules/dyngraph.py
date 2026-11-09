import math
import torch
from torch import nn

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4, mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1, connections=None, use_dynamic_spatial_similarity=False, use_mlp_temporal_similarity=False): 
        super().__init__()
        
        self.relu = nn.ReLU()
        self.neighbour_num = neighbour_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mode = mode
        self.num_nodes = num_nodes
        
        self.use_temporal_similarity = use_temporal_similarity
        self.use_dynamic_spatial_similarity = use_dynamic_spatial_similarity
        self.use_mlp_temporal_similarity = use_mlp_temporal_similarity

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

       
        self.use_projection = False
        if (mode == 'temporal' and use_mlp_temporal_similarity) or \
           (mode == 'spatial' and use_dynamic_spatial_similarity):
            self.use_projection = True
            self.W_Q = nn.Linear(self.dim_in, self.dim_in // 2) 
            self.W_K = nn.Linear(self.dim_in, self.dim_in // 2)
            nn.init.xavier_uniform_(self.W_Q.weight)
            nn.init.xavier_uniform_(self.W_K.weight)
        
    
        self.PA = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.constant_(self.PA, 1e-6) 

        
        self.softmax = nn.Softmax(dim=-1)

        self._init_gcn()

        self.connections = connections
        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        else:
            self.adj = self._init_temporal_adj(temporal_connection_len)

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS
        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    def _init_temporal_adj(self, connection_length):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            try:
                for j in range(connection_length + 1):
                    adj[i, i + j] = 1
            except IndexError:
                pass
        return adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        b, t, j, c = x.shape
        
        if self.mode == 'temporal':
            x = x.transpose(1, 2)
            curr_nodes = t
            x_reshaped = x.reshape(-1, t, c)
        else: # spatial
            curr_nodes = j
            x_reshaped = x.reshape(-1, j, c)

        base_adj = self.change_adj_device_to_cuda(self.adj)
        
        use_dynamic = False
        if self.mode == 'temporal' and self.use_temporal_similarity:
            use_dynamic = True
        elif self.mode == 'spatial' and self.use_dynamic_spatial_similarity:
            use_dynamic = True

        if use_dynamic:
            if self.use_projection:
                Q = self.W_Q(x_reshaped) 
                K = self.W_K(x_reshaped)
                similarity = Q @ K.transpose(1, 2) 
                similarity = similarity / math.sqrt(Q.size(-1)) 
            else:
                similarity = x_reshaped @ x_reshaped.transpose(1, 2)
            
            similarity = similarity + self.PA 


            self.last_sim = similarity.detach() 

            topk_val, _ = similarity.topk(k=self.neighbour_num, dim=-1, largest=True)
            threshold = topk_val[..., -1].view(similarity.shape[0], curr_nodes, 1)
            
            mask = (similarity >= threshold)
            similarity_masked = similarity.masked_fill(~mask, -1e9)
            
            adj = self.softmax(similarity_masked)
            self.last_adj = adj.detach()  
            
        else:
            adj = base_adj.unsqueeze(0) + self.PA.unsqueeze(0)
            adj = (adj > 0.5).float()
            adj = adj.repeat(x_reshaped.shape[0], 1, 1)

        if use_dynamic:
            aggregate = adj @ self.V(x_reshaped)
        else:
            norm_adj = self.normalize_digraph(adj) 
            aggregate = norm_adj @ self.V(x_reshaped)

        if self.dim_in == self.dim_out:
            out = x_reshaped + self.batch_norm(aggregate + self.U(x_reshaped))
        else:
            out = self.batch_norm(aggregate + self.U(x_reshaped))
        
        out = self.relu(out)

        if self.mode == 'spatial':
            out = out.reshape(-1, t, j, self.dim_out)
        else:
            out = out.reshape(-1, j, t, self.dim_out).transpose(1, 2)
            
        return out

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape
        node_degrees = adj.detach().sum(dim=-1).clamp(min=1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.diag_embed(deg_inv_sqrt)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)
        return norm_adj