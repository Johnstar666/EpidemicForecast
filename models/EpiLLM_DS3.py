import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from layers.mlp import MLP
from layers.influence_matrix import InfluenceMatrixLayer
from torch_geometric.nn import GATv2Conv
# from accelerate import dispatch_model

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim ,time_point=3,node_num=105):
        super().__init__()
        self.temporal_gat = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim // 4,
            edge_dim=1,
            heads=4,  
            dropout=0.1
        )
        self.adj_proj = nn.Sequential(
            nn.Linear(node_num*time_point, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim)
        )
        # SSTKG-inspired influence matrix layer for epidemic branch
        self.influence_layer = InfluenceMatrixLayer(num_nodes=node_num, feat_dim=node_dim)
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.temporal_forward_weight = nn.Parameter(torch.tensor(1.0)) 
        self.temporal_backward_weight = nn.Parameter(torch.tensor(0.5))
        self.time_gate = nn.Parameter(torch.ones(time_point,1))
        self.time_gate2 = nn.Parameter(torch.ones(time_point,1)) #3 time step
    def build_temporal_edges(self, num_nodes,time_point=3):
        base = torch.arange(num_nodes)
        edges = []
        for k in range(time_point-1):
            src = base + k * num_nodes
            dst = base + (k+1) * num_nodes
            edges.append(torch.stack([src, dst], dim=0))
            edges.append(torch.stack([dst, src], dim=0))
        return torch.cat(edges, dim=1) if edges else torch.empty(time_point-1, 0, dtype=torch.long)

    def forward(self, node_feats, adj_sequence):

        num_nodes = node_feats.size(0)
        time_point = node_feats.size(1)
        # spatial_feats = self.spatial_proj(node_feats)  # [105,3,768]
        
        joint_feats = node_feats.reshape(-1, node_feats.size(-1))  # [315,768]
        
        # SSTKG-inspired: compute influence matrix from adj + case features
        influence_matrix = self.influence_layer(adj_sequence, node_feats)  # [T, N, N]
        # Epidemic branch uses influence matrix instead of raw adjacency
        adj = influence_matrix.permute(1,0,2) # [T, N, N] → [N, T, N]
        adj = adj.reshape(adj.size(0),-1) # [N, T*N]
        adj_pred = self.adj_proj(adj) # [N, hidden_dim]
        temporal_edges = self.build_temporal_edges(num_nodes,time_point).to(node_feats.device)
        # temporal_edge_weights = torch.ones(temporal_edges.size(1), 1, device=node_feats.device)
        if temporal_edges.size(1) > 0:
            src_time = temporal_edges[0] // num_nodes
            dst_time = temporal_edges[1] // num_nodes
            delta_time = dst_time - src_time

            temporal_edge_weights = torch.where(
                delta_time == 1,
                self.temporal_forward_weight,
                self.temporal_backward_weight
            ).unsqueeze(-1)
        else:
            temporal_edge_weights = torch.empty(0, 1, device=node_feats.device)        
        
        spatial_edges = []
        for k in range(time_point):
            edge_index = adj_sequence[k].nonzero(as_tuple=False).t()  #  [2, E]
            edge_index += k * num_nodes
            spatial_edges.append(edge_index)
        spatial_edges = torch.cat(spatial_edges, dim=1)  # [2, total_E]
        
        all_edges = torch.cat([temporal_edges, spatial_edges], dim=1)  # [2, total_E]
        
        if spatial_edges.size(1) > 0:
            k_values = spatial_edges[0] // num_nodes  # [total_E]
            src_nodes = spatial_edges[0] % num_nodes   # [total_E]
            dst_nodes = spatial_edges[1] % num_nodes   # [total_E]
            spatial_edge_weights = adj_sequence[k_values, src_nodes, dst_nodes].unsqueeze(-1)  # [total_E,1]
        else:
            spatial_edge_weights = torch.empty(0, 1, device=node_feats.device)
        
        all_edge_weights = torch.cat([temporal_edge_weights, spatial_edge_weights])  # [total_E,1]
        
        joint_emb = self.temporal_gat(joint_feats, all_edges, all_edge_weights)  # [315,768]
        # joint_emb2  = self.temporal_gat2(joint_emb, all_edges, all_edge_weights)  # [315,768]
        weighted_feats = joint_emb.reshape(num_nodes, time_point, -1) * self.time_gate  # [105,768]
        # weighted_feats2 = joint_emb2.reshape(num_nodes, time_point, -1) * self.time_gate2  # [105,768]
        return weighted_feats.sum(dim=1),adj_pred  # [105,768]




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = torch.device('cuda')
        print(self.device)
        self.hidden_dim = 1536
        self.seq_len = configs.seq_len
        self.token_len = configs.token_len
        self.node_num = configs.node_num
        self.encoder = SpatioTemporalEncoder(
            node_dim=self.token_len,  
            hidden_dim=self.hidden_dim,
            time_point=self.token_len, 
            node_num=self.node_num 
        ).to(self.device)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map='balanced_low_0',
            torch_dtype=torch.float16 if configs.use_amp else torch.float32   
        )
        self.hidden_dim_of_qwen = 1536
        
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False

        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_dim_of_qwen, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.token_len)
        ).to(self.device)
        self.adj_mlp_head = nn.Sequential(
            nn.Linear(self.hidden_dim_of_qwen, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.node_num*self.token_len)
        ).to(self.device)
            
    def process_sequence(self, x, adj):
        batch_size = x.size(0)
        time_steps = self.seq_len//self.token_len
        time_point = self.token_len
        node_num = x.size(2)
        feat_dim = x.size(3)
        x = x.view(batch_size, time_steps, time_point, node_num, feat_dim)  # [b,18,105,3] → [b,6,3,105,3]
        adj = adj.view(batch_size, time_steps, time_point, node_num, node_num)  # [b,18,105,105] → [b,6,3,105,105]
        
        embeddings = []
        emb_adj = []
        for t in range(time_steps): 
            # [b,3,105,3]
            x_step = x[:, t] 
            adj_step = adj[:, t]  # [b,3,105,105]
            
            step_embs = []
            step_adj = []
            for b in range(batch_size):
                single_x = x_step[b].permute(1, 0, 2)
                # [3,105,105]
                single_adj = adj_step[b]

                emb,adj_emb = self.encoder(single_x, single_adj)  # [105,768]
                step_embs.append(emb)
                step_adj.append(adj_emb)

            batch_emb = torch.stack(step_embs, dim=0)  # [b,105,768]
            batch_adj = torch.stack(step_adj, dim=0) # [b,105,768]
            embeddings.append(batch_emb)
            emb_adj.append(batch_adj)
        
        return torch.stack(embeddings, dim=1), torch.stack(emb_adj)  # [b,6,105,768]    

    def forecast(self, x_enc, adj):
        
        # means = x_enc.mean(dim=(1,2,3), keepdim=True)  # [b,1,1,1]
        # std = torch.sqrt(torch.var(x_enc, dim=(1,2,3), keepdim=True) + 1e-5)
        # x_norm = (x_enc - means) / std  # [b,18,105,3]
        
        node_embeds,adj_emb = self.process_sequence(x_enc, adj)  
        # node_embeds = self.encoder(x_enc, adj)  # [b,105,768]
        batch_size = node_embeds.size(0)
        tok_num = node_embeds.size(1)
        node_num = node_embeds.size(2)
        gpt_input = node_embeds.reshape(batch_size, tok_num * node_num, self.hidden_dim_of_qwen) # [b,630,768]
        gpt_adj_input = adj_emb.reshape(batch_size, tok_num * node_num, self.hidden_dim_of_qwen)  # [b,630,768]
        outputs = self.qwen(inputs_embeds=gpt_input,output_hidden_states=True)  # [b,630,768]
        outputs = outputs.hidden_states[-1] 
        adj_outputs = self.qwen(inputs_embeds=gpt_adj_input, output_hidden_states=True)# [b,630,768]
        adj_outputs = adj_outputs.hidden_states[-1]
        final_out = outputs.view(batch_size, tok_num, node_num, self.hidden_dim_of_qwen)  # [b,6,105,768]
        adj_out = adj_outputs.view(batch_size, tok_num, node_num, self.hidden_dim_of_qwen) # [b,6,105,768]
        self.adj_mlp_head = self.adj_mlp_head.to(adj_out.device)
        adj_predictions = self.adj_mlp_head(adj_out).permute(0,1,3,2)  # [b,6,315, 105]
        self.mlp_head = self.mlp_head.to(final_out.device)
        predictions = self.mlp_head(final_out).permute(0,1,3,2)  # [b, 18, 105]
        adj_predictions = adj_predictions.reshape(batch_size, predictions.size(1)*predictions.size(2),node_num,node_num)
        predictions = predictions.reshape(batch_size, predictions.size(1)*predictions.size(2), predictions.size(3))
        return predictions,adj_predictions
    
    def forward(self, x_enc, adj):
        return self.forecast(x_enc, adj)