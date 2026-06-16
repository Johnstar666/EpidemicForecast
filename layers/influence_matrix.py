import torch
import torch.nn as nn
import torch.nn.functional as F


class InfluenceMatrixLayer(nn.Module):
    """
    SSTKG-inspired influence matrix layer for the epidemic branch.

    Replaces the raw adjacency matrix in the epidemic branch with a refined
    influence matrix that combines structural connectivity (population flow)
    with epidemiological similarity (case pattern coupling).

    Key design from SSTKG (WWW 2024):
    - p (self-influence): diagonal element, how much an entity influences itself
    - I (cross-influence): off-diagonal scaling, how much other entities influence this one
    - influence[i][j] = how much entity j influences entity i

    The influence matrix fuses two signals:
    1. Structural adjacency (spatial connectivity / population flow)
    2. Case-pattern similarity (epidemiological coupling between regions)
    via a learnable blending parameter alpha.
    """

    def __init__(self, num_nodes, feat_dim):
        super(InfluenceMatrixLayer, self).__init__()
        # SSTKG-inspired learnable parameters
        self.p = nn.Parameter(torch.ones(num_nodes))        # self-influence (diagonal)
        self.I = nn.Parameter(torch.ones(num_nodes))        # cross-influence scaling (off-diagonal)
        # Balance between structural adjacency and epidemiological similarity
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Project case features for similarity computation
        self.case_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()
        )

    def forward(self, adj_sequence, node_feats):
        """
        Args:
            adj_sequence: [T, N, N] adjacency matrices per time step
            node_feats:   [N, T, D] case features per node per time step
        Returns:
            influence_matrix: [T, N, N] influence matrices per time step
        """
        T, N, _ = adj_sequence.shape
        alpha = torch.sigmoid(self.alpha)

        influence_list = []
        for t in range(T):
            adj_t = adj_sequence[t]              # [N, N]
            feat_t = node_feats[:, t, :]          # [N, D]

            # --- Normalize adjacency to [0, 1] for blending with similarity ---
            adj_max = adj_t.abs().max()
            if adj_max > 0:
                adj_norm = adj_t / adj_max
            else:
                adj_norm = adj_t

            # --- Compute case-pattern similarity ---
            proj_feat = self.case_proj(feat_t)                          # [N, D]
            feat_norm = F.normalize(proj_feat, p=2, dim=-1)             # [N, D]
            case_sim = torch.mm(feat_norm, feat_norm.t())               # [N, N]
            case_sim = (case_sim + 1.0) / 2.0                           # map [-1,1] → [0,1]

            # --- Blend adjacency with case similarity ---
            blended = alpha * adj_norm + (1.0 - alpha) * case_sim      # [N, N]

            # --- Apply SSTKG-style influence: diagonal = p, off-diagonal scaled by I ---
            diag_mask = torch.eye(N, device=adj_t.device)
            influence_t = (blended * (1.0 - diag_mask) * self.I.unsqueeze(0)
                           + torch.diag(self.p) * diag_mask)

            influence_list.append(influence_t)

        return torch.stack(influence_list, dim=0)   # [T, N, N]
