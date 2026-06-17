import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SSTKGConstraintLayer(nn.Module):
    """
    SSTKG-based constraint layer for multi-step epidemic forecasting.
    
    This layer addresses two key challenges:
    1. Multi-step prediction error accumulation: Validates predictions against 
       historical influence patterns and corrects deviations
    2. Quantified propagation analysis: Generates interpretable influence graphs
       showing transmission sources and impact ranges
    
    Based on SSTKG principles (WWW 2024):
    - High-influence regions should significantly affect their connected regions
    - Prediction deviations from influence patterns indicate potential errors
    - Dynamic influence matrices capture temporal evolution of transmission patterns
    """
    
    def __init__(self, num_nodes: int, influence_threshold: float = 0.05):
        super(SSTKGConstraintLayer, self).__init__()
        self.num_nodes = num_nodes
        self.influence_threshold = influence_threshold
        
        # Learnable correction parameters
        self.correction_weight = nn.Parameter(torch.tensor(0.5))
        self.influence_smoothing = nn.Parameter(torch.tensor(0.1))
        
    def compute_influence_matrix(self, adj_matrix: torch.Tensor, 
                               case_features: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic influence matrix combining structural and epidemiological signals.
        
        Args:
            adj_matrix: [N, N] - Structural adjacency (population flow/mobility)
            case_features: [N, D] - Current case features for similarity computation
            
        Returns:
            influence_matrix: [N, N] - Combined influence matrix
        """
        N = adj_matrix.size(0)
        
        # Normalize adjacency matrix
        adj_max = adj_matrix.abs().max()
        if adj_max > 0:
            adj_norm = adj_matrix / adj_max
        else:
            adj_norm = adj_matrix
            
        # Compute case pattern similarity
        feat_norm = F.normalize(case_features, p=2, dim=-1)
        case_sim = torch.mm(feat_norm, feat_norm.t())
        case_sim = (case_sim + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # Blend with learnable parameter
        alpha = torch.sigmoid(self.influence_smoothing)
        blended = alpha * adj_norm + (1.0 - alpha) * case_sim
        
        return blended
    
    def validate_prediction_consistency(self, 
                                      current_pred: torch.Tensor,
                                      previous_pred: torch.Tensor,
                                      influence_matrix: torch.Tensor,
                                      historical_cases: torch.Tensor) -> torch.Tensor:
        """
        Validate prediction consistency with influence patterns and correct errors.
        
        Args:
            current_pred: [N] - Current step prediction
            previous_pred: [N] - Previous step prediction  
            influence_matrix: [N, N] - Current influence matrix
            historical_cases: [T, N] - Historical case data for context
            
        Returns:
            corrected_pred: [N] - Error-corrected prediction
        """
        # Compute expected change based on influence patterns
        influence_effect = torch.matmul(influence_matrix, previous_pred.unsqueeze(-1)).squeeze(-1)
        
        # Compute deviation from expected pattern
        deviation = current_pred - influence_effect
        
        # Identify high-influence regions that should have significant impact
        high_influence_mask = (influence_matrix.sum(dim=1) > self.influence_threshold)
        
        # Apply stronger correction to high-influence regions
        correction_factor = torch.sigmoid(self.correction_weight)
        corrected_pred = current_pred - correction_factor * deviation * high_influence_mask.float()
        
        return corrected_pred
    
    def generate_propagation_graph(self, 
                                 influence_matrix: torch.Tensor,
                                 target_region: int,
                                 top_k: int = 5) -> dict:
        """
        Generate quantified propagation graph for interpretability.
        
        Args:
            influence_matrix: [N, N] - Current influence matrix
            target_region: int - Target region ID to analyze
            top_k: int - Number of top influencers to return
            
        Returns:
            propagation_info: dict containing propagation analysis
        """
        # Get incoming influences to target region
        incoming_influences = influence_matrix[target_region, :]
        
        # Filter significant influences
        significant_mask = incoming_influences > self.influence_threshold
        significant_influences = incoming_influences * significant_mask
        
        # Get top-k influencers
        top_indices = torch.argsort(significant_influences, descending=True)[:top_k]
        top_weights = significant_influences[top_indices]
        
        # Remove zero influences
        valid_mask = top_weights > 0
        top_indices = top_indices[valid_mask]
        top_weights = top_weights[valid_mask]
        
        propagation_info = {
            'target_region': target_region,
            'top_influencers': top_indices.cpu().numpy().tolist(),
            'influence_weights': top_weights.cpu().numpy().tolist(),
            'total_influence': significant_influences.sum().item(),
            'num_significant_sources': valid_mask.sum().item()
        }
        
        return propagation_info
    
    def analyze_temporal_evolution(self, 
                                 influence_matrices: torch.Tensor,
                                 region_id: int) -> dict:
        """
        Analyze temporal evolution of influence patterns for a specific region.
        
        Args:
            influence_matrices: [T, N, N] - Time-series of influence matrices
            region_id: int - Region to analyze
            
        Returns:
            evolution_info: dict containing temporal analysis
        """
        T = influence_matrices.size(0)
        total_influences = []
        out_degrees = []
        
        for t in range(T):
            inf_mat = influence_matrices[t]
            # Total incoming influence to region
            total_influence = inf_mat[region_id, :].sum().item()
            total_influences.append(total_influence)
            
            # Out-degree (how much this region influences others)
            out_degree = inf_mat[:, region_id].sum().item()
            out_degrees.append(out_degree)
        
        evolution_info = {
            'region_id': region_id,
            'total_influences': total_influences,
            'out_degrees': out_degrees,
            'peak_influence_time': np.argmax(total_influences).item(),
            'influence_trend': 'increasing' if total_influences[-1] > total_influences[0] else 'decreasing'
        }
        
        return evolution_info
    
    def forward(self, 
                predictions: torch.Tensor,
                adj_sequence: torch.Tensor,
                case_features: torch.Tensor,
                historical_cases: torch.Tensor,
                apply_correction: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Main forward pass applying SSTKG constraints.
        
        Args:
            predictions: [B, T_pred, N] - Multi-step predictions
            adj_sequence: [T_hist, N, N] - Historical adjacency sequence
            case_features: [B, T_hist, N, D] - Historical case features
            historical_cases: [B, T_hist, N] - Historical case values
            apply_correction: bool - Whether to apply error correction
            
        Returns:
            corrected_predictions: [B, T_pred, N] - Potentially corrected predictions
            analysis_results: dict - Propagation analysis results
        """
        B, T_pred, N = predictions.shape
        T_hist = adj_sequence.size(0)
        
        # Use last historical time step for initial influence matrix
        last_adj = adj_sequence[-1]  # [N, N]
        last_case_feat = case_features[:, -1, :, :].mean(dim=0)  # [N, D]
        
        influence_matrix = self.compute_influence_matrix(last_adj, last_case_feat)
        
        corrected_preds = predictions.clone()
        analysis_results = {}
        
        if apply_correction:
            # Apply step-by-step correction for multi-step predictions
            for b in range(B):
                prev_pred = historical_cases[b, -1, :]  # Last historical value
                
                for t in range(T_pred):
                    current_pred = corrected_preds[b, t, :]
                    
                    # Validate and correct prediction
                    corrected_pred = self.validate_prediction_consistency(
                        current_pred, prev_pred, influence_matrix, historical_cases[b]
                    )
                    
                    corrected_preds[b, t, :] = corrected_pred
                    prev_pred = corrected_pred
        
        # Generate propagation analysis for first batch and first prediction step
        if B > 0:
            analysis_results['propagation_graphs'] = []
            for region in range(min(N, 10)):  # Analyze first 10 regions
                prop_graph = self.generate_propagation_graph(influence_matrix, region)
                analysis_results['propagation_graphs'].append(prop_graph)
        
        return corrected_preds, analysis_results