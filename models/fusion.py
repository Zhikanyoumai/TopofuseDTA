import torch
from torch import nn
import einops


class ResidueImportanceEvolution(nn.Module):
    def __init__(self, hidden_dim, num_layers, decay=0.8, beta=0.3):
        super().__init__()
        self.decay = decay
        self.beta = beta

        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1))
            for _ in range(num_layers)])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1))
            for _ in range(num_layers)])

    def forward(self, node_features, layer_idx, cumulative_score):
        current_score = self.scorers[layer_idx](node_features)

        if cumulative_score is None:
            gate_input = torch.cat(
                [current_score, torch.zeros_like(current_score)], dim=-1)
            modulation = torch.tanh(self.gates[layer_idx](gate_input))
            new_cumulative = current_score.detach()
        else:
            gate_input = torch.cat(
                [current_score, cumulative_score], dim=-1)
            modulation = torch.tanh(self.gates[layer_idx](gate_input))
            new_cumulative = (self.decay * cumulative_score
                              + (1 - self.decay) * current_score.detach())

        modulated = node_features * (1.0 + self.beta * modulation)
        return modulated, new_cumulative


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features, n1, n2, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features * 4, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1))

    def forward(self, v_chem, v_prot, v_seq):
        c1, c2 = v_chem.shape[1], v_prot.shape[1]
        ce = einops.repeat(v_chem, 'b c1 h -> b c1 c2 h', c2=c2)
        pe = einops.repeat(v_prot, 'b c2 h -> b c1 c2 h', c1=c1)
        te = einops.repeat(v_seq, 'b h -> b c1 c2 h', c1=c1, c2=c2)
        act = torch.cat([ce, pe, te, ce * pe], dim=-1)
        B, c1, c2, fd = act.shape
        y = self.mlp(act.view(-1, fd)).view(B, c1, c2)
        return y
