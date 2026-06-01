import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, mlp_ratio=4,
                 dropout=0.1, layer_scale=1e-4):
        super().__init__()
        self.drug_attn_norm = nn.LayerNorm(hidden_dim)
        self.drug_attends_protein = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.protein_attn_norm = nn.LayerNorm(hidden_dim)
        self.protein_attends_drug = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.drug_mlp_norm = nn.LayerNorm(hidden_dim)
        self.drug_mlp = self._build_ffn(hidden_dim, mlp_ratio, dropout)
        self.protein_mlp_norm = nn.LayerNorm(hidden_dim)
        self.protein_mlp = self._build_ffn(hidden_dim, mlp_ratio, dropout)

        self.drug_mlp_scale = nn.Parameter(
            torch.ones(hidden_dim) * layer_scale)
        self.protein_mlp_scale = nn.Parameter(
            torch.ones(hidden_dim) * layer_scale)

        self.drug_cond_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.protein_cond_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.drug_dynamic_scale = nn.Linear(hidden_dim, hidden_dim)
        self.protein_dynamic_scale = nn.Linear(hidden_dim, hidden_dim)

        self.drug_prior_proj = nn.Linear(hidden_dim, hidden_dim)
        self.protein_prior_proj = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _build_ffn(dim, mlp_ratio, dropout):
        return nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, drug_hidden, protein_hidden,
                drug_prior=None, protein_prior=None):
        if drug_hidden.dim() == 2:
            drug_hidden = drug_hidden.unsqueeze(1)
        if protein_hidden.dim() == 2:
            protein_hidden = protein_hidden.unsqueeze(1)

        drug_norm = self.drug_attn_norm(drug_hidden)
        prot_global = protein_hidden.mean(dim=1, keepdim=True)
        drug_q = drug_norm * self.drug_cond_gate(prot_global)
        if drug_prior is not None:
            drug_q = drug_q + self.drug_prior_proj(drug_prior)
        drug_attn_out, _ = self.drug_attends_protein(
            drug_q, protein_hidden, protein_hidden)
        drug_dyn = torch.sigmoid(self.drug_dynamic_scale(prot_global))
        drug_hidden = drug_hidden + drug_dyn * drug_attn_out

        protein_norm = self.protein_attn_norm(protein_hidden)
        drug_global = drug_hidden.mean(dim=1, keepdim=True)
        protein_q = protein_norm * self.protein_cond_gate(drug_global)
        if protein_prior is not None:
            protein_q = protein_q + self.protein_prior_proj(protein_prior)
        protein_attn_out, _ = self.protein_attends_drug(
            protein_q, drug_hidden, drug_hidden)
        protein_dyn = torch.sigmoid(self.protein_dynamic_scale(drug_global))
        protein_hidden = protein_hidden + protein_dyn * protein_attn_out

        drug_hidden = drug_hidden + self.drug_mlp_scale * self.drug_mlp(
            self.drug_mlp_norm(drug_hidden))
        protein_hidden = protein_hidden + self.protein_mlp_scale * \
            self.protein_mlp(self.protein_mlp_norm(protein_hidden))

        return drug_hidden, protein_hidden, drug_attn_out, protein_attn_out
