import torch
from torch import nn
from torch_geometric.nn import SAGPooling, LayerNorm, global_add_pool
from torch_geometric.utils import degree
from torch_scatter import scatter_mean

from config import TOTAL_TOPO_INPUT_DIM
from .gtat_conv import GTATConv
from .layers import EnhancedSAGEConv


class EnhancedGATBlock(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats,
                 ifConv1d=False, window_size=5,
                 dropout_1=0.4, dropout_2=0.1, layernorm=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.ifConv1d = ifConv1d
        self.topology_channels = 15

        if self.ifConv1d:
            self.conv1d = nn.Conv1d(
                in_features, in_features,
                kernel_size=window_size, padding=window_size // 2)

        self.topo_mlp = nn.Linear(
            TOTAL_TOPO_INPUT_DIM, self.topology_channels)

        self.gtat_standard = GTATConv(
            in_features, head_out_feats, n_heads,
            topology_channels=self.topology_channels,
            dropout=dropout_1, concat=True, share_weights=True)

        h1 = max(1, n_heads // 3)
        h2 = max(1, n_heads // 2)
        h3 = n_heads
        f1 = head_out_feats // 3
        f2 = head_out_feats // 3
        f3 = head_out_feats - f1 - f2

        self.gtat_multi_scale = nn.ModuleList([
            GTATConv(in_features, f1, h1,
                     topology_channels=self.topology_channels,
                     dropout=dropout_1, concat=True, share_weights=True),
            GTATConv(in_features, f2, h2,
                     topology_channels=self.topology_channels,
                     dropout=dropout_1, concat=True, share_weights=True),
            GTATConv(in_features, f3, h3,
                     topology_channels=self.topology_channels,
                     dropout=dropout_1, concat=True, share_weights=True),
        ])

        self.multi_scale_total_feats = f1 * h1 + f2 * h2 + f3 * h3
        total = n_heads * head_out_feats

        self.affine_in = nn.Parameter(torch.ones(1, total))
        self.affine_bias = nn.Parameter(torch.zeros(1, total))
        self.self_attention = nn.Sequential(
            nn.Linear(total, total), nn.Tanh(),
            nn.Linear(total, total), nn.Sigmoid())

        if self.multi_scale_total_feats != total:
            self.dimension_matcher = nn.Linear(
                self.multi_scale_total_feats, total)
        else:
            self.dimension_matcher = None

        self.fusion_gate = nn.Sequential(
            nn.Linear(total * 2, total), nn.ReLU(),
            nn.Dropout(dropout_2 * 0.5),
            nn.Linear(total, 2), nn.Softmax(dim=-1))
        self.residual_transform = nn.Linear(in_features, total)

        self.readout = SAGPooling(total, min_score=-1)
        self.norm = LayerNorm(total) if layernorm else nn.BatchNorm1d(total)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_2)

    def forward(self, data):
        orig_x = data.x

        if self.ifConv1d and hasattr(self, 'conv1d'):
            data.x = self.conv1d(
                data.x.t().unsqueeze(0)).squeeze(0).t()

        deg = degree(
            data.edge_index[0], num_nodes=data.x.size(0)).unsqueeze(1)
        if data.edge_index.shape[1] > 0:
            nd = scatter_mean(
                deg[data.edge_index[1]], data.edge_index[0],
                dim=0, dim_size=data.x.size(0))
            nd = nd.unsqueeze(1) if nd.dim() == 1 else nd
        else:
            nd = torch.zeros_like(deg)

        online_topo = torch.cat(
            [deg, nd.view(deg.size(0), -1)[:, :1]], dim=1)
        precomputed_topo = data.topo_feat
        topo_input = torch.cat([online_topo, precomputed_topo], dim=1)
        topology = self.topo_mlp(topo_input)

        xs_std, _ = self.gtat_standard(
            data.x, data.edge_index, topology=topology)
        xs_ms = torch.cat([
            g(data.x, data.edge_index, topology=topology)[0]
            for g in self.gtat_multi_scale], dim=-1)

        if self.dimension_matcher is not None:
            xs_ms = self.dimension_matcher(xs_ms)

        xs_std_n = xs_std * self.affine_in + self.affine_bias
        xs_ms_n = xs_ms * self.affine_in + self.affine_bias
        xs_std_e = xs_std_n * self.self_attention(xs_std_n)
        xs_ms_e = xs_ms_n * self.self_attention(xs_ms_n)

        w = self.fusion_gate(torch.cat([xs_std_e, xs_ms_e], dim=-1))
        data.x = (w[:, 0:1] * xs_std_e + w[:, 1:2] * xs_ms_e
                  + self.residual_transform(orig_x))

        att_x, _, _, att_batch, _, att_scores = self.readout(
            data.x, data.edge_index, batch=data.batch)
        global_emb = global_add_pool(att_x, att_batch)
        data.x = self.dropout(self.relu(self.norm(data.x)))

        return data, global_emb, att_scores


class GraphSAGEBlock(nn.Module):
    def __init__(self, in_features, out_features,
                 aggrs=("mean", "max", "sum"),
                 ifConv1d=False, window_size=5,
                 dropout_1=0.4, dropout_2=0.1, layernorm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ifConv1d = ifConv1d

        dims = []
        rem = out_features
        for _ in range(len(aggrs) - 1):
            d = out_features // len(aggrs)
            dims.append(d)
            rem -= d
        dims.append(rem)

        if self.ifConv1d:
            self.conv1d = nn.Conv1d(
                in_features, in_features,
                kernel_size=window_size, padding=window_size // 2)

        self.sage_convs = nn.ModuleList([
            EnhancedSAGEConv(in_features, dims[i], aggr=a)
            for i, a in enumerate(aggrs)])

        self.feature_transform = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(), nn.Dropout(dropout_1))

        self.readout = SAGPooling(out_features, min_score=-1)
        self.norm = LayerNorm(out_features) if layernorm \
            else nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_2)

    def forward(self, data):
        orig_x = data.x

        if self.ifConv1d and hasattr(self, 'conv1d'):
            data.x = self.conv1d(
                data.x.t().unsqueeze(0)).squeeze(0).t()

        x = self.feature_transform(torch.cat(
            [s(data.x, data.edge_index) for s in self.sage_convs],
            dim=-1))

        if x.shape[-1] == orig_x.shape[-1]:
            x = x + orig_x
        data.x = x

        att_x, _, _, att_batch, _, att_scores = self.readout(
            data.x, data.edge_index, batch=data.batch)
        global_emb = global_add_pool(att_x, att_batch)
        data.x = self.dropout(self.relu(self.norm(data.x)))

        return data, global_emb, att_scores
