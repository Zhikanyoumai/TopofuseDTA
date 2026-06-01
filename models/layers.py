from torch import nn
from torch_geometric.nn import SAGEConv


class EnhancedSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean',
                 normalize=False, bias=True, **kwargs):
        super().__init__()
        self.sage_conv = SAGEConv(
            in_channels, out_channels, aggr=aggr,
            normalize=normalize, bias=bias, **kwargs)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, **kwargs):
        out = self.sage_conv(x, edge_index, **kwargs)
        return self.dropout(self.activation(self.batch_norm(out)))
