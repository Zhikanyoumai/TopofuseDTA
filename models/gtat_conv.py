import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_geometric.typing import OptTensor


class GTATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels, out_channels, heads, topology_channels=15,
                 concat=True, negative_slope=0.2, dropout=0.,
                 add_self_loops=True, bias=True, share_weights=False, **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.topology_channels = topology_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att2 = nn.Parameter(torch.Tensor(1, heads, self.topology_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha1 = None
        self._alpha2 = None
        self.bias2 = nn.Parameter(torch.Tensor(self.topology_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        glorot(self.att2)
        if self.bias is not None:
            zeros(self.bias)
        zeros(self.bias2)

    def forward(self, x, edge_index, topology, size=None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)

        topology = topology.unsqueeze(dim=1).repeat(1, self.heads, 1)
        x_l = torch.cat((x_l, topology), dim=-1)
        x_r = torch.cat((x_r, topology), dim=-1)

        if self.add_self_loops and isinstance(edge_index, torch.Tensor):
            num_nodes = x_l.size(0)
            if x_r is not None:
                num_nodes = min(num_nodes, x_r.size(0))
            if size is not None:
                num_nodes = min(size[0], size[1])
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out_all = self.propagate(edge_index, x=(x_l, x_r), size=size)
        out = out_all[:, :, :self.out_channels]
        out2 = out_all[:, :, self.out_channels:]
        alpha1, self._alpha1 = self._alpha1, None
        alpha2, self._alpha2 = self._alpha2, None

        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        out2 = out2.mean(dim=1) + self.bias2

        if isinstance(return_attention_weights, bool):
            if return_attention_weights:
                return out, out2, (edge_index, alpha1)
            return out, out2
        return out, out2

    def message(self, x_j, x_i, index, ptr, size_i):
        x = x_i + x_j
        alpha1 = F.leaky_relu(
            (x[:, :, :self.out_channels] * self.att).sum(-1),
            self.negative_slope)
        alpha2 = F.leaky_relu(
            (x[:, :, self.out_channels:] * self.att2).sum(-1),
            self.negative_slope)
        alpha1 = softmax(alpha1, index, ptr, size_i)
        alpha2 = softmax(alpha2, index, ptr, size_i)
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        alpha1 = F.dropout(alpha1, p=self.dropout, training=self.training)
        alpha2 = F.dropout(alpha2, p=self.dropout, training=self.training)
        return torch.cat(
            (x_j[:, :, :self.out_channels] * alpha2.unsqueeze(-1),
             x_j[:, :, self.out_channels:] * alpha1.unsqueeze(-1)), dim=-1)
