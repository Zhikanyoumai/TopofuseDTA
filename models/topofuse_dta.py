import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import SAGPooling, LayerNorm, global_add_pool
from torch_geometric.data import Data
import pytorch_lightning as pl

from .blocks import EnhancedGATBlock, GraphSAGEBlock
from .transformer import TransformerBlock
from .fusion import CoAttentionLayer, ResidueImportanceEvolution
from utils.metrics import get_metrics_reg


def _deep_copy_pyg(data):
    nd = data.clone()
    nd.x = data.x.clone()
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        nd.edge_index = data.edge_index.clone()
    if hasattr(data, 'batch') and data.batch is not None:
        nd.batch = data.batch.clone()
    if hasattr(data, 'topo_feat') and data.topo_feat is not None:
        nd.topo_feat = data.topo_feat.clone()
    return nd


class TopoFuseDTA(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.chem_initial_norm = LayerNorm(cfg["chem_in_features"])
        self.prot_initial_norm = LayerNorm(cfg["prot_in_features"])

        self.chem_blocks = ModuleList()
        chem_in = cfg["chem_in_features"]
        chem_dims = []
        for i, (hof, nh) in enumerate(zip(
                cfg["chem_heads_out_feat_params"],
                cfg["chem_blocks_params"])):
            blk = EnhancedGATBlock(
                nh, chem_in, hof, ifConv1d=False,
                dropout_1=cfg["dropout_1"], dropout_2=cfg["dropout_2"])
            self.add_module(f"chem_block{i}", blk)
            self.chem_blocks.append(blk)
            chem_in = hof * nh
            chem_dims.append(chem_in)

        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.num_transformer_layers = cfg.get("num_transformer_layers", 6)
        self.drug_seq_proj = nn.Linear(
            cfg.get("drug_seq_dim", 768), self.hidden_dim)
        self.prot_seq_proj = nn.Linear(
            cfg.get("prot_seq_dim", 1152), self.hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=cfg.get("transformer_num_heads", 8),
                mlp_ratio=cfg.get("transformer_mlp_ratio", 4),
                dropout=cfg["dropout_3"],
                layer_scale=cfg.get("layer_scale", 1e-4))
            for _ in range(self.num_transformer_layers)])

        graph_repr_dim = chem_dims[-1]
        self.seq_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.seq_out_norm = nn.LayerNorm(self.hidden_dim)
        self.seq_out_proj = nn.Linear(self.hidden_dim, graph_repr_dim)

        self.prot_gat_blocks = ModuleList()
        self.prot_sage_blocks = ModuleList()
        prot_in = cfg["prot_in_features"]
        prot_dims = []
        for i, (hof, nh, ws) in enumerate(zip(
                cfg["prot_heads_out_feat_params"],
                cfg["prot_blocks_params"],
                cfg["prot_windows_params"])):
            gb = EnhancedGATBlock(
                nh, prot_in, hof, ifConv1d=True, window_size=ws,
                dropout_1=cfg["dropout_1"], dropout_2=cfg["dropout_2"])
            sb = GraphSAGEBlock(
                prot_in, hof * nh, aggrs=["mean", "max", "sum"],
                ifConv1d=True, window_size=ws,
                dropout_1=cfg["dropout_1"], dropout_2=cfg["dropout_2"])
            self.add_module(f"prot_gat_block{i}", gb)
            self.add_module(f"prot_sage_block{i}", sb)
            self.prot_gat_blocks.append(gb)
            self.prot_sage_blocks.append(sb)
            prot_in = hof * nh
            prot_dims.append(prot_in)

        self.gat_importance = nn.Parameter(
            torch.FloatTensor([0.8] * len(self.prot_gat_blocks)))

        num_prot_layers = len(cfg["prot_blocks_params"])
        self.residue_importance = ResidueImportanceEvolution(
            hidden_dim=prot_dims[0],
            num_layers=num_prot_layers,
            decay=cfg.get("modulator_decay", 0.8),
            beta=cfg.get("modulator_beta", 0.3))

        self.prot_readouts = nn.ModuleList([
            SAGPooling(d, min_score=-1) for d in prot_dims])

        n_c = len(cfg["chem_blocks_params"])
        n_p = len(cfg["prot_blocks_params"])
        self.n_c = n_c
        self.n_p = n_p

        self.co_attention = CoAttentionLayer(
            graph_repr_dim, n_c, n_p, dropout=cfg["dropout_3"])
        self.rel_logits = nn.Parameter(torch.zeros(n_c, n_p))
        self.raw_temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, chem_fp, chem_graph, prot_graph,
                drug_seq_feat, prot_seq_feat):
        chem_graph.x = self.chem_initial_norm(chem_graph.x)
        prot_graph.x = self.prot_initial_norm(prot_graph.x)

        drug_h = self.drug_seq_proj(drug_seq_feat)
        prot_h = self.prot_seq_proj(prot_seq_feat)
        d_prior, p_prior = None, None
        for blk in self.transformer_blocks:
            drug_h, prot_h, d_attn, p_attn = blk(
                drug_h, prot_h, d_prior, p_prior)
            if d_prior is None:
                d_prior, p_prior = d_attn, p_attn
            else:
                d_prior = d_prior + d_attn
                p_prior = p_prior + p_attn

        seq_repr = self.seq_fusion(torch.cat(
            [drug_h.squeeze(1), prot_h.squeeze(1)], dim=-1))
        seq_repr = self.seq_out_proj(self.seq_out_norm(seq_repr))

        repr_chem = []
        cg = _deep_copy_pyg(chem_graph)
        for blk in self.chem_blocks:
            cg, emb, _ = blk(cg)
            repr_chem.append(emb)

        repr_prot = []
        prot_x = prot_graph.x
        prot_ei = prot_graph.edge_index
        prot_batch = prot_graph.batch
        prot_topo = prot_graph.topo_feat
        cumulative = None

        for i in range(len(self.prot_gat_blocks)):
            pg_gat = Data(
                x=prot_x.clone(), edge_index=prot_ei.clone(),
                batch=prot_batch.clone(), topo_feat=prot_topo.clone())
            pg_sage = Data(
                x=prot_x.clone(), edge_index=prot_ei.clone(),
                batch=prot_batch.clone(), topo_feat=prot_topo.clone())

            pg_gat, _, _ = self.prot_gat_blocks[i](pg_gat)
            pg_sage, _, _ = self.prot_sage_blocks[i](pg_sage)

            w = torch.sigmoid(self.gat_importance[i])
            fused_x = w * pg_gat.x + (1 - w) * pg_sage.x

            fused_x, cumulative = self.residue_importance(
                fused_x, i, cumulative)

            att_x, _, _, att_batch, _, _ = self.prot_readouts[i](
                fused_x, prot_ei, batch=prot_batch)
            repr_prot.append(global_add_pool(att_x, att_batch))
            prot_x = fused_x

        repr_chem = torch.stack(repr_chem, dim=1)
        repr_prot = torch.stack(repr_prot, dim=1)

        fusion = self.co_attention(repr_chem, repr_prot, seq_repr)

        T = F.softplus(self.raw_temperature) + 0.1
        weight = F.softmax(
            self.rel_logits.view(-1) / T, dim=0
        ).view(self.n_c, self.n_p)

        return (fusion * weight).sum(dim=(-1, -2))


class LitTopoFuseDTA(pl.LightningModule):
    def __init__(self, **cfg):
        super().__init__()
        criterion = cfg.pop('criterion', nn.MSELoss())
        self.save_hyperparameters(cfg)
        cfg['criterion'] = criterion
        self.model = TopoFuseDTA(**cfg)
        self.criterion = criterion
        self.lr = cfg["lr"]
        self.batch_size = cfg["batch_size"]

    def forward(self, fp, lg, pg, ds, ps):
        return self.model(fp, lg, pg, ds, ps)

    def _step(self, batch, stage, **kw):
        y, fp, lg, pg, ds, ps = batch
        pred = self(fp, lg, pg, ds, ps)
        loss = self.criterion(y, pred)
        m = get_metrics_reg(
            y.detach().cpu().numpy(),
            pred.detach().cpu().numpy(), stage, **kw)
        m["loss" if stage == "train" else f"{stage}_loss"] = loss
        self.log_dict(
            m, prog_bar=(stage != "train"), batch_size=self.batch_size)
        return m

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid", with_rm2=True, with_ci=True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test", with_rm2=True, with_ci=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
