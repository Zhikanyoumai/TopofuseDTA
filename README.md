# TopoFuse-DTA

**Dual-Channel Topology-Aware Graph Attention and Cross-Granularity Hierarchical Fusion for Drug–Target Affinity Prediction**

---

## Architecture

TopoFuse-DTA consists of three parallel encoding branches and a hierarchical fusion stage:

```
                        ┌──────────────────────────────────┐
                        │    A. Input & Preprocessing      │
                        │                                  │
                        │  Drug SMILES ──► Molecular Graph │
                        │  Drug SMILES ──► ChemBERTa-2     │
                        │  Protein PDB ──► Residue Graph   │
                        │  Protein Seq ──► ESM-C 600M      │
                        │  Both Graphs ──► 9-dim Topology   │
                        └──────────┬───────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   Drug Graph Encoder      CGPCA Module          Protein Graph Encoder
   (Ld GTAT layers)     (Ls Transformer          (Lp dual-path layers)
                          layers)
   g_d^1 ... g_d^Ld      H_dp^s                 g_p^1 ... g_p^Lp
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                    Cross-Granularity Co-Attention Fusion
                    (Ld × Lp interaction matrix + temp-scaled weights)
                                   │
                                   ▼
                            Predicted Affinity ŷ
```

**Drug Graph Encoder**: Stacks `Ld` GTAT layers with dual-channel topology-aware attention. Each layer computes independent semantic and topological attention coefficients, applied cross-gated: topological attention weights semantic features, and vice versa. Multi-scale attention heads capture patterns at different granularities. Each layer outputs a graph-level representation via pooling.

**Protein Graph Encoder**: Dual-path parallel architecture at each of `Lp` layers:
- **Global path**: GTAT with topology-aware attention
- **Local path**: Multi-aggregation GraphSAGE (mean + max + sum)
- Paths fused via learnable gating → residue importance evolution modulates fused features → graph-level readout

**CGPCA Module**: `Ls` stacked cross-attention layers between drug and protein sequence embeddings. Each layer gates queries by the counterpart modality's global context and accumulates attention priors from preceding layers, enabling coarse-to-fine alignment.

**Cross-Granularity Fusion**: Retains all layer-wise representations, constructs a full `Ld × Lp` interaction matrix via co-attention MLP, and aggregates through temperature-scaled learnable softmax weights.

---

## Project Structure

```
TopoFuse-DTA/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                           # Hyperparameters and CLI arguments
├── train.py                            # Training entry point
├── models/
│   ├── topofuse_dta.py                 # Main model and Lightning wrapper
│   ├── gtat_conv.py                    # Dual-channel topology-aware convolution
│   ├── blocks.py                       # EnhancedGATBlock, GraphSAGEBlock
│   ├── transformer.py                  # CGPCA TransformerBlock
│   ├── fusion.py                       # CoAttentionLayer, ResidueImportanceEvolution
│   └── layers.py                       # EnhancedSAGEConv
├── data/
│   └── dataset.py                      # DTADataset and DTADataModule
├── utils/
│   ├── metrics.py                      # Evaluation metrics (MSE, CI, r²_m, etc.)
│   └── results.py                      # Result saving and visualization
└── preprocessing/
    ├── build_mol_graph.py              # Drug molecular graph construction
    ├── build_prot_graph.py             # Protein residue graph construction
    └── build_topology.py              # 7-dim precomputed topology descriptors
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.13
- PyTorch Geometric ≥ 2.3
- PyTorch Lightning ≥ 2.0
- RDKit (for molecular graph construction)
- Biopython + DSSP (for protein graph construction)

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric should be installed following their official guides to match your CUDA version:
- https://pytorch.org/get-started
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Feature Description

### Drug Atom Features (23-dim)

| Feature | Dim |
|---------|-----|
| One-hot atom type (C, N, O, F, S, Cl, Br, P, I) | 9 |
| Atom mass (min–max scaled) | 1 |
| Number of directly bonded neighbors | 1 |
| Total number of bonded hydrogens | 1 |
| One-hot hybridization (sp2, sp3) | 2 |
| Is in ring (binary) | 1 |
| Is aromatic (binary) | 1 |
| Is hydrophobic (binary) | 1 |
| Is metal (binary) | 1 |
| Is halogen (binary) | 1 |
| Is hydrogen-bond donor (binary) | 1 |
| Is hydrogen-bond acceptor (binary) | 1 |
| Is negatively charged (binary) | 1 |
| Is positively charged (binary) | 1 |

### Protein Residue Features (41-dim)

| Feature | Dim |
|---------|-----|
| Solvent-accessible surface area (z-score scaled) | 1 |
| Phi angle (÷ 180°) | 1 |
| Psi angle (÷ 180°) | 1 |
| One-hot secondary structure (6 classes) | 6 |
| AAPHY7 physicochemical descriptors | 7 |
| BLOSUM62 substitution matrix descriptors | 23 |
| Phosphorylated (binary) | 1 |
| Mutated (binary) | 1 |

### Protein Edge Features (10-dim, defines graph connectivity)

| Feature | Dim |
|---------|-----|
| Covalent bond | 1 |
| Hydrophobic contact | 1 |
| Hydrogen bond (donor → acceptor) | 1 |
| Hydrogen bond (acceptor → donor) | 1 |
| Salt bridge (cation → anion) | 1 |
| Salt bridge (anion → cation) | 1 |
| Pi-cation (ring → cation) | 1 |
| Pi-cation (cation → ring) | 1 |
| Parallel pi-stacking | 1 |
| Perpendicular pi-stacking | 1 |

### Topology Descriptors (9-dim per node, both drug and protein)

| Feature | Source |
|---------|--------|
| Node degree | Computed online |
| Average neighbor degree | Computed online |
| Clustering coefficient | Precomputed |
| Betweenness centrality | Precomputed |
| Closeness centrality | Precomputed |
| Eigenvector centrality | Precomputed |
| PageRank | Precomputed |
| Average shortest path length | Precomputed |
| 2-hop neighborhood size | Precomputed |

All precomputed features are min–max normalized within each graph instance.

### Sequence Embeddings

| Modality | Backbone | Output Dim |
|----------|----------|------------|
| Drug | ChemBERTa-2 (77M) | 768 |
| Protein | ESM-C 600M | 1152 |

---

## Data Preparation

### Directory Layout

```
data/
└── davis/
    ├── DTAs.pkl                            # [(ligand_idx, protein_idx, affinity), ...]
    ├── ligands_can.txt                     # {"CID": "SMILES", ...}
    ├── proteins.txt                        # {"ProteinID": "sequence", ...}
    ├── ligand_to_pyg.pkl                   # [PyG Data, ...] for ligands
    ├── protein_to_pyg.pkl                  # [PyG Data, ...] for proteins
    ├── ligand_to_fp.pkl                    # [fingerprint_array, ...]
    ├── drug_seq_features.pkl               # {"CID": ndarray(768,), ...}
    ├── protein_seq_features.pkl            # {"ProteinID": ndarray(1152,), ...}
    ├── ligand_topology_features.pkl        # {idx: Tensor(N, 7), ...}
    ├── protein_topology_features.pkl       # {idx: Tensor(N, 7), ...}
    └── folds/
        ├── train_fold_setting1.txt
        └── test_fold_setting1.txt
```

### Preprocessing Pipeline

Run the following steps in order:

```bash
# Step 1: Build molecular graphs from SMILES (produces ligand_to_pyg.pkl)
python preprocessing/build_mol_graph.py --data_dir data/davis

# Step 2: Build protein residue graphs from PDB files (produces protein_to_pyg.pkl)
python preprocessing/build_prot_graph.py \
    --data_dir data/davis \
    --pdb_dir data/davis/pdbs \
    --dssp_bin mkdssp \
    --lookup_dir preprocessing

# Step 3: Compute topology descriptors (produces *_topology_features.pkl)
python preprocessing/build_topology.py --data_dir data/davis

# Step 4: Extract sequence embeddings (user-provided scripts)
#   - ChemBERTa-2 → drug_seq_features.pkl   (768-dim per drug)
#   - ESM-C 600M  → protein_seq_features.pkl (1152-dim per protein)
```

---

## Training

```bash
python train.py \
    --dataset davis \
    --data_dir ./data \
    --batch_size 64 \
    --lr 1e-4 \
    --max_epochs 2000 \
    --gpu 0 \
    --log_dir ./logs
```

### Hyperparameter Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 64 |
| Loss function | MSE |
| Drug graph encoder depth (Ld) | 6 |
| Protein graph encoder depth (Lp) | 4 |
| CGPCA Transformer layers (Ls) | 6 |
| Hidden dimension | 256 |
| Attention heads (GTAT) | 8 |
| Head output features | 32 |
| Protein Conv1d window sizes | [7, 7, 7, 7] |
| Transformer attention heads | 8 |
| Transformer FFN ratio | 4 |
| Layer scale init (γ) | 1 × 10⁻⁴ |
| Dropout (GTAT attention) | 0.4 |
| Dropout (block output) | 0.1 |
| Dropout (fusion MLP) | 0.5 |
| Residue modulator decay (λ) | 0.8 |
| Residue modulator strength (β) | 0.3 |
| Drug sequence backbone | ChemBERTa-2 (768-dim) |
| Protein sequence backbone | ESM-C 600M (1152-dim) |

---

## Results

Performance comparison on four benchmark datasets (best results in **bold**):

### Davis & KIBA

| Method | Davis MSE ↓ | Davis CI ↑ | Davis r²_m ↑ | KIBA MSE ↓ | KIBA CI ↑ | KIBA r²_m ↑ |
|--------|------------|-----------|-------------|-----------|----------|------------|
| GraphDTA | 0.245 | 0.881 | 0.675 | 0.139 | 0.881 | 0.772 |
| MgraphDTA | 0.207 | 0.900 | 0.705 | 0.130 | 0.906 | 0.793 |
| FusionDTA | 0.208 | 0.913 | 0.743 | 0.128 | 0.902 | 0.801 |
| TDGraphDTA | 0.201 | 0.906 | 0.742 | 0.124 | 0.899 | 0.808 |
| MSN-DTA | 0.194 | 0.918 | 0.752 | 0.123 | 0.910 | 0.809 |
| Adaptive-DTA | 0.170 | 0.913 | 0.689 | 0.126 | 0.905 | 0.774 |
| RRGDTA | 0.196 | 0.909 | 0.749 | 0.122 | 0.905 | 0.810 |
| **TopoFuse-DTA** | **0.157** | **0.921** | **0.784** | **0.121** | **0.912** | **0.811** |

### Metz & BindingDB

| Method | Metz MSE ↓ | Metz CI ↑ | Metz r²_m ↑ | BindingDB MSE ↓ | BindingDB CI ↑ | BindingDB r²_m ↑ |
|--------|-----------|----------|------------|----------------|---------------|-----------------|
| KANPM-DTA | 0.237 | 0.838 | 0.723 | — | — | — |
| MGraphDTA | — | — | — | 0.261 | 0.896 | 0.738 |
| **TopoFuse-DTA** | **0.221** | **0.842** | **0.734** | **0.247** | **0.902** | **0.743** |

---

## Citation

```bibtex
@article{topofuse-dta-2025,
  title={TopoFuse-DTA: Dual-Channel Topology-Aware Graph Attention and
         Cross-Granularity Hierarchical Fusion for Drug--Target Affinity Prediction},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
