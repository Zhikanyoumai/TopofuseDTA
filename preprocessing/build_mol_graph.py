import os
import json
import pickle
import warnings
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore")

ACCEPTOR = Chem.MolFromSmarts(
    '[$([O;H1;v2]),'
    '$([O;H0;v2;!$(O=N-*),'
    '$([O;-;!$(*-N=O)]),'
    '$([o;+0])]),'
    '$([n;+0;!X3;!$([n;H1](cc)cc),'
    '$([$([N;H0]#[C&v4])]),'
    '$([N&v3;H0;$(Nc)])]),'
    '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')

DONOR = Chem.MolFromSmarts(
    '[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
    '$([$(n[n;H1]),'
    '$(nc[n;H1])])]),'
    '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
    '$([O,S;H1;+0])]')

BASIC = Chem.MolFromSmarts(
    '[$([N;H2&+0][$([C,a]);!$([C,a](=O))]),'
    '$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),'
    '$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),'
    '$([N,n;X2;+0])]')

ACIDIC = Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]')

METAL_ANUMS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103}

HALOGEN_ANUMS = {9, 17, 35, 53}

ATOM_SYMBOLS = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I']
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3]


def _one_hot(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [x == s for s in allowable]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    donor_idx = set(np.array(
        mol.GetSubstructMatches(DONOR, maxMatches=100000)).flatten())
    acceptor_idx = set(np.array(
        mol.GetSubstructMatches(ACCEPTOR, maxMatches=100000)).flatten())
    basic_idx = set(np.array(
        mol.GetSubstructMatches(BASIC, maxMatches=100000)).flatten())
    acidic_idx = set(np.array(
        mol.GetSubstructMatches(ACIDIC, maxMatches=100000)).flatten())

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        anum = atom.GetAtomicNum()
        edges.append([idx, idx])

        neighbor_anums = [a.GetAtomicNum() for a in atom.GetNeighbors()]
        feat = (
            _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
            + [atom.GetMass(), atom.GetDegree(), atom.GetTotalNumHs()]
            + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            + [atom.IsInRing(), atom.GetIsAromatic()]
            + [anum == 6 and all(a in (6, 1, 0) for a in neighbor_anums),
               anum in METAL_ANUMS,
               anum in HALOGEN_ANUMS,
               idx in donor_idx and anum != 6,
               idx in acceptor_idx and anum != 6]
            + [int((idx in acidic_idx and anum != 6)
                   or atom.GetFormalCharge() < 0)]
            + [int((idx in basic_idx and anum != 6)
                   or atom.GetFormalCharge() > 0)]
        )
        features.append(feat)

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def build_ligand_graphs(data_dir):
    ligands = json.load(open(os.path.join(data_dir, "ligands_can.txt")))
    graphs = []
    for cid, smi in tqdm(ligands.items(), desc="Building ligand graphs"):
        g = smiles_to_graph(smi)
        if g is None:
            raise ValueError(f"Failed to parse SMILES for {cid}: {smi}")
        graphs.append(g)

    out_path = os.path.join(data_dir, "ligand_to_pyg.pkl")
    pickle.dump(graphs, open(out_path, "wb"))
    print(f"Saved {len(graphs)} ligand graphs to {out_path}")
    return graphs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    build_ligand_graphs(args.data_dir)
