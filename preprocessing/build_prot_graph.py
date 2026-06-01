import os
import json
import pickle
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from torch_geometric.data import Data
from tqdm import tqdm

THREE_TO_ONE = {
    'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
    'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
    'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
    'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'}

SS_CLASSES = ['-', 'B', 'E', 'G', 'H', 'S']

AAPHY7 = None
BLOSUM62 = None


def _load_lookups(lookup_dir):
    global AAPHY7, BLOSUM62
    if AAPHY7 is None:
        AAPHY7 = json.load(
            open(os.path.join(lookup_dir, "aa_phy7.txt")))
    if BLOSUM62 is None:
        BLOSUM62 = json.load(
            open(os.path.join(lookup_dir, "BLOSUM62_dim23.txt")))


def _one_hot(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [x == s for s in allowable]


def pdb_to_graph(pdb_path, chain_id="A", contact_threshold=6.0,
                 dssp_bin="mkdssp"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]
    chain = model[chain_id]
    dssp = DSSP(model, pdb_path, dssp=dssp_bin)

    features = []
    positions = []

    for idx, residue in enumerate(chain):
        resname_3 = residue.get_resname()
        if resname_3 not in THREE_TO_ONE:
            continue
        aa = THREE_TO_ONE[resname_3]

        dssp_entry = dssp[idx]
        sasa = dssp_entry[3]
        phi = dssp_entry[4] / 180.0
        psi = dssp_entry[5] / 180.0
        ss = dssp_entry[2]

        feat = (
            [sasa, phi, psi]
            + _one_hot(ss, SS_CLASSES)
            + AAPHY7[aa]
            + BLOSUM62[aa]
        )
        features.append(feat)
        positions.append(list(residue['CA'].get_vector()))

    x = torch.tensor(features, dtype=torch.float32)
    pos = np.array(positions)

    dist = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(-1))
    edge_index = torch.tensor(
        np.array(np.where(dist <= contact_threshold)),
        dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def build_protein_graphs(data_dir, pdb_dir, chain_id="A",
                         contact_threshold=6.0, dssp_bin="mkdssp",
                         lookup_dir="."):
    _load_lookups(lookup_dir)
    proteins = json.load(
        open(os.path.join(data_dir, "proteins.txt")))

    graphs = []
    for pid in tqdm(proteins.keys(), desc="Building protein graphs"):
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        g = pdb_to_graph(
            pdb_path, chain_id=chain_id,
            contact_threshold=contact_threshold,
            dssp_bin=dssp_bin)
        graphs.append(g)

    out_path = os.path.join(data_dir, "protein_to_pyg.pkl")
    pickle.dump(graphs, open(out_path, "wb"))
    print(f"Saved {len(graphs)} protein graphs to {out_path}")
    return graphs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--chain_id", type=str, default="A")
    parser.add_argument("--contact_threshold", type=float, default=6.0)
    parser.add_argument("--dssp_bin", type=str, default="mkdssp")
    parser.add_argument("--lookup_dir", type=str, default=".")
    args = parser.parse_args()
    build_protein_graphs(
        args.data_dir, args.pdb_dir,
        chain_id=args.chain_id,
        contact_threshold=args.contact_threshold,
        dssp_bin=args.dssp_bin,
        lookup_dir=args.lookup_dir)
