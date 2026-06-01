import os
import pickle
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm


def _pyg_to_nx(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    edges = data.edge_index.t().numpy()
    for i, j in edges:
        if i != j:
            G.add_edge(int(i), int(j))
    return G


def compute_node_topology(G, num_nodes):
    feats = np.zeros((num_nodes, 7), dtype=np.float32)
    if num_nodes == 0:
        return feats

    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    pagerank = nx.pagerank(G, max_iter=1000)

    try:
        avg_sp = dict(nx.all_pairs_shortest_path_length(G))
    except Exception:
        avg_sp = {}

    for node in range(num_nodes):
        if node not in G:
            continue
        feats[node, 0] = clustering.get(node, 0.0)
        feats[node, 1] = betweenness.get(node, 0.0)
        feats[node, 2] = closeness.get(node, 0.0)
        feats[node, 3] = eigenvector.get(node, 0.0)
        feats[node, 4] = pagerank.get(node, 0.0)

        if node in avg_sp:
            lengths = list(avg_sp[node].values())
            feats[node, 5] = np.mean(lengths) if lengths else 0.0
        feats[node, 6] = len(set(
            nx.single_source_shortest_path_length(G, node, cutoff=2)))

    for col in range(7):
        col_min = feats[:, col].min()
        col_max = feats[:, col].max()
        if col_max - col_min > 1e-8:
            feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)

    return feats


def build_topology_features(data_dir, graph_file, out_file):
    graphs = pickle.load(
        open(os.path.join(data_dir, graph_file), "rb"))

    topo_dict = {}
    for idx, g in enumerate(tqdm(graphs, desc=f"Computing {out_file}")):
        G = _pyg_to_nx(g)
        feats = compute_node_topology(G, g.x.shape[0])
        topo_dict[idx] = torch.tensor(feats, dtype=torch.float32)

    out_path = os.path.join(data_dir, out_file)
    pickle.dump(topo_dict, open(out_path, "wb"))
    print(f"Saved topology features to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    build_topology_features(
        args.data_dir,
        "ligand_to_pyg.pkl",
        "ligand_topology_features.pkl")

    build_topology_features(
        args.data_dir,
        "protein_to_pyg.pkl",
        "protein_topology_features.pkl")
