import argparse

PRECOMPUTED_TOPO_DIM = 7
ONLINE_TOPO_DIM = 2
TOTAL_TOPO_INPUT_DIM = ONLINE_TOPO_DIM + PRECOMPUTED_TOPO_DIM


def get_default_config():
    return {
        "chem_in_features": 23,
        "prot_in_features": 41,
        "hidden_dim": 256,
        "chem_heads_out_feat_params": [32, 32, 32, 32, 32, 32],
        "chem_blocks_params": [8, 8, 8, 8, 8, 8],
        "dropout_1": 0.4,
        "dropout_2": 0.1,
        "dropout_3": 0.5,
        "prot_heads_out_feat_params": [32, 32, 32, 32],
        "prot_blocks_params": [8, 8, 8, 8],
        "prot_windows_params": [7, 7, 7, 7],
        "batch_size": 64,
        "lr": 1e-4,
        "dataset_name": "davis",
        "model_name": "TopoFuse-DTA",
        "num_transformer_layers": 6,
        "transformer_num_heads": 8,
        "transformer_mlp_ratio": 4,
        "layer_scale": 1e-4,
        "drug_seq_dim": 768,
        "prot_seq_dim": 1152,
        "modulator_decay": 0.8,
        "modulator_beta": 0.3,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TopoFuse-DTA")
    parser.add_argument("--dataset", type=str, default="davis",
                        choices=["davis", "kiba", "metz", "bindingdb"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()
