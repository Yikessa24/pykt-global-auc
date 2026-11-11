"""Train AKT with noisy-group robust AUC and fixed history-based IRT tertiles."""

import argparse

from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="assist2015")
    parser.add_argument("--model_name", default="akt_noisyauc", choices=["akt_noisyauc"])
    parser.add_argument("--emb_type", default="qid")
    parser.add_argument("--save_dir", default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--noisy_gamma", type=float, nargs="+", required=True,
                        help="One TV radius or nine radii, positive group first and negative group second")
    parser.add_argument("--noisy_pair_cap", type=int, default=32)
    parser.add_argument("--noisy_sam_radius", type=float, default=0.0005)
    parser.add_argument("--noisy_dual_lr", type=float, default=0.01)
    parser.add_argument("--noisy_dual_limit", type=float, default=1.0)
    parser.add_argument("--minimax_prefix_length", type=int, default=15)
    parser.add_argument("--minimax_item_col", choices=["concepts", "questions"], default="concepts")
    parser.add_argument("--init_checkpoint", default="", help="Optional AKT checkpoint from the same training split")
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS, help="Nominal number of stratified targets per update")
    parser.add_argument("--num_epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--add_uuid", type=int, choices=[0, 1], default=1)
    main(vars(parser.parse_args()))
