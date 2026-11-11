import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="akt_dro", choices=["akt_dro"])
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)

    # AKT backbone
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # DRO-specific parameters
    parser.add_argument("--dro_gamma", type=float, default=2.0)
    parser.add_argument("--dro_lambda", type=float, default=0.7)
    parser.add_argument("--dro_reward_alpha", type=float, default=0.5)
    parser.add_argument("--dro_cluster_aware", type=int, default=1)
    parser.add_argument("--dro_num_clusters", type=int, default=4)
    parser.add_argument("--dro_pair_temp", type=float, default=0.2)
    parser.add_argument("--dro_margin", type=float, default=0.0)
    parser.add_argument("--dro_max_pairs", type=int, default=4096)
    parser.add_argument("--dro_loss_weight", type=float, default=0.2)
    parser.add_argument("--dro_bad_th", type=float, default=0.4)
    parser.add_argument("--dro_good_th", type=float, default=0.7)
    parser.add_argument("--dro_chaotic_th", type=float, default=0.35)
    parser.add_argument("--dro_entropy_weight", type=float, default=1e-3)
    parser.add_argument("--dro_value_weight", type=float, default=5e-2)

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)

    args = parser.parse_args()

    params = vars(args)
    main(params)
