import argparse
import json
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    param.update(args)
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Train HASTEN for class-incremental learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="exps/hasten/cifar_0_10.json",
        help="Json file of settings.",
    )
    parser.add_argument("--sample_num", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="hasten")
    parser.add_argument("--seed", type=int, default=1993, help="Random seed.")
    parser.add_argument("--hierarchy_weight", type=float, default=0.5)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    parser.add_argument("--entail_weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--mapping_mode", type=str, default="mlp", choices=["fix", "mlp"]
    )
    parser.add_argument(
        "--project_grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable shared-mapper gradient projection (default: enabled).",
    )
    parser.add_argument("--curvature", type=float, default=1)
    parser.add_argument("--kcone", type=float, default=0.1)
    parser.add_argument(
        "--nullspace_energy_threshold",
        type=float,
        default=0.95,
        help="Energy threshold used when computing the null-space for gradient projection.",
    )
    return parser


if __name__ == "__main__":
    main()
