import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")

    # config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml configuration file",
    )

    # task
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["node_classification", "link_prediction"],
        help="override task type",
    )

    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="override dataset name",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="override dataset root",
    )

    # model
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="override model name",
    )

    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        default=None,
        help="override model hidden dimension",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="override dropout",
    )

    # optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=None,
        help="override weight decay",
    )

    # train
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
    )

    # runtime
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="random seeds",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--compile",
        choices=[
            "auto",
            "true",
            "false",
        ],
        default=None,
    )

    # experiment
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint path",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
