from gnn.config.loader import from_cli
from gnn.config.parser import parse_args


def main():
    args = parse_args()
    cfg = from_cli(args.config, overrides=vars(args))
    print(cfg)


if __name__ == "__main__":
    main()
