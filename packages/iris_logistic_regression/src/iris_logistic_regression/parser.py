import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="iris_logistic_regression",
        description="Iris 分类示例",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="训练模型并保存最佳权重；不指定时执行预测",
    )

    return parser.parse_args()
