# my-ai

我的深度学习代码库，包含神经网络、GNN、知识图谱、多模态推荐等。

## 目录结构

```
my-ai/
├── packages/
│   ├── iris_logistic_regression/   # 鸢尾花分类
│   └── gnn/                        # 图神经网络
├── pyproject.toml                  # uv workspace 根配置
└── uv.lock
```

## 环境配置

需要 **Python ≥ 3.13** 和 **[uv](https://docs.astral.sh/uv/)**。

```bash
uv sync
```

## 子项目

| 项目 | 说明 | 状态 |
|---|---|---|
| [iris_logistic_regression](packages/iris_logistic_regression/) | 基于鸢尾花数据集的前馈神经网络 | 已完成 |
| gnn | 基于 PyTorch Geometric 的图神经网络 (Cora, CiteSeer, PubMed) | 开发中 |

## 开发

```bash
uv run ruff check .     # 代码检查
uv run mypy .           # 类型检查
uv run pytest           # 测试
```
