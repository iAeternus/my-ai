# iris-logistic-regression

基于 PyTorch 的鸢尾花分类神经网络。4 维特征 -> 3 种鸢尾花类别。

## 模型

简单前馈网络：`Linear(4, 10)` -> `ReLU` -> `Linear(10, 3)`，使用 AdamW 优化器和交叉熵损失训练。测试准确率 **100%**。

## 快速开始

```bash
uv sync
uv run iris-logistic-regression --train   # 训练并保存模型
uv run iris-logistic-regression           # 加载模型进行单样本推理
```

## 功能特性

- **训练**：80/10/10 分层划分，StandardScaler 标准化，早停机制 (patience=30)
- **可视化**：训练完成后自动生成 loss、准确率、学习率、每轮耗时的四合一图表
- **设备**：自动检测 CUDA / CPU
- **日志**：控制台输出 + 写入 `train.log`

## 项目结构

```
iris_logistic_regression/
├── data/iris/          # UCI 鸢尾花数据集
├── experiments/        # 模型权重、训练曲线图、日志
└── src/
    ├── main.py         # CLI 入口
    ├── model.py        # IrisClassifier 网络定义
    ├── trainer.py      # 训练循环 / 评估 / 推理
    ├── dataset.py      # 数据加载与预处理
    ├── config.py       # 超参数配置
    ├── parser.py       # 命令行参数解析
    ├── plots.py        # 训练曲线绘制
    └── utils/          # 设备检测、早停、日志
```

## 命令行

```
usage: iris-logistic-regression [-h] [--train]

options:
  --train    训练模式；省略则为推理模式
```
