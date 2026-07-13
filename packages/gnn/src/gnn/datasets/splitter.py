from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data


def split_link_prediction_data(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    neg_sampling_ratio: float = 1.0,
) -> tuple[Data, Data, Data]:
    """将 Planetoid 单图转换为链接预测的 train/val/test 三分数据。

    每个返回的 Data 对象包含:
      - edge_index:        消息传递边
      - edge_label_index:  监督边 (2, E_sup)
      - edge_label:        正/负标签 (E_sup,)
    """

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    return transform(data)
