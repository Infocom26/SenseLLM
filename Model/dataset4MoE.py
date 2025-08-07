import torch
from torch.utils.data import Dataset
import numpy as np

from dataset import FallNoFallMatDataset
from dataset4Gait import MultiActionMatDataset as GaitDataset
from dataset_HARbeifen import MultiActionMatDataset as HARDataset

class MoEMultiTaskDataset(Dataset):
    """
    多任务MoE数据集，自动根据任务类型分别加载Fall、Gait、HAR数据，
    并返回统一格式：(data, label, task_type)
    task_type: 0=Fall, 1=Gait, 2=HAR
    """
    def __init__(self, fall_root, gait_root, har_root, 
                 fall_kwargs=None, gait_kwargs=None, har_kwargs=None):
        super().__init__()
        self.datasets = []
        self.task_types = []
        self.lengths = []
        # Fall
        fall_kwargs = fall_kwargs or {}
        self.datasets.append(FallNoFallMatDataset(fall_root, **fall_kwargs))
        self.task_types.append(0)
        self.lengths.append(len(self.datasets[-1]))
        # Gait
        gait_kwargs = gait_kwargs or {}
        self.datasets.append(GaitDataset(gait_root, **gait_kwargs))
        self.task_types.append(1)
        self.lengths.append(len(self.datasets[-1]))
        # HAR
        har_kwargs = har_kwargs or {}
        self.datasets.append(HARDataset(har_root, **har_kwargs))
        self.task_types.append(2)
        self.lengths.append(len(self.datasets[-1]))
        # 记录每个任务的起始索引
        self.cum_lengths = np.cumsum([0] + self.lengths)
        print(f"MoEMultiTaskDataset loaded: Fall={self.lengths[0]}, Gait={self.lengths[1]}, HAR={self.lengths[2]}")

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # 判断属于哪个子数据集
        for i in range(3):
            if self.cum_lengths[i] <= idx < self.cum_lengths[i+1]:
                local_idx = idx - self.cum_lengths[i]
                data, label = self.datasets[i][local_idx]
                task_type = self.task_types[i]
                return data, label, task_type
        raise IndexError(f"Index {idx} out of range for MoEMultiTaskDataset") 