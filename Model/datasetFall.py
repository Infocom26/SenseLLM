import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os
from glob import glob

class FallNoFallMatDataset(Dataset):
    def __init__(self, root_dir, indices=None, data_key='csi_data', label_key='activity', fall_dir='fall', nofall_dir='nonfall', is_train=True, split_ratio=0.8, random_seed=42):
        self.samples = []
        # 递归查找fall目录下所有mat文件
        fall_path = os.path.join(root_dir, fall_dir, '**', '*.mat')
        fall_files = glob(fall_path, recursive=True)
        print(f"Found {len(fall_files)} fall .mat files in {fall_path}")
        for f in fall_files:
            self.samples.append((f, 1))  # fall标签为1
        # 递归查找nofall目录下所有mat文件
        nofall_path = os.path.join(root_dir, nofall_dir, '**', '*.mat')
        nofall_files = glob(nofall_path, recursive=True)
        print(f"Found {len(nofall_files)} nofall .mat files in {nofall_path}")
        for f in nofall_files:
            self.samples.append((f, 0))  # nofall标签为0

        self.data_key = data_key
        self.label_key = label_key

        # 新增：如果传入indices，则只保留指定索引的样本
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        else:
            np.random.seed(random_seed)
            idx = np.arange(len(self.samples))
            np.random.shuffle(idx)
            split = int(len(idx) * split_ratio)
            if is_train:
                selected_idx = idx[:split]
            else:
                selected_idx = idx[split:]
            self.samples = [self.samples[i] for i in selected_idx]

        self.all_data = []
        self.all_labels = []
        # 预加载所有数据，每个mat文件为一个样本
        for file_path, label in self.samples:
            try:
              mat = scipy.io.loadmat(file_path)
              data = mat[self.data_key]  # [2000, 30, 3]，保持原始复数类型
              self.all_data.append(data)
              self.all_labels.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        self.all_data = np.array(self.all_data)  # [num_files, 2000, 30, 3]
        self.all_labels = np.array(self.all_labels)  # [num_files]
        print(f"Loaded {self.all_data.shape[0]} files. indices={indices is not None}")
        print("Each data shape:", self.all_data[0].shape if len(self.all_data) > 0 else "N/A")
        print("Labels shape:", self.all_labels.shape)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # 返回复数类型Tensor，shape为[2000, 30, 3]
        return torch.from_numpy(self.all_data[idx]).to(torch.complex64), self.all_labels[idx]