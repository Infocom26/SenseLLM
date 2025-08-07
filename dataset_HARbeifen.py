import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os
from glob import glob
from collections import defaultdict

class MultiActionMatDataset(Dataset):
    def __init__(self, root_dir, indices=None, data_key='cfr_array', is_train=True, split_ratio=0.8, random_seed=42):
        self.groups = defaultdict(list)
        # 只加载user1目录下的mat文件
        pattern = os.path.join(root_dir, 'user2', '*.mat')
        mat_files = glob(pattern, recursive=True)
        
        # 只处理活动类型a=1,2,3,4的文件
        valid_activity_types = {1, 2, 3, 4}
        
        for f in mat_files:
            filename = os.path.basename(f)
            # 解析文件名格式：id-a-b-c-d-Rx
            parts = filename.split('-')
            if len(parts) >= 2:
                try:
                    activity_type = int(parts[1])  # 活动类型a
                    if activity_type in valid_activity_types:
                        # 前缀为去掉最后一个-及其后的部分
                        prefix = '-'.join(parts[:-1])
                        self.groups[prefix].append(f)
                except ValueError:
                    continue
        
        # 只保留r1~r6齐全的组
        self.samples = []
        for prefix, files in self.groups.items():
            if len(files) == 6:  # r1~r6
                # label 取第一个-后的数字（活动类型）
                parts = prefix.split('-')
                if len(parts) < 2:
                    continue
                try:
                    activity_type = int(parts[1])
                    if activity_type in valid_activity_types:
                        # 将活动类型1,2,3,4映射为0,1,2,3
                        label = activity_type - 1
                        self.samples.append((prefix, sorted(files), label))
                except ValueError:
                    continue
        
        # indices分割
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
        
        print(f"Loaded {len(self.samples)} grouped samples for 4-class classification (activities 1,2,3,4).")
        print(f"Label mapping: 0->Activity1, 1->Activity2, 2->Activity3, 3->Activity4")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, files, label = self.samples[idx]
        datas = []
        FIXED_LEN = 2500
        for f in sorted(files):  # 保证r1~r6顺序
            mat = scipy.io.loadmat(f)
            data = mat['cfr_array']
            if data.shape[0] > FIXED_LEN:
                data = data[:FIXED_LEN, :]
            elif data.shape[0] < FIXED_LEN:
                pad = np.zeros((FIXED_LEN - data.shape[0], data.shape[1]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
            datas.append(data)
        datas = np.stack(datas, axis=0)  # [6, T', 90]
        # 拼接实部和虚部到最后一维
        datas_real = np.real(datas)
        datas_imag = np.imag(datas)
        datas_cat = np.concatenate([datas_real, datas_imag], axis=-1)  # [6, T', 180]
        return torch.from_numpy(datas_cat).float(), label

    @staticmethod
    def get_min_length(root_dir, data_key='cfr_array'):
        min_len = None
        # 只扫描user1目录下的mat文件
        pattern = os.path.join(root_dir, 'user1', '*.mat')
        mat_files = glob(pattern, recursive=True)
        print(f"Scanning {len(mat_files)} files in user1 directory for min length...")
        for f in mat_files:
            try:
                mat = scipy.io.loadmat(f)
                data = mat[data_key]
                cur_len = data.shape[0]
                if (min_len is None) or (cur_len < min_len):
                    min_len = cur_len
            except Exception as e:
                print(f"Error loading {f}: {e}")
        print(f"Min length in dataset: {min_len}")
        return min_len