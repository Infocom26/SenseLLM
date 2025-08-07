import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os
from glob import glob
from collections import defaultdict

class MultiActionMatDataset(Dataset):
    def __init__(self, root_dir, users=None, indices=None, data_key='cfr_array', is_train=True, split_ratio=0.8, random_seed=42):
        self.groups = defaultdict(list)
        # 支持加载多个user目录
        if users is None:
            users = ['user1', 'user2', 'user3', 'user4', 'user7']
        mat_files = []
        for user in users:
            if user == 'user3':
                # user3的文件直接在user3目录下
                pattern = os.path.join(root_dir, user, '*.mat')
            else:
                # 其他用户的文件在userX/userX子目录中
                pattern = os.path.join(root_dir, user, user, '*.mat')
            mat_files.extend(glob(pattern, recursive=True))
        # 以id-a-b为分组前缀，收集r1~r6
        for f in mat_files:
            filename = os.path.basename(f)
            # 解析文件名格式：id-a-b-c-d-Rx
            parts = filename.split('-')
            if len(parts) >= 3:
                prefix = '-'.join(parts[:3])  # id-a-b
                self.groups[prefix].append(f)
        # 只保留r1~r6齐全的组
        self.samples = []
        user_label_map = {'user1': 0, 'user2': 1, 'user7': 2}
        for prefix, files in self.groups.items():
            # 只保留r1~r6齐全的分组
            if len(files) == 6:
                parts = prefix.split('-')
                if len(parts) < 1:
                    continue
                user_id = parts[0]
                if user_id in user_label_map:
                    label = user_label_map[user_id]
                    self.samples.append((prefix, sorted(files), label))
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
        print(f"Loaded {len(self.samples)} grouped samples for gait (user) classification.")
        print(f"Label mapping: {user_label_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, files, label = self.samples[idx]
        datas = []
        FIXED_LEN = 3000
        FIXED_GROUP_LEN = 6
        files = sorted(files)
        if len(files) < FIXED_GROUP_LEN:
            files = files + [files[0]] * (FIXED_GROUP_LEN - len(files))
        elif len(files) > FIXED_GROUP_LEN:
            files = files[:FIXED_GROUP_LEN]
        for f in files:  # 保证r1~r6顺序
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