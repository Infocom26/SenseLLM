import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
from models.GPT4FALL_LoRA import Model as FallModel
from models.GPT4Gait import Model as GaitModel
from models.GPT4HAR import Model as HARModel
from peft import PeftModel
import numpy as np
import shutil
from dataset import FallNoFallMatDataset
from dataset4Gait import MultiActionMatDataset as GaitDataset
from dataset_HARbeifen import MultiActionMatDataset as HARDataset
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
from torch.utils.data import Subset
import argparse

lr = 0.0001
epochs = 35
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TASK_CONFIGS = {
    1: {
        'name': 'Fall',
        'model_class': FallModel,
        'num_classes': 2,
        'save_path': "xxx",
        'root_dir': "xxx",
        'dataset_class': FallNoFallMatDataset,
        'gpt_type': 'gpt2-medium'
    },
    2: {
        'name': 'HAR',
        'model_class': HARModel,
        'num_classes': 4,
        'save_path': "xxx",
        'root_dir': "xxx",
        'dataset_class': HARDataset,
        'gpt_type': 'gpt2-large'
    },
    3: {
        'name': 'Gait',
        'model_class': GaitModel,
        'num_classes': 3,
        'save_path': "xxx",
        'root_dir': "xxx",
        'dataset_class': GaitDataset,
        'gpt_type': 'gpt2-medium'
    }
}

random_seed = 42

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_best_checkpoint(model, save_path):
    if isinstance(model, nn.DataParallel):
        model.module.gpt2.save_pretrained(save_path)
    else:
        model.gpt2.save_pretrained(save_path)

def train_fall_task():
    config = TASK_CONFIGS[1]
    
    all_samples = []
    fall_path = os.path.join(config['root_dir'], 'fall', '**', '*.mat')
    fall_files = glob(fall_path, recursive=True)
    for f in fall_files:
        all_samples.append((f, 1))
    nofall_path = os.path.join(config['root_dir'], 'nonfall', '**', '*.mat')
    nofall_files = glob(nofall_path, recursive=True)
    for f in nofall_files:
        all_samples.append((f, 0))

    np.random.seed(random_seed)
    idx = np.arange(len(all_samples))
    np.random.shuffle(idx)
    n_total = len(idx)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]

    n_train_80 = int(len(train_idx) * 0.8)
    np.random.seed(random_seed)
    train_idx_80 = np.random.choice(train_idx, n_train_80, replace=False)

    train_set = FallNoFallMatDataset(config['root_dir'], indices=train_idx_80)
    validate_set = FallNoFallMatDataset(config['root_dir'], indices=val_idx)

    model = config['model_class'](gpt_type=config['gpt_type'], num_classes=config['num_classes']).to(device)

    if os.path.exists(config['save_path']):
        model.gpt2 = PeftModel.from_pretrained(model.gpt2, config['save_path'], inference_mode=False)

    for name, param in model.gpt2.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    writer = SummaryWriter(log_dir=f'xxx')
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=False, pin_memory=True, drop_last=False)
    
    best_acc = train_single_task(model, training_data_loader, validate_data_loader, writer, config)
    writer.close()
    return best_acc

def train_har_task():
    config = TASK_CONFIGS[2]
    
    full_dataset = HARDataset(config['root_dir'], indices=None, is_train=True, split_ratio=1.0, random_seed=random_seed)
    num_classes = len(set([sample[2] for sample in full_dataset.samples]))

    def extract_prefix(file_path):
        base = os.path.basename(file_path)
        prefix = base.split('_')[0]
        return prefix

    grouped_indices = defaultdict(list)
    for idx, sample in enumerate(full_dataset.samples):
        prefix = extract_prefix(sample[0])
        grouped_indices[prefix].append(idx)

    all_groups = list(grouped_indices.keys())
    random.seed(random_seed)
    random.shuffle(all_groups)

    n_total_groups = len(all_groups)
    n_train_groups = int(n_total_groups * 0.8)
    n_val_groups = n_total_groups - n_train_groups

    train_groups = all_groups[:n_train_groups]
    val_groups = all_groups[n_train_groups:]

    train_idx = []
    val_idx = []

    for g in train_groups:
        train_idx.extend(grouped_indices[g])
    for g in val_groups:
        val_idx.extend(grouped_indices[g])

    train_set_full = HARDataset(config['root_dir'], indices=train_idx, is_train=True, split_ratio=1.0, random_seed=random_seed)
    train_sample_num = int(len(train_set_full) * 0.8)
    random.seed(random_seed)
    sampled_indices = random.sample(range(len(train_set_full)), train_sample_num)
    train_set = Subset(train_set_full, sampled_indices)
    validate_set = HARDataset(config['root_dir'], indices=val_idx, is_train=False, split_ratio=1.0, random_seed=random_seed)

    model = config['model_class'](gpt_type=config['gpt_type'], num_classes=config['num_classes']).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if os.path.exists(config['save_path']):
        if isinstance(model, nn.DataParallel):
            model.module.gpt2 = PeftModel.from_pretrained(model.module.gpt2, config['save_path'], inference_mode=False)
        else:
            model.gpt2 = PeftModel.from_pretrained(model.gpt2, config['save_path'], inference_mode=False)

    if isinstance(model, nn.DataParallel):
        gpt2_params = model.module.gpt2.named_parameters()
    else:
        gpt2_params = model.gpt2.named_parameters()
    for name, param in gpt2_params:
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    writer = SummaryWriter(log_dir=f'xxx')
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=False, pin_memory=True, drop_last=False)
    
    best_acc = train_single_task(model, training_data_loader, validate_data_loader, writer, config)
    writer.close()
    return best_acc

def train_gait_task():
    config = TASK_CONFIGS[3]
    
    users = ["user1", "user2", "user7"]
    train_set = GaitDataset(config['root_dir'], users=users, indices=None, is_train=True, split_ratio=1.0, random_seed=random_seed)
    validate_set = GaitDataset(config['root_dir'], users=users, indices=None, is_train=True, split_ratio=1.0, random_seed=random_seed)

    num_classes = len(set([sample[2] for sample in train_set.samples + validate_set.samples]))

    model = config['model_class'](gpt_type=config['gpt_type'], num_classes=config['num_classes']).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if os.path.exists(config['save_path']):
        if isinstance(model, nn.DataParallel):
            model.module.gpt2 = PeftModel.from_pretrained(model.module.gpt2, config['save_path'], inference_mode=False)
        else:
            model.gpt2 = PeftModel.from_pretrained(model.gpt2, config['save_path'], inference_mode=False)

    if isinstance(model, nn.DataParallel):
        gpt2_params = model.module.gpt2.named_parameters()
    else:
        gpt2_params = model.gpt2.named_parameters()
    for name, param in gpt2_params:
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    writer = SummaryWriter(log_dir=f'xxx')
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=False, pin_memory=True, drop_last=False)
    
    best_acc = train_single_task(model, training_data_loader, validate_data_loader, writer, config)
    writer.close()
    return best_acc

def train_single_task(model, training_data_loader, validate_data_loader, writer, config):
    best_loss = 100
    best_acc = 0
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    patience = 20
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        epoch_train_loss = []
        correct = 0
        total = 0
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            prev, pred_t = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
            optimizer.zero_grad()
            pred_m = model(prev)
            loss = criterion(pred_m, pred_t)
            epoch_train_loss.append(loss.item())

            _, predicted = torch.max(pred_m.data, 1)
            correct += (predicted == pred_t).sum().item()
            total += pred_t.size(0)
            loss.backward()
            optimizer.step()

        t_loss = np.nanmean(np.array(epoch_train_loss))
        acc = correct / total if total > 0 else 0
        print(f'Epoch: {epoch+1}/{epochs}')

        model.eval()
        epoch_val_loss = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                prev, pred_t = Variable(batch[0]).to(device), Variable(batch[1]).to(device)
                pred_m = model(prev)
                loss = criterion(pred_m, pred_t)
                epoch_val_loss.append(loss.item())
                _, predicted = torch.max(pred_m.data, 1)
                val_correct += (predicted == pred_t).sum().item()
                val_total += pred_t.size(0)
        
        v_loss = np.nanmean(np.array(epoch_val_loss))
        val_acc = val_correct / val_total if val_total > 0 else 0

        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = val_acc
            save_best_checkpoint(model, config['save_path'])

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        scheduler.step(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

    return best_acc

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Training Script')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True,
                        help='Task ID: 1=Fall, 2=HAR, 3=Gait')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.task == 1:
        best_acc = train_fall_task()
    elif args.task == 2:
        best_acc = train_har_task()
    elif args.task == 3:
        best_acc = train_gait_task()
    else:
        print("Invalid task ID. Please choose 1 (Fall), 2 (HAR), or 3 (Gait)")
        return
    
    print(f"Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main() 