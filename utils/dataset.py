import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import torch.utils.data as Data
import numpy as np
class ContrastDataset(Dataset):
    
    def __init__(self, palm_dir, vein_dir, transform=None):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform
        
        self.id_dict = {}
        
        for f in os.listdir(palm_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                pid = f.split('_')[0]
                if pid not in self.id_dict:
                    self.id_dict[pid] = {'palm': [], 'vein': []}
                self.id_dict[pid]['palm'].append(f)
        
        for f in os.listdir(vein_dir):
            if f.lower().endswith('.jpg'):
                pid = f.split('_')[0]
                if pid not in self.id_dict:
                    self.id_dict[pid] = {'palm': [], 'vein': []}
                self.id_dict[pid]['vein'].append(f)
        
        self.all_ids = sorted(self.id_dict.keys())
        
        self.samples = []
        for pid in self.all_ids:
            for f in self.id_dict[pid]['palm']:
                self.samples.append(('palm', pid, f))
            for f in self.id_dict[pid]['vein']:
                self.samples.append(('vein', pid, f))
        
        print(f"数据集：{len(self.all_ids)}人，{len(self.samples)}样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        modality, person_id, anchor_file = self.samples[idx]
        anchor_path = os.path.join(self.palm_dir if modality == 'palm' else self.vein_dir, anchor_file)
        anchor_img = Image.open(anchor_path).convert('L')
        
        pos_candidates = []
        for f in self.id_dict[person_id]['palm']:
            if not (modality == 'palm' and f == anchor_file):
                pos_candidates.append(('palm', f))
        for f in self.id_dict[person_id]['vein']:
            if not (modality == 'vein' and f == anchor_file):
                pos_candidates.append(('vein', f))
        
        if pos_candidates:
            pos_mod, pos_file = random.choice(pos_candidates)
        else:
            pos_mod, pos_file = modality, anchor_file
        
        pos_path = os.path.join(self.palm_dir if pos_mod == 'palm' else self.vein_dir, pos_file)
        positive_img = Image.open(pos_path).convert('L')
        
        neg_id = random.choice([pid for pid in self.all_ids if pid != person_id])
        neg_files = self.id_dict[neg_id]['palm'] + self.id_dict[neg_id]['vein']
        neg_file = random.choice(neg_files)
        
        if neg_file in self.id_dict[neg_id]['palm']:
            neg_path = os.path.join(self.palm_dir, neg_file)
        else:
            neg_path = os.path.join(self.vein_dir, neg_file)
        negative_img = Image.open(neg_path).convert('L')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        label = int(person_id) - 1
        return anchor_img, positive_img, negative_img, label
    
class PairDataset(Data.Dataset):
    def __init__(self, palm_dir, vein_dir, transform=None, split='train'):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform
        
        # 获取所有人的ID（文件夹名）
        self.pids = sorted([d for d in os.listdir(palm_dir) if os.path.isdir(os.path.join(palm_dir, d))])
        self.pid_to_label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)
        
        # 收集所有图像对
        all_pairs = []
        for pid in self.pids:
            palm_path = os.path.join(palm_dir, pid)
            vein_path = os.path.join(vein_dir, pid)
            
            if not os.path.exists(vein_path):
                continue
                
            # 获取该人所有的图像文件
            palm_files = sorted([f for f in os.listdir(palm_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            vein_files = sorted([f for f in os.listdir(vein_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            # 按文件名配对
            for pf in palm_files:
                if pf in vein_files:  # 如果文件名完全相同
                    all_pairs.append((os.path.join(pid, pf), os.path.join(pid, pf), self.pid_to_label[pid]))
        
        # 对整个数据集进行划分
        split_idx = int(len(all_pairs) * 0.8)
        self.samples = all_pairs[:split_idx] if split == 'train' else all_pairs[split_idx:]
        
        print(f"{'训练' if split == 'train' else '验证'}集: {len(self.samples)}对图像，{self.num_classes}个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pf, vf, label = self.samples[idx]
        palm_img = Image.open(os.path.join(self.palm_dir, pf)).convert('L')
        vein_img = Image.open(os.path.join(self.vein_dir, vf)).convert('L')
        
        if self.transform:
            palm_img = self.transform(palm_img)
            vein_img = self.transform(vein_img)
            
        if not isinstance(palm_img, torch.Tensor):
            palm_img = torch.tensor(np.array(palm_img), dtype=torch.float32).unsqueeze(0) / 255.
        if not isinstance(vein_img, torch.Tensor):
            vein_img = torch.tensor(np.array(vein_img), dtype=torch.float32).unsqueeze(0) / 255.
            
        return palm_img, vein_img, torch.tensor(label, dtype=torch.long)
