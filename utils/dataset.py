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
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                pid = f.split('_')[0]
                if pid not in self.id_dict:
                    self.id_dict[pid] = {'palm': [], 'vein': []}
                self.id_dict[pid]['vein'].append(f)
        
        self.all_ids = sorted(self.id_dict.keys())
        # 映射 pid -> label （可靠，不依赖文件名能否转换为 int）
        self.pid_to_label = {pid: i for i, pid in enumerate(self.all_ids)}

        self.samples = []
        for pid in self.all_ids:
            for f in self.id_dict[pid]['palm']:
                self.samples.append(('palm', pid, f))
            for f in self.id_dict[pid]['vein']:
                self.samples.append(('vein', pid, f))
        
        print(f"dataset：{len(self.all_ids)}persons，{len(self.samples)}samples")
    
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
        
        # 随机选择一个有样本的不同 person id 作为负样本
        neg_candidates = [pid for pid in self.all_ids if pid != person_id and (self.id_dict[pid]['palm'] + self.id_dict[pid]['vein'])]
        if not neg_candidates:
            raise RuntimeError('No negative candidates found in dataset')
        neg_id = random.choice(neg_candidates)
        neg_files = self.id_dict[neg_id]['palm'] + self.id_dict[neg_id]['vein']
        neg_file = random.choice(neg_files)
        
        if neg_file in self.id_dict[neg_id]['palm']:
            neg_path = os.path.join(self.palm_dir, neg_file)
        else:
            neg_path = os.path.join(self.vein_dir, neg_file)
        negative_img = Image.open(neg_path).convert('L')
        
        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            except Exception as e:
                raise RuntimeError(f"Error applying transform: {e}")

        # 使用 pid 到 label 的映射，避免 pid 不是数字时报错
        label = self.pid_to_label.get(person_id, 0)
        return anchor_img, positive_img, negative_img, label
    
class PairDataset(Data.Dataset):
    def __init__(self, palm_dir, vein_dir, transform=None, split='train'):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform
        

        self.pids = sorted([d for d in os.listdir(palm_dir) if os.path.isdir(os.path.join(palm_dir, d))])
        self.pid_to_label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)
        

        all_pairs = []
        for pid in self.pids:
            palm_path = os.path.join(palm_dir, pid)
            vein_path = os.path.join(vein_dir, pid)
            
            if not os.path.exists(vein_path):
                continue
                

            palm_files = sorted([f for f in os.listdir(palm_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            vein_files = sorted([f for f in os.listdir(vein_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            

            for pf in palm_files:
                if pf in vein_files:  
                    all_pairs.append((os.path.join(pid, pf), os.path.join(pid, pf), self.pid_to_label[pid]))
        

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
