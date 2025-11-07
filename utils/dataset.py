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
    def __init__(self, palm_dir, vein_dir, transform=None, is_train=True, train_ratio=1):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform

        palm_files = [f for f in os.listdir(palm_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
        vein_files = [f for f in os.listdir(vein_dir) if f.lower().endswith(('.jpg', '.JPG'))]

        # 简单配对，文件名格式：pid_...确保配对
        pairs_by_pid = {}
        for pf in palm_files:
            parts = pf.split('_')
            if len(parts) < 3: continue
            pid = parts[0]
            suf = '_'.join(parts[1:])
            vf = next((v for v in vein_files if v.startswith(f"{pid}_") and v.endswith(suf)), None)
            if vf:
                pairs_by_pid.setdefault(pid, []).append((pf, vf))

        self.pids = sorted(pairs_by_pid)
        self.pid_to_label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)

        self.samples = []
        for pid in self.pids:
            pairs = pairs_by_pid[pid]
            split = int(len(pairs) * train_ratio)
            sub = pairs[:split] if is_train else pairs[split:]
            label = self.pid_to_label[pid]
            for pf, vf in sub:
                self.samples.append((pf, vf, label))

        # 验证集兜底
        if not is_train:
            have = set(lbl for _, _, lbl in self.samples)
            for pid in self.pids:
                lbl = self.pid_to_label[pid]
                if lbl not in have and pairs_by_pid[pid]:
                    pf, vf = pairs_by_pid[pid][-1]
                    self.samples.append((pf, vf, lbl))

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