from net import (Restormer_Encoder, DeformableAlignment, 
                 BaseFeatureExtraction, DetailFeatureExtraction, ArcFaceClassifier)
import os
import numpy as np
from utils.metrics import compute_eer, roc_auc, tar_at_far
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import PairDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    palm_dir = '/root/autodl-tmp/CDDFuse/MMIF-CDDFuse-main/roi/CASIA/vi/roi'
    vein_dir = '/root/autodl-tmp/CDDFuse/MMIF-CDDFuse-main/roi/CASIA/ir/roi' 
    ckpt_path = 'MMIF-CDDFuse-main/models/full_model_phase2_best.pth' 
    num_classes = 100
    dim = 64
    num_heads = 8

config = Config()

def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def forward_model(encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier, labels=None):
    base_p, detail_p, _ = encoder(palm)
    base_v, detail_v, _ = encoder(vein)
    # 对齐
    ab_p = alignment(base_v, base_p)
    ab_v = alignment(base_p, base_v)
    ad_p = alignment(detail_v, detail_p)
    ad_v = alignment(detail_p, detail_v)
    # 融合
    fused_base = base_fuse(ab_p + ab_v)
    fused_detail = detail_fuse(ad_p + ad_v)
    # 分类
    fused = torch.cat([fused_base, fused_detail], dim=1)
    logits, feature_vector = classifier(fused, labels)
    return logits, feature_vector

def extract_features(encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier):
    with torch.no_grad():
        logits, feature_vector = forward_model(encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier)
    return feature_vector.cpu().numpy()

model_name = "CDDFuse    "
device = config.device

encoder = nn.DataParallel(Restormer_Encoder(
    inp_channels=1, dim=config.dim, num_blocks=[4, 4],
    heads=[config.num_heads]*3
    )).to(device)

alignment = nn.DataParallel(DeformableAlignment(config.dim)).to(device)
base_fuse = nn.DataParallel(BaseFeatureExtraction(config.dim, config.num_heads)).to(device)
detail_fuse = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
classifier = nn.DataParallel(ArcFaceClassifier(config.dim*2, config.num_classes)).to(device)

# 加载权重（兼容是否使用 DataParallel 与不同保存格式）
if os.path.exists(config.ckpt_path):
    ckpt = torch.load(config.ckpt_path, map_location=device)

    def pick_state(d, primary_key, alt_keys=None):
        if isinstance(d, dict) and primary_key in d:
            return d[primary_key]
        if isinstance(alt_keys, (list, tuple)):
            for k in alt_keys:
                if isinstance(d, dict) and k in d:
                    return d[k]
        # 若本身就是纯 state_dict
        if isinstance(d, dict) and all(isinstance(v, torch.Tensor) for v in d.values()):
            return d
        raise KeyError(f"State dict key not found: {primary_key}")

    def normalize_dp_keys(state_dict, need_module_prefix: bool):
        has_module = any(k.startswith('module.') for k in state_dict.keys())
        if need_module_prefix:
            if not has_module:
                return {f'module.{k}': v for k, v in state_dict.items()}
            return state_dict
        else:
            if has_module:
                return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            return state_dict

    use_dp = isinstance(encoder, nn.DataParallel)

    # 逐组件提取并规范化键名
    enc_sd = pick_state(ckpt, 'encoder_state_dict', alt_keys=['DIDF_Encoder', 'Encoder', 'encoder'])
    ali_sd = pick_state(ckpt, 'alignment_state_dict', alt_keys=['Alignment', 'alignment'])
    base_sd = pick_state(ckpt, 'base_fuse_state_dict', alt_keys=['BaseFuseLayer', 'base_fuse'])
    det_sd = pick_state(ckpt, 'detail_fuse_state_dict', alt_keys=['DetailFuseLayer', 'detail_fuse'])
    cls_sd  = pick_state(ckpt, 'classifier_state_dict', alt_keys=['Classifier', 'classifier'])

    enc_sd = normalize_dp_keys(enc_sd, need_module_prefix=use_dp)
    ali_sd = normalize_dp_keys(ali_sd, need_module_prefix=use_dp)
    base_sd = normalize_dp_keys(base_sd, need_module_prefix=use_dp)
    det_sd = normalize_dp_keys(det_sd, need_module_prefix=use_dp)
    cls_sd  = normalize_dp_keys(cls_sd, need_module_prefix=use_dp)

    # 容忍微小不匹配
    (encoder.module if use_dp else encoder).load_state_dict(enc_sd, strict=False)
    (alignment.module if use_dp else alignment).load_state_dict(ali_sd, strict=False)
    (base_fuse.module if use_dp else base_fuse).load_state_dict(base_sd, strict=False)
    (detail_fuse.module if use_dp else detail_fuse).load_state_dict(det_sd, strict=False)
    (classifier.module if use_dp else classifier).load_state_dict(cls_sd, strict=False)

for model in [encoder, alignment, base_fuse, detail_fuse, classifier]:
    model.eval()

test_dataset = PairDataset(config.palm_dir, config.vein_dir, get_transforms(), is_train=False, train_ratio=0.8)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

all_features = []
all_labels = []
all_scores = []  

with torch.no_grad():
    for palm, vein, labels in tqdm(test_loader, desc="处理中"):
        palm, vein = palm.to(device), vein.to(device)
        
        logits, feature_vector = forward_model(
            encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier
        )
        
        scores = torch.max(logits, dim=1)[0].cpu().numpy()
        
        all_features.append(feature_vector.cpu().numpy())
        all_labels.append(labels.numpy())
        all_scores.append(scores)

all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_scores = np.concatenate(all_scores, axis=0)

genuine_scores = []  
impostor_scores = []  

id_to_indices = {}
for idx, label in enumerate(all_labels):
    if label not in id_to_indices:
        id_to_indices[label] = []
    id_to_indices[label].append(idx)

for label, indices in id_to_indices.items():
    for i, j in itertools.combinations(indices, 2):
        sim = cosine_similarity([all_features[i]], [all_features[j]])[0][0]
        genuine_scores.append(sim)
    
    other_labels = [l for l in id_to_indices.keys() if l != label]
    for other_label in other_labels[:min(5, len(other_labels))]:  
        for i in indices[:min(3, len(indices))]:  
            for j in id_to_indices[other_label][:min(3, len(id_to_indices[other_label]))]:
                sim = cosine_similarity([all_features[i]], [all_features[j]])[0][0]
                impostor_scores.append(sim)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"正样本对数量: {len(genuine_scores)}")
print(f"负样本对数量: {len(impostor_scores)}")

print(f"genuine_scores: {np.sum(genuine_scores):.6f}") 
print(f"impostor_scores: {np.sum(impostor_scores):.6f}")

scores = np.concatenate([genuine_scores, impostor_scores])
labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]).astype(int)

eer, eer_threshold = compute_eer(scores, labels, is_similarity=True, return_threshold=True)
fpr, tpr, thresholds, auc_score = roc_auc(scores, labels, is_similarity=True)

target_fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
rows = []
for far in target_fars:
    res = tar_at_far(scores, labels, target_far=far, is_similarity=True)
    rows.append([f"{far:.5f}", f"{res['TAR']:.4f}", f"{res['threshold']:.4f}" if np.isfinite(res['threshold']) else 'inf'])

# 输出 EER 与 AUC
print(f"EER: {eer:.4f}, EER 阈值: {eer_threshold:.4f}")
print(f"AUC: {auc_score:.4f}")

col_widths = [10, 10, 14]
headers = ["FAR", "TAR", "Threshold"]
header_line = f"{headers[0]:<{col_widths[0]}} | {headers[1]:<{col_widths[1]}} | {headers[2]:<{col_widths[2]}}"
sep_line = f"{'-'*col_widths[0]}+{'-'*col_widths[1]}+{'-'*col_widths[2]}"
print(header_line)
print(sep_line)
for row in rows:
    line = f"{row[0]:<{col_widths[0]}} | {row[1]:<{col_widths[1]}} | {row[2]:<{col_widths[2]}}"
    print(line)

