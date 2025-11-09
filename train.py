import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from net import (Restormer_Encoder, DeformableAlignment, 
                     BaseFeatureExtraction, DetailFeatureExtraction, ArcFaceClassifier)
from utils.loss import TripletLoss, RecognitionLoss
from utils.dataset import ContrastDataset, PairDataset

# 测试git
# 配置参数
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'models'
    log_dir = 'runs' 
    palm_dir1, vein_dir1 = '/root/autodl-tmp/CDDFuse/MMIF-CDDFuse-main/roi/CASIA/vi/roi', '/root/autodl-tmp/CDDFuse/MMIF-CDDFuse-main/roi/CASIA/ir/roi'
    #palm_dir2, vein_dir2 = 
    p1_epochs, p1_batch, p1_lr = 50, 8, 1e-4 
    p1_patience = 8 
    
    p2_epochs, p2_batch, p2_lr, p2_enc_lr = 50, 8, 1e-4, 1e-5 
    p2_patience = 15 
    num_classes, train_ratio = 100, 0.8
    
    dim = 64            # 网络特征主通道数
    num_heads = 8       # 多头注意力机制的头数

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value, mode='min'):
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        improved = (current_value < self.best_value - self.min_delta) if mode == 'min' else \
                   (current_value > self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def get_transforms(strong=True):
    base = [transforms.Resize((128, 128))]
    if strong:
        base += [transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3)]
    else:
        base += [transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.05, 0.05))]
    
    base += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(base)


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
    logits, _ = classifier(fused, labels)
    
    return logits, base_p, base_v, detail_p, detail_v, fused_base, fused_detail


# ============================================================
# 第一阶段：对比学习
# ============================================================
def train_phase1(encoder, config, writer=None):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    dataset = ContrastDataset(config.palm_dir1, config.vein_dir1, get_transforms(strong=True))
    loader = DataLoader(dataset, config.p1_batch, shuffle=True, num_workers=4, drop_last=True)
    
    criterion = TripletLoss(margin=0.5)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.p1_lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p1_epochs)
    early_stop = EarlyStopping(patience=config.p1_patience, min_delta=0.001)
    
    best_loss = float('inf')
    
    for epoch in range(config.p1_epochs):
        encoder.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f'P1 Epoch {epoch+1}/{config.p1_epochs}')
        for batch_idx, (anchor, positive, negative, _) in enumerate(pbar):
            anchor, positive, negative = anchor.to(config.device), positive.to(config.device), negative.to(config.device)
            
            feat_a = torch.nn.functional.adaptive_avg_pool2d(encoder(anchor)[0], 1).flatten(1)
            feat_p = torch.nn.functional.adaptive_avg_pool2d(encoder(positive)[0], 1).flatten(1)
            feat_n = torch.nn.functional.adaptive_avg_pool2d(encoder(negative)[0], 1).flatten(1)
            
            # 损失和优化
            total_loss, dist_ap, dist_an = criterion(feat_a, feat_p, feat_n)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 
                            'd(a,p)': f'{dist_ap.item():.3f}', 
                            'd(a,n)': f'{dist_an.item():.3f}'})
            
            # TensorBoard记录（每个batch）
            if writer is not None:
                global_step = epoch * len(loader) + batch_idx
                writer.add_scalar('Phase1/BatchLoss', total_loss.item(), global_step)
                writer.add_scalar('Phase1/Dist_AP', dist_ap.item(), global_step)
                writer.add_scalar('Phase1/Dist_AN', dist_an.item(), global_step)
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard记录（每个epoch）
        if writer is not None:
            writer.add_scalar('Phase1/EpochLoss', avg_loss, epoch)
            writer.add_scalar('Phase1/LearningRate', current_lr, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch+1, 'encoder_state_dict': encoder.state_dict(), 'loss': avg_loss},
                      os.path.join(config.save_dir, 'encoder_phase1_best.pth'))
        
        if early_stop(avg_loss, mode='min'):
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_loss


# ============================================================
# 第二阶段：识别任务
# ============================================================
def train_phase2(encoder, config, writer=None):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    ckpt_path = os.path.join(config.save_dir, 'encoder_phase1_best.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=config.device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    
    train_ds = PairDataset(config.palm_dir1, config.vein_dir1, get_transforms(strong=False), True, config.train_ratio)
    val_ds = PairDataset(config.palm_dir1, config.vein_dir1, get_transforms(strong=False), False, config.train_ratio)
    train_loader = DataLoader(train_ds, config.p2_batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, config.p2_batch, shuffle=False, num_workers=4)
    
    alignment = nn.DataParallel(DeformableAlignment(config.dim)).to(config.device)
    base_fuse = nn.DataParallel(BaseFeatureExtraction(config.dim, config.num_heads)).to(config.device)
    detail_fuse = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(config.device)
    classifier = nn.DataParallel(ArcFaceClassifier(config.dim*2, config.num_classes)).to(config.device)
    
    criterion = RecognitionLoss(w_cls=1.0, w_decomp=0.3, label_smoothing=0.1)
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': config.p2_enc_lr},
        {'params': alignment.parameters(), 'lr': config.p2_lr},
        {'params': base_fuse.parameters(), 'lr': config.p2_lr},
        {'params': detail_fuse.parameters(), 'lr': config.p2_lr},
        {'params': classifier.parameters(), 'lr': config.p2_lr}
    ], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)
    early_stop = EarlyStopping(patience=config.p2_patience, min_delta=0.1)
    
    best_acc = 0.0
    
    for epoch in range(config.p2_epochs):
        for model in [encoder, alignment, base_fuse, detail_fuse, classifier]:
            model.train()
        
        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f'P2 Epoch {epoch+1}/{config.p2_epochs}')
        
        for palm, vein, labels in pbar:
            palm, vein, labels = palm.to(config.device), vein.to(config.device), labels.to(config.device)
            
            logits, base_p, base_v, detail_p, detail_v, fused_base, fused_detail = forward_model(
                encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier, labels
            )
            
            # 损失和优化
            total_loss, _, _, _ = criterion(
                logits, labels, base_p, base_v, detail_p, detail_v, fused_base, fused_detail
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            _, pred = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            train_loss_sum += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss_sum / len(train_loader)
        
        # 验证
        for model in [encoder, alignment, base_fuse, detail_fuse, classifier]:
            model.eval()
        
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for palm, vein, labels in val_loader:
                palm, vein, labels = palm.to(config.device), vein.to(config.device), labels.to(config.device)
                logits, base_p, base_v, detail_p, detail_v, fused_base, fused_detail = forward_model(
                    encoder, palm, vein, alignment, base_fuse, detail_fuse, classifier)
                total_loss, _, _, _ = criterion(
                    logits, labels, base_p, base_v, detail_p, detail_v, fused_base, fused_detail)
                _, pred = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
                val_loss_sum += total_loss.item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard记录
        if writer is not None:
            writer.add_scalar('Phase2/TrainLoss', train_loss, epoch)
            writer.add_scalar('Phase2/TrainAcc', train_acc, epoch)
            writer.add_scalar('Phase2/ValLoss', val_loss, epoch)
            writer.add_scalar('Phase2/ValAcc', val_acc, epoch)
            writer.add_scalar('Phase2/LearningRate', current_lr, epoch)
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch+1, 'encoder_state_dict': encoder.state_dict(),
                'alignment_state_dict': alignment.state_dict(),
                'base_fuse_state_dict': base_fuse.state_dict(),
                'detail_fuse_state_dict': detail_fuse.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'best_acc': best_acc
            }, os.path.join(config.save_dir, 'full_model_phase2_best.pth'))
        
        if early_stop(train_acc, mode='max'):
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_acc

def main():
    writer = SummaryWriter(log_dir=config.log_dir)
    
    encoder = Restormer_Encoder(
        inp_channels=1, dim=config.dim, num_blocks=[4,4],
        heads=[config.num_heads]*3
    ).to(config.device)
    
    p1_loss = train_phase1(encoder, config, writer)
    p2_acc = train_phase2(encoder, config, writer)
    
    print(f"Phase1 Min Loss: {p1_loss:.4f}")
    print(f"Phase2 Max Accuracy: {p2_acc:.2f}%")
    
    writer.close()

if __name__ == '__main__':
    main()