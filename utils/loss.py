import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================================
# 工具函数
# ====================================================================================
def cc(img1, img2):
    """
    计算两个特征图的相关系数（用于特征分解损失）
    """
    eps = torch.finfo(torch.float32).eps
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
        eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * 
        torch.sqrt(torch.sum(img2 ** 2, dim=-1))
    )
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


# ====================================================================================
# 第一阶段：三元组损失
# ====================================================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, feat_anchor, feat_positive, feat_negative):
        feat_anchor = F.normalize(feat_anchor, p=2, dim=1)
        feat_positive = F.normalize(feat_positive, p=2, dim=1)
        feat_negative = F.normalize(feat_negative, p=2, dim=1)

        dist_ap = 1.0 - F.cosine_similarity(feat_anchor, feat_positive, dim=1)
        dist_an = 1.0 - F.cosine_similarity(feat_anchor, feat_negative, dim=1)

        triplet_loss = F.relu(dist_ap - dist_an + self.margin).mean()

        total_loss = triplet_loss

        return total_loss, dist_ap.mean(), dist_an.mean()


# ====================================================================================
# 第二阶段：识别任务损失
# ====================================================================================
class RecognitionLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_decomp=0.5, w_consistency=0.1, label_smoothing=0.1):
        super(RecognitionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.w_cls = w_cls
        self.w_decomp = w_decomp
        self.w_consistency = w_consistency
    
    def forward(self, logits, labels, 
                base_palm, base_vein, detail_palm, detail_vein,
                fused_base, fused_detail):
        # 1. 分类损失
        cls_loss = self.ce_loss(logits, labels)
        
        # 2. 特征分解损失
        cc_base = cc(base_palm, base_vein)      # 跨模态基础特征相关性
        cc_detail = cc(detail_palm, detail_vein)  # 跨模态细节特征相关性
        cc_palm = cc(base_palm, detail_palm)     # 同模态正交性
        cc_vein = cc(base_vein, detail_vein)     # 同模态正交性
        
        decomp_loss = (cc_detail ** 2) / (1.01 + cc_base) + 0.1 * (cc_palm ** 2 + cc_vein ** 2)
        
        # 3. 融合一致性损失
        avg_base = (base_palm + base_vein) / 2.0
        avg_detail = (detail_palm + detail_vein) / 2.0
        consistency_loss = F.mse_loss(fused_base, avg_base) + F.mse_loss(fused_detail, avg_detail)
        
        total_loss = (self.w_cls * cls_loss + 
                     self.w_decomp * decomp_loss +
                     self.w_consistency * consistency_loss)
        
        return total_loss, cls_loss, decomp_loss, consistency_loss
