from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import image_read_cv2
import warnings
import logging
import cv2
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_chinese_visualization(dataset_name, model_type="IVF"):
    """
    创建支持中文的CDDFuse模块可视化
    """
    
    # 选择模型路径
    if model_type == "IVF":
        ckpt_path = r"models/CDDFuse_IVF.pth"
    else:
        ckpt_path = r"models/CDDFuse_MIF.pth"
    
    # 设置路径
    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('test_result', dataset_name + '_chinese_vis')
    
    # 检查输入文件夹
    ir_folder = os.path.join(test_folder, "ir")
    vi_folder = os.path.join(test_folder, "vi")
    
    if not os.path.exists(ir_folder) or not os.path.exists(vi_folder):
        print(f"错误：请在 {test_folder} 下创建 'ir' 和 'vi' 子文件夹")
        return
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")
    
    # 加载模型
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
    
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    
    # 处理第一张图片
    ir_images = os.listdir(ir_folder)
    img_name = ir_images[0]
    
    if img_name in os.listdir(vi_folder):
        # 读取图片
        data_IR = image_read_cv2(os.path.join(ir_folder, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        data_VIS = image_read_cv2(os.path.join(vi_folder, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        
        # 转换为张量
        data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        
        print(f"处理图片：{img_name}")
        
        with torch.no_grad():
            # 获取所有中间结果
            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            
            if model_type == "IVF":
                data_Fuse, _ = Decoder(data_IR + data_VIS, feature_F_B, feature_F_D)
            else:
                data_Fuse, _ = Decoder(None, feature_F_B, feature_F_D)
            
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            
            # 创建中文可视化
            create_chinese_plots(
                data_IR, data_VIS, data_Fuse,
                feature_V_B, feature_V_D, feature_I_B, feature_I_D,
                feature_F_B, feature_F_D,
                img_name, test_out_folder
            )

def create_chinese_plots(data_IR, data_VIS, data_Fuse,
                       feature_V_B, feature_V_D, feature_I_B, feature_I_D,
                       feature_F_B, feature_F_D,
                       img_name, save_path):
    """创建支持中文的可视化图"""
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 转换为numpy数组
    ir_img = np.squeeze((data_IR * 255).cpu().numpy()).astype(np.uint8)
    vis_img = np.squeeze((data_VIS * 255).cpu().numpy()).astype(np.uint8)
    fuse_img = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
    
    # 计算特征图
    vis_b_feat = torch.mean(feature_V_B, dim=1).squeeze().cpu().numpy()
    vis_d_feat = torch.mean(feature_V_D, dim=1).squeeze().cpu().numpy()
    ir_b_feat = torch.mean(feature_I_B, dim=1).squeeze().cpu().numpy()
    ir_d_feat = torch.mean(feature_I_D, dim=1).squeeze().cpu().numpy()
    fuse_b_feat = torch.mean(feature_F_B, dim=1).squeeze().cpu().numpy()
    fuse_d_feat = torch.mean(feature_F_D, dim=1).squeeze().cpu().numpy()
    
    # 创建主可视化图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'CDDFuse模块处理结果 - {img_name}', fontsize=16, fontweight='bold')
    
    # 第一行：输入和输出图像
    axes[0, 0].imshow(ir_img, cmap='gray')
    axes[0, 0].set_title('红外图像 (IR)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vis_img, cmap='gray')
    axes[0, 1].set_title('可见光图像 (VIS)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fuse_img, cmap='gray')
    axes[0, 2].set_title('融合结果', fontsize=12)
    axes[0, 2].axis('off')
    
    # 添加处理流程图
    axes[0, 3].text(0.5, 0.8, 'CDDFuse处理流程', fontsize=12, fontweight='bold', ha='center')
    axes[0, 3].text(0.5, 0.6, '1. Encoder特征提取', fontsize=10, ha='center')
    axes[0, 3].text(0.5, 0.5, '2. 特征分解(Base/Detail)', fontsize=10, ha='center')
    axes[0, 3].text(0.5, 0.4, '3. 特征融合', fontsize=10, ha='center')
    axes[0, 3].text(0.5, 0.3, '4. Decoder图像重建', fontsize=10, ha='center')
    axes[0, 3].axis('off')
    
    # 第二行：VIS特征
    axes[1, 0].imshow(vis_b_feat, cmap='hot')
    axes[1, 0].set_title('VIS基础特征\n(全局信息)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(vis_d_feat, cmap='hot')
    axes[1, 1].set_title('VIS细节特征\n(局部信息)', fontsize=10)
    axes[1, 1].axis('off')
    
    # VIS特征统计
    vis_b_mean = np.mean(vis_b_feat)
    vis_b_std = np.std(vis_b_feat)
    vis_d_mean = np.mean(vis_d_feat)
    vis_d_std = np.std(vis_d_feat)
    
    axes[1, 2].text(0.1, 0.8, 'VIS基础特征统计:', fontsize=10, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, f'均值: {vis_b_mean:.3f}', fontsize=9)
    axes[1, 2].text(0.1, 0.6, f'标准差: {vis_b_std:.3f}', fontsize=9)
    axes[1, 2].text(0.1, 0.4, 'VIS细节特征统计:', fontsize=10, fontweight='bold')
    axes[1, 2].text(0.1, 0.3, f'均值: {vis_d_mean:.3f}', fontsize=9)
    axes[1, 2].text(0.1, 0.2, f'标准差: {vis_d_std:.3f}', fontsize=9)
    axes[1, 2].axis('off')
    
    # VIS特征分布直方图
    axes[1, 3].hist(vis_b_feat.flatten(), bins=50, alpha=0.7, label='基础特征', color='blue')
    axes[1, 3].hist(vis_d_feat.flatten(), bins=50, alpha=0.7, label='细节特征', color='red')
    axes[1, 3].set_title('VIS特征分布', fontsize=10)
    axes[1, 3].set_xlabel('特征值', fontsize=9)
    axes[1, 3].set_ylabel('频次', fontsize=9)
    axes[1, 3].legend(fontsize=8)
    
    # 第三行：IR特征
    axes[2, 0].imshow(ir_b_feat, cmap='hot')
    axes[2, 0].set_title('IR基础特征\n(全局信息)', fontsize=10)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ir_d_feat, cmap='hot')
    axes[2, 1].set_title('IR细节特征\n(局部信息)', fontsize=10)
    axes[2, 1].axis('off')
    
    # IR特征统计
    ir_b_mean = np.mean(ir_b_feat)
    ir_b_std = np.std(ir_b_feat)
    ir_d_mean = np.mean(ir_d_feat)
    ir_d_std = np.std(ir_d_feat)
    
    axes[2, 2].text(0.1, 0.8, 'IR基础特征统计:', fontsize=10, fontweight='bold')
    axes[2, 2].text(0.1, 0.7, f'均值: {ir_b_mean:.3f}', fontsize=9)
    axes[2, 2].text(0.1, 0.6, f'标准差: {ir_b_std:.3f}', fontsize=9)
    axes[2, 2].text(0.1, 0.4, 'IR细节特征统计:', fontsize=10, fontweight='bold')
    axes[2, 2].text(0.1, 0.3, f'均值: {ir_d_mean:.3f}', fontsize=9)
    axes[2, 2].text(0.1, 0.2, f'标准差: {ir_d_std:.3f}', fontsize=9)
    axes[2, 2].axis('off')
    
    # IR特征分布直方图
    axes[2, 3].hist(ir_b_feat.flatten(), bins=50, alpha=0.7, label='基础特征', color='blue')
    axes[2, 3].hist(ir_d_feat.flatten(), bins=50, alpha=0.7, label='细节特征', color='red')
    axes[2, 3].set_title('IR特征分布', fontsize=10)
    axes[2, 3].set_xlabel('特征值', fontsize=9)
    axes[2, 3].set_ylabel('频次', fontsize=9)
    axes[2, 3].legend(fontsize=8)
    
    plt.tight_layout()
    
    # 保存主可视化结果
    main_path = os.path.join(save_path, f'{img_name.split(".")[0]}_chinese_main.png')
    plt.savefig(main_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"中文主可视化图已保存：{main_path}")
    
    # 创建融合特征对比图
    create_fusion_comparison(
        vis_b_feat, vis_d_feat, ir_b_feat, ir_d_feat,
        fuse_b_feat, fuse_d_feat, img_name, save_path
    )

def create_fusion_comparison(vis_b_feat, vis_d_feat, ir_b_feat, ir_d_feat,
                           fuse_b_feat, fuse_d_feat, img_name, save_path):
    """创建融合特征对比图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'特征融合对比分析 - {img_name}', fontsize=16, fontweight='bold')
    
    # 基础特征对比
    axes[0, 0].imshow(vis_b_feat, cmap='hot')
    axes[0, 0].set_title('VIS基础特征', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir_b_feat, cmap='hot')
    axes[0, 1].set_title('IR基础特征', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fuse_b_feat, cmap='hot')
    axes[0, 2].set_title('融合基础特征', fontsize=12)
    axes[0, 2].axis('off')
    
    # 细节特征对比
    axes[1, 0].imshow(vis_d_feat, cmap='hot')
    axes[1, 0].set_title('VIS细节特征', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ir_d_feat, cmap='hot')
    axes[1, 1].set_title('IR细节特征', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(fuse_d_feat, cmap='hot')
    axes[1, 2].set_title('融合细节特征', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = os.path.join(save_path, f'{img_name.split(".")[0]}_chinese_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"中文特征对比图已保存：{comparison_path}")
    
    # 创建特征差异分析图
    create_difference_analysis(
        vis_b_feat, vis_d_feat, ir_b_feat, ir_d_feat,
        fuse_b_feat, fuse_d_feat, img_name, save_path
    )

def create_difference_analysis(vis_b_feat, vis_d_feat, ir_b_feat, ir_d_feat,
                             fuse_b_feat, fuse_d_feat, img_name, save_path):
    """创建特征差异分析图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'特征融合差异分析 - {img_name}', fontsize=16, fontweight='bold')
    
    # 计算差异
    diff_b = np.abs(fuse_b_feat - (vis_b_feat + ir_b_feat) / 2)
    diff_d = np.abs(fuse_d_feat - (vis_d_feat + ir_d_feat) / 2)
    
    # 基础特征差异
    im1 = axes[0, 0].imshow(diff_b, cmap='hot')
    axes[0, 0].set_title('基础特征融合差异', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 细节特征差异
    im2 = axes[0, 1].imshow(diff_d, cmap='hot')
    axes[0, 1].set_title('细节特征融合差异', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 差异统计
    axes[1, 0].text(0.1, 0.8, '基础特征差异统计:', fontsize=12, fontweight='bold')
    axes[1, 0].text(0.1, 0.7, f'平均差异: {np.mean(diff_b):.4f}', fontsize=10)
    axes[1, 0].text(0.1, 0.6, f'最大差异: {np.max(diff_b):.4f}', fontsize=10)
    axes[1, 0].text(0.1, 0.5, f'标准差: {np.std(diff_b):.4f}', fontsize=10)
    axes[1, 0].text(0.1, 0.3, '细节特征差异统计:', fontsize=12, fontweight='bold')
    axes[1, 0].text(0.1, 0.2, f'平均差异: {np.mean(diff_d):.4f}', fontsize=10)
    axes[1, 0].text(0.1, 0.1, f'最大差异: {np.max(diff_d):.4f}', fontsize=10)
    axes[1, 0].text(0.1, 0.0, f'标准差: {np.std(diff_d):.4f}', fontsize=10)
    axes[1, 0].axis('off')
    
    # 差异分布直方图
    axes[1, 1].hist(diff_b.flatten(), bins=50, alpha=0.7, label='基础特征差异', color='blue')
    axes[1, 1].hist(diff_d.flatten(), bins=50, alpha=0.7, label='细节特征差异', color='red')
    axes[1, 1].set_title('差异分布', fontsize=12)
    axes[1, 1].set_xlabel('差异值', fontsize=10)
    axes[1, 1].set_ylabel('频次', fontsize=10)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存差异分析图
    diff_path = os.path.join(save_path, f'{img_name.split(".")[0]}_chinese_difference.png')
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"中文差异分析图已保存：{diff_path}")

if __name__ == "__main__":
    print("CDDFuse 中文可视化")
    print("=" * 50)
    
    # 创建中文可视化
    create_chinese_visualization("palm", "IVF")
    
    print("\n中文可视化完成！")
    print("请查看 test_result/palm_chinese_vis/ 文件夹中的结果")
