import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

# 导入我们的主干网络和数据加载模块
from sei_pipeline_agglomerative import Config, StudentModel, list_available_class_indices, load_data_by_split

def plot_similarity_distribution(features, labels, save_path="similarity_distribution.png"):
    """绘制同类与不同类之间的余弦相似度直方图分布"""
    print("计算全局余弦相似矩阵...")
    # 计算全对全余弦相似度
    sim_matrix = np.dot(features, features.T)
    
    # 提取标签匹配矩阵
    labels_grid = np.expand_dims(labels, 0) == np.expand_dims(labels, 1)
    
    # 提取上三角矩阵的索引以避免重复计算和自身(对角线)计算
    N = len(labels)
    upper_tri_indices = np.triu_indices(N, k=1)
    
    sims_upper = sim_matrix[upper_tri_indices]
    labels_upper = labels_grid[upper_tri_indices]
    
    # 分离同类相似度与不同类相似度
    intra_class_sims = sims_upper[labels_upper]
    inter_class_sims = sims_upper[~labels_upper]
    
    print(f"同类对数量: {len(intra_class_sims)}")
    print(f"非同类对数量: {len(inter_class_sims)}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(inter_class_sims, bins=100, alpha=0.6, color='blue', label='Inter-class (Different)', density=True)
    plt.hist(intra_class_sims, bins=100, alpha=0.6, color='red', label='Intra-class (Same)', density=True)
    
    plt.axvline(x=np.mean(intra_class_sims), color='darkred', linestyle='dashed', linewidth=1, label='Intra Mean')
    plt.axvline(x=np.mean(inter_class_sims), color='darkblue', linestyle='dashed', linewidth=1, label='Inter Mean')
    
    plt.title('Cosine Similarity Distribution of RF Fingerprints')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分布图已保存至: {save_path}")
    print(f"   => 提示: 请观察红蓝交界处的波谷，那个 X 轴的值就是你最完美的 merge_threshold (τ)！\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze tau threshold for SEI features")
    parser.add_argument('--data_dir', type=str, default="../data/LFM_dataset/data_noise_30")
    parser.add_argument('--ckpt', type=str, default="", help="Optional trained model checkpoint")
    args = parser.parse_args()

    config = Config()
    config.data_dir = args.data_dir
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    available_classes = list_available_class_indices(config.data_dir)
    config.class_to_label = {cls_idx: label_idx for label_idx, cls_idx in enumerate(available_classes)}
    
    print("提取完整测试集数据...")
    # 取出所有类的 test 数据
    x_test, y_test = load_data_by_split(config, available_classes, 'test')
    if x_test is None:
        print("未加载到测试集数据。")
        return

    student = StudentModel(config).to(config.device)
    if args.ckpt and os.path.exists(args.ckpt):
         student.load_state_dict(torch.load(args.ckpt))
         print(f"已加载权重: {args.ckpt}")
    else:
         print("注意：当前使用未经训练的初始网络权重提取特征！如果特征聚拢不明显，请传入训练后的ckpt。")

    student.eval()
    
    loader_test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test)), 
        batch_size=256, shuffle=False
    )
    
    feats_all, targets_all = [], []
    print("正在前向传播计算特征...")
    with torch.no_grad():
        for d, t in loader_test:
            d = d.to(config.device)
            f, _, _, _ = student(d)
            f = F.normalize(f, dim=1) # 必须做 L2 归一化
            feats_all.append(f.cpu().numpy())
            targets_all.append(t.numpy())
            
    feats = np.concatenate(feats_all)
    y_true = np.concatenate(targets_all)
    
    plot_similarity_distribution(feats, y_true, save_path=os.path.join(os.path.dirname(__file__), "tau_analysis_distribution.png"))

if __name__ == "__main__":
    main()
