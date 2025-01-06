print("Starting evaluate_model.py...")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from mol_model import Conv1DModel, CustomDataset, extract_and_save_features
from mol import MolFormer


# 配置文件路径
config_path = './Data/mol/hparams.yaml'
ckpt_path = '/home/longyh/database/molformer/PretrainedMoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
vocab_path = './Data/mol/bert_vocab.txt'

# 初始化模型
print("Initializing MolFormer model...")
molformer = MolFormer(config_path, ckpt_path, vocab_path, device='cuda')

# 测试集路径
test_set = '/work/longyh/MolProtMHC/Data/val1k.csv'

# 读取测试集数据
print("Reading test dataset...")
test_df = pd.read_csv(test_set)
test_peptide = test_df.smiles_peptide
test_HLA = test_df.smiles_HLA
test_labels = test_df.label

# 提取测试集特征
print("Extracting test features...")
test_peptide_features = extract_and_save_features(molformer, test_peptide, '/work/longyh/mhc/test_peptide_features.npy', 'cuda').half()
test_HLA_features = extract_and_save_features(molformer, test_HLA, '/work/longyh/mhc/test_HLA_features.npy', 'cuda').half()
test_labels = torch.tensor(test_labels.values).to('cuda')

# 创建数据集和数据加载器
print("Creating test DataLoader...")
test_dataset = CustomDataset(test_peptide_features, test_HLA_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False)

# 定义模型
print("Defining Conv1DModel...")
input_channels = test_peptide_features.size(1) + test_HLA_features.size(1)  # 使用特征的第一个维度作为输入通道数
input_size = test_peptide_features.size(2)  # 使用特征的第二个维度作为输入大小
num_classes = len(test_labels.unique())

model = Conv1DModel(input_channels, input_size, num_classes).to('cuda')
# model.load_state_dict(torch.load('/work/longyh/MolProtMHC/model1_20250102_164712.pth'))
model.load_state_dict(torch.load('/work/longyh/MolProtMHC/model1_20250103_002600.pth'))

model.eval()

# 评估模型
print("Evaluating model...")
all_labels = []
all_outputs = []

with torch.no_grad():
    for peptide_batch, HLA_batch, label_batch in test_loader:
        outputs = model(torch.cat((peptide_batch, HLA_batch), dim=1).float())
        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(label_batch.cpu().numpy())

all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)

# 计算AUC
print("Calculating AUC...")
auc = roc_auc_score(all_labels, all_outputs[:, 1])
fpr, tpr, _ = roc_curve(all_labels, all_outputs[:, 1])

# 计算AUPR
print("Calculating AUPR...")
aupr = average_precision_score(all_labels, all_outputs[:, 1])
precision, recall, _ = precision_recall_curve(all_labels, all_outputs[:, 1])

# 计算SRCC
print("Calculating SRCC...")
srcc, _ = spearmanr(all_labels, all_outputs[:, 1])

# 绘制AUC曲线
print("Plotting ROC curve...")
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('/work/longyh/MolProtMHC/train_output/mol_output/roc_curve_500k.png')
plt.close()

# 绘制AUPR曲线
print("Plotting Precision-Recall curve...")
plt.figure()
plt.plot(recall, precision, label=f'AUPR = {aupr:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('/work/longyh/MolProtMHC/train_output/mol_output/pr_curve_500k.png')
plt.close()

# 打印SRCC
print(f'Spearman Rank Correlation Coefficient (SRCC): {srcc:.4f}')
