from mol import MolFormer
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import yaml
import argparse
import os
import time  # 添加计时模块
from datetime import datetime  # 添加日期时间模块
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split  # 添加数据集划分模块
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 添加学习率调度器
import matplotlib.pyplot as plt  # 添加绘图模块
import pickle  # 添加pickle模块


# 特征提取
def extract_features(mol_model, smiles, device):
    mol_features = mol_model.embed(smiles).to(device)
    return mol_features
# 
class Conv1DModel(nn.Module):
    def __init__(self, input_channels, input_size, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * (input_size // 8), num_classes)  # 调整全连接层输入大小
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(f"Conv1DModel - Flattened x shape: {x.shape}")  # 打印形状
        x = self.fc(x)
        return x

class SeparateConv1DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SeparateConv1DModel, self).__init__()
        self.conv1_peptide = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_HLA = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * (input_size // 8), num_classes)  # 调整全连接层输入大小
        self.relu = nn.ReLU()

    def forward(self, peptide, HLA):
        peptide = self.relu(self.conv1_peptide(peptide))
        HLA = self.relu(self.conv1_HLA(HLA))
        x = torch.cat((peptide, HLA), dim=1)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(f"SeparateConv1DModel - Flattened x shape: {x.shape}")  # 打印形状
        x = self.fc(x)
        return x

# 配置文件路径
datapath = '/work/longyh/MolProtMHC/Data/test1k.csv'
config_path = './Data/mol/hparams.yaml'
ckpt_path = '/home/longyh/database/molformer/PretrainedMoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
vocab_path = './Data/mol/bert_vocab.txt'

# 初始化模型
molformer = MolFormer(config_path, ckpt_path, vocab_path, device='cuda')

# 设置训练和验证数据
train_set='/work/longyh/mhc/train_data_fold4.csv'
valid_set='/work/longyh/mhc/val_data_fold4.csv'

# train_set='/work/longyh/MolProtMHC/Data/test100k.csv'
# val_set='/work/longyh/MolProtMHC/Data/val1k.csv'

# 读取数据
df = pd.read_csv(datapath)
peptide = df.smiles_peptide
HLA = df.smiles_HLA
lable=df.label

# 读取特征
if os.path.exists('/work/longyh/mhc/peptide_features.npy') and os.path.exists('/work/longyh/mhc/HLA_features.npy'):
    peptide_features = np.load('/work/longyh/mhc/peptide_features.npy')
    HLA_features = np.load('/work/longyh/mhc/HLA_features.npy')
else:
    peptide_features = extract_features(molformer, peptide, device='cuda')
    HLA_features = extract_features(molformer, HLA, device='cuda')
    np.save('/work/longyh/mhc/peptide_features.npy', peptide_features.cpu().numpy())
    np.save('/work/longyh/mhc/HLA_features.npy', HLA_features.cpu().numpy())

#打印shape
# print(peptide_features.shape)
# print(HLA_features.shape)



# 转换为Tensor并调整维度
peptide_features = torch.tensor(peptide_features).unsqueeze(1).to('cuda')
HLA_features = torch.tensor(HLA_features).unsqueeze(1).to('cuda')
labels = torch.tensor(lable.values).to('cuda')

# 合并特征
features = torch.cat((peptide_features, HLA_features), dim=1)

# 定义模型
input_channels = features.size(1)  # 使用特征的第一个维度作为输入通道数
input_size = features.size(2)  # 使用特征的第二个维度作为输入大小
num_classes = len(labels.unique())

# 模式1：先合并特征再经过卷积
model1 = Conv1DModel(input_channels, input_size, num_classes).to('cuda')

# 模式2：卷积之后再合并特征
model2 = SeparateConv1DModel(input_size, num_classes).to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# 定义学习率调度器
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=5, verbose=True)
scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=5, verbose=True)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    # 模式1训练
    model1.train()
    outputs1 = model1(features)
    loss1 = criterion(outputs1, labels)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    # 模式2训练
    model2.train()
    outputs2 = model2(peptide_features, HLA_features)
    loss2 = criterion(outputs2, labels)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')

    # 更新学习率
    scheduler1.step(loss1)
    scheduler2.step(loss2)

# 保存模型
# torch.save(model1.state_dict(), 'conv1d_model1.pth')
# torch.save(model2.state_dict(), 'conv1d_model2.pth')









