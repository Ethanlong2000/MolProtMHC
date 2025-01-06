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
from tqdm import tqdm  # 添加进度条库
from torch.cuda.amp import GradScaler, autocast  # 添加混合精度训练


# 特征提取
def extract_features(mol_model, smiles, device, batch_size):  # 进一步减小批处理大小
    mol_features = mol_model.embed(smiles, batch_size=batch_size).to(device)
    return mol_features

def extract_and_save_features(molformer, smiles, feature_path, device, batch_size=256):  # 进一步减小批处理大小
    if os.path.exists(feature_path):
        features = np.load(feature_path)
    else:
        features = extract_features(molformer, smiles, device, batch_size=batch_size)
        np.save(feature_path, features.cpu().numpy())
    return torch.tensor(features).unsqueeze(1).to(device)


class Conv1DModel(nn.Module):
    def __init__(self, input_channels, input_size, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)  # 减小通道数
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 减小通道数
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)  # 减小通道数
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * (input_size // 8), num_classes)  # 调整全连接层输入大小
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
        self.conv1_peptide = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 减小通道数
        self.conv1_HLA = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 减小通道数
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 减小通道数
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)  # 减小通道数
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * (input_size // 8), num_classes)  # 调整全连接层输入大小
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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class CustomDataset(Dataset):
    def __init__(self, peptide_features, HLA_features, labels):
        self.peptide_features = peptide_features
        self.HLA_features = HLA_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        peptide_feature = self.peptide_features[idx]
        HLA_feature = self.HLA_features[idx]
        label = self.labels[idx]
        return peptide_feature, HLA_feature, label

if __name__ == "__main__":
    print("Starting mol_model.py main block...")

    # 配置文件路径
    config_path = './Data/mol/hparams.yaml'
    ckpt_path = '/home/longyh/database/molformer/PretrainedMoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
    vocab_path = './Data/mol/bert_vocab.txt'

    # 初始化模型
    molformer = MolFormer(config_path, ckpt_path, vocab_path, device='cuda')

    # 设置训练和验证数据
    # train_set = '/work/longyh/MolProtMHC/Data/train100k.csv'
    # val_set = '/work/longyh/MolProtMHC/Data/val100k.csv'
    train_set = '/work/longyh/mhc/train_data_fold2.csv'
    val_set = '/work/longyh/mhc/val_data_fold2.csv'

    # 读取训练集数据
    train_df = pd.read_csv(train_set)
    train_peptide = train_df.smiles_peptide
    train_HLA = train_df.smiles_HLA
    train_labels = train_df.label

    # 读取验证集数据
    val_df = pd.read_csv(val_set)
    val_peptide = val_df.smiles_peptide
    val_HLA = val_df.smiles_HLA
    val_labels = val_df.label

    # 提取训练集特征
    train_peptide_features = extract_and_save_features(molformer, train_peptide, '/work/longyh/mhc/train_peptide_features.npy', 'cuda').half()
    train_HLA_features = extract_and_save_features(molformer, train_HLA, '/work/longyh/mhc/train_HLA_features.npy', 'cuda').half()
    train_labels = torch.tensor(train_labels.values).to('cuda')

    # 提取验证集特征
    val_peptide_features = extract_and_save_features(molformer, val_peptide, '/work/longyh/mhc/val_peptide_features.npy', 'cuda').half()
    val_HLA_features = extract_and_save_features(molformer, val_HLA, '/work/longyh/mhc/val_HLA_features.npy', 'cuda').half()
    val_labels = torch.tensor(val_labels.values).to('cuda')

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(train_peptide_features, train_HLA_features, train_labels)
    val_dataset = CustomDataset(val_peptide_features, val_HLA_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False)

    # 定义模型
    input_channels = train_peptide_features.size(1) + train_HLA_features.size(1)  # 使用特征的第一个维度作为输入通道数
    input_size = train_peptide_features.size(2)  # 使用特征的第二个维度作为输入大小
    num_classes = len(train_labels.unique())

    print(f"Input channels: {input_channels}, Input size: {input_size}, Num classes: {num_classes}")

    # 选择模型
    model_choice = 'model1'  # 'model1' 或 'model2'

    if model_choice == 'model1':
        model = Conv1DModel(input_channels, input_size, num_classes).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)
    elif model_choice == 'model2':
        model = SeparateConv1DModel(input_size, num_classes).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义保存路径
    output_dir = '/work/longyh/MolProtMHC/train_output/mol_output'
    os.makedirs(output_dir, exist_ok=True)

    # 记录损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 混合精度训练
    scaler = GradScaler()

    # 训练模型
    num_epochs = 2000
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for peptide_batch, HLA_batch, label_batch in train_loader:
            optimizer.zero_grad()
            with autocast():
                if model_choice == 'model1':
                    outputs = model(torch.cat((peptide_batch, HLA_batch), dim=1).float())
                elif model_choice == 'model2':
                    outputs = model(peptide_batch.float(), HLA_batch.float())
                loss = criterion(outputs, label_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()  # 清理缓存

            train_loss += loss.item()  # 将 train_loss += loss.item() 移动到这里
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += label_batch.size(0)
            correct_train += (predicted == label_batch).sum().item()

            # 释放显存
            del outputs, loss
            torch.cuda.empty_cache()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # 验证
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for peptide_batch, HLA_batch, label_batch in val_loader:
                if model_choice == 'model1':
                    val_outputs = model(torch.cat((peptide_batch, HLA_batch), dim=1).float())
                elif model_choice == 'model2':
                    val_outputs = model(peptide_batch.float(), HLA_batch.float())
                val_loss += criterion(val_outputs, label_batch).item()
                
                # 计算验证准确率
                _, predicted = torch.max(val_outputs.data, 1)
                total_val += label_batch.size(0)
                correct_val += (predicted == label_batch).sum().item()

                # 释放显存
                del val_outputs
                torch.cuda.empty_cache()

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # 记录损失
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, Val_Loss: {val_loss / len(val_loader):.4f}, Train_Acc: {train_accuracy:.2f}%, Val_Acc: {val_accuracy:.2f}%')

        # 更新学习率
        scheduler.step(val_loss / len(val_loader))

        # 检查早停
        early_stopping(val_loss / len(val_loader))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'{model_choice}_{timestamp}.pth')

    # 绘制损失和准确率曲线
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_choice}_metrics_curve_{timestamp}.png'))
    plt.close()









