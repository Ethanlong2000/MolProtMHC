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

from prot import ProteinBertModel
from mol import MolFormer

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 多层感知机分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):  # 增加一个隐藏层
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # 添加批归一��层
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)  # 添加批归一化层
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # 添加Dropout层
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 新增隐藏层
        self.bn3 = nn.BatchNorm1d(hidden_dim3)  # 新增批归一化层
        self.relu3 = nn.ReLU()  # 新增激活函数
        self.dropout3 = nn.Dropout(0.5)  # 添加Dropout层
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # 修改输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 批归一化
        x = self.relu1(x)
        x = self.dropout1(x)  # 应用Dropout
        x = self.fc2(x)
        x = self.bn2(x)  # 批归一化
        x = self.relu2(x)
        x = self.dropout2(x)  # 应用Dropout
        x = self.fc3(x)  # 新增隐藏层
        x = self.bn3(x)  # 新增批归一化层
        x = self.relu3(x)  # 新增激活函数
        x = self.dropout3(x)  # 应用Dropout
        x = self.fc4(x)  # 修改输出层
        x = self.sigmoid(x)
        return x

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=5, scheduler=None):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间
        epoch_loss = 0.0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).float()  # 转换标签为浮点数
            labels = labels.view(-1, 1)  # 调整标签形状
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        end_time = time.time()  # 记录结束时间
        epoch_duration = end_time - start_time  # 计算训练时间

        # 验证模型
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).float()
                labels = labels.view(-1, 1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Val Loss: {val_loss}, Duration: {epoch_duration:.2f} seconds")

        if scheduler:
            scheduler.step(val_loss)  # 根据验证损失更新学习率

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses, val_losses

# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 设置日志记录
def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(message)s')

# 数据预处理
def preprocess_data(csv_file, protein_col, smiles_col, label_col):
    try:
        df = pd.read_csv(csv_file)
        protein_sequences = df[protein_col].tolist()
        smiles = df[smiles_col].tolist()
        labels = df[label_col].tolist()
        return protein_sequences, smiles, labels
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

# 保存模型
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
    model_path = f"{path}_{timestamp}.pt"  # 在模型名称上添加时间戳
    torch.save(model.state_dict(), model_path)

# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))

# 提取特征
def extract_features(protein_model, mol_model, protein_sequences, smiles, batch_size, device):
    protein_features = []
    mol_features = []
    for i in range(0, len(protein_sequences), batch_size):
        batch_protein_sequences = protein_sequences[i:i+batch_size]
        batch_smiles = smiles[i:i+batch_size]
        protein_features.append(protein_model.get_pooler_output(batch_protein_sequences).to(device))
        mol_features.append(mol_model.embed(batch_smiles).to(device))
    protein_features = torch.cat(protein_features, dim=0)
    mol_features = torch.cat(mol_features, dim=0)
    return torch.cat((protein_features, mol_features), dim=1)

# 保存特征
def save_features(features, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(features, f)

# 加载特征
def load_features(path):
    with open(path, 'rb') as f:
        features = pickle.load(f)
    return features

# 冻结模型参数
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# 主函数
def main(config):
    setup_logging(config['log_file'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()  # 记录脚本开始时间

    protein_model = ProteinBertModel(config['protein_model_path'], device=device)
    mol_model = MolFormer(config['mol_config_path'], config['mol_ckpt_path'], config['mol_vocab_path'], device=device)

    # 冻结预训练模型的参数
    freeze_model_parameters(protein_model)
    freeze_model_parameters(mol_model)

    protein_sequences, smiles, labels = preprocess_data(config['csv_file'], 
                                                        config['protein_col'], 
                                                        config['smiles_col'], 
                                                        config['label_col'])

    feature_path = config.get('feature_save_path', 'features.pkl') 
    if os.path.exists(feature_path): # 如果特征文件存在，则加载特征
        features = load_features(feature_path)
        print(f"Loaded features from {feature_path}")
    else:
        feature_start_time = time.time()  # 记录特征提取开始时间
        features = extract_features(protein_model, mol_model, protein_sequences, smiles, config['batch_size'], device)
        feature_end_time = time.time()  # 记录特征提取结束时间
        feature_duration = feature_end_time - feature_start_time  # 计算特征提取时间
        print(f"Feature extraction duration: {feature_duration:.2f} seconds")
        save_features(features, feature_path)
        print(f"Saved features to {feature_path}")

    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_features, torch.tensor(train_labels).to(device))
    val_dataset = CustomDataset(val_features, torch.tensor(val_labels).to(device))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    classifier = MLPClassifier(config['input_dim'], config['hidden_dim1'], config['hidden_dim2'], config['hidden_dim3'], config['output_dim']).to(device)  # 修改初始化参数
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=float(config['learning_rate']), weight_decay=1e-5)  # 添加L2正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)  # 调整patience参数

    try:
        train_losses, val_losses = train_model(classifier, train_loader, val_loader, criterion, optimizer, device, num_epochs=config['num_epochs'], patience=config.get('patience', 10), scheduler=scheduler)  # 调整早停的耐心参数
        save_model(classifier, config['model_save_path'])
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    # 绘制损失曲线
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_curve_path = os.path.join(os.path.dirname(config['model_save_path']), f'loss_curve_{timestamp}.png')
    plt.savefig(loss_curve_path)
    # plt.show()

    end_time = time.time()  # 记录脚本结束时间
    total_duration = end_time - start_time  # 计算总时间花费
    print(f"Total script duration: {total_duration:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction and Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    config['hidden_dim3'] = 64  # 新增隐藏层维度
    main(config)