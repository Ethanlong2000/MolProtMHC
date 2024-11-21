import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import yaml
import argparse
import os
from torch.utils.data import DataLoader, Dataset

from prot import ProteinBertModel
from mol import MolFormer

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, protein_sequences, smiles, labels):
        self.protein_sequences = protein_sequences
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.protein_sequences[idx], self.smiles[idx], self.labels[idx]

# 多层感知机分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 训练模型函数
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for protein_seq, smiles, labels in dataloader:
            protein_seq, smiles, labels = protein_seq.to(device), smiles.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(protein_seq, smiles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

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
    torch.save(model.state_dict(), path)

# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))

# 提取特征
def extract_features(protein_model, mol_model, protein_sequences, smiles, batch_size):
    protein_features = []
    mol_features = []
    for i in range(0, len(protein_sequences), batch_size):
        batch_protein_sequences = protein_sequences[i:i+batch_size]
        batch_smiles = smiles[i:i+batch_size]
        protein_features.append(protein_model.get_pooler_output(batch_protein_sequences))
        mol_features.append(mol_model.embed(batch_smiles).cpu().numpy())
    protein_features = torch.cat(protein_features, dim=0)
    mol_features = torch.cat(mol_features, dim=0)
    return torch.cat((protein_features, mol_features), dim=1)

# 冻结模型参数
def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# 主函数
def main(config):
    setup_logging(config['log_file'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    protein_model = ProteinBertModel(config['protein_model_path'], device=device)
    mol_model = MolFormer(config['mol_config_path'], config['mol_ckpt_path'], config['mol_vocab_path'], device=device)

    # 冻结预训练模型的参数
    freeze_model_parameters(protein_model)
    freeze_model_parameters(mol_model)

    protein_sequences, smiles, labels = preprocess_data(config['csv_file'], 
                                                        config['protein_col'], 
                                                        config['smiles_col'], 
                                                        config['label_col'])

    features = extract_features(protein_model, mol_model, protein_sequences, smiles, config['batch_size'])

    dataset = CustomDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    classifier = MLPClassifier(config['input_dim'], config['hidden_dim'], config['output_dim']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    try:
        train_model(classifier, dataloader, criterion, optimizer, device, num_epochs=config['num_epochs'], patience=config.get('patience', 5))
        save_model(classifier, config['model_save_path'])
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction and Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)