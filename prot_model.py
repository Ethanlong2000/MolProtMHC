import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from prot import ProteinBertModel

class AminoAcidFeatures:
    """氨基酸物理化学特征"""
    def __init__(self):
        # 氨基酸特征字典
        self.aa_features = {
            'A': [0.62,  0.0,   0.0,   0.0,   1.8],  # 疏水性,极性,电荷,大小,pKa
            'R': [-2.53, 3.0,   1.0,   1.0,   10.76],
            'N': [-0.78, 2.0,   0.0,   0.0,   8.33],
            'D': [-0.90, 3.0,   -1.0,  0.0,   3.86],
            'C': [0.29,  1.0,   0.0,   0.0,   8.33],
            'E': [-0.74, 3.0,   -1.0,  0.0,   4.25],
            'Q': [-0.85, 2.0,   0.0,   0.0,   8.33],
            'G': [0.48,  0.0,   0.0,   0.0,   7.03],
            'H': [-0.40, 2.0,   0.5,   0.0,   6.0],
            'I': [1.38,  0.0,   0.0,   1.0,   5.94],
            'L': [1.06,  0.0,   0.0,   1.0,   5.98],
            'K': [-1.50, 3.0,   1.0,   1.0,   9.74],
            'M': [0.64,  1.0,   0.0,   1.0,   5.74],
            'F': [1.19,  0.0,   0.0,   1.0,   5.48],
            'P': [0.12,  0.0,   0.0,   0.0,   6.30],
            'S': [-0.18, 1.0,   0.0,   0.0,   5.68],
            'T': [-0.05, 1.0,   0.0,   0.0,   6.16],
            'W': [0.81,  1.0,   0.0,   1.0,   5.89],
            'Y': [0.26,  2.0,   0.0,   1.0,   5.66],
            'V': [1.08,  0.0,   0.0,   1.0,   5.96],
            'X': [0.0,   0.0,   0.0,   0.0,   7.00],  # 未知氨基酸
            '<PAD>': [0.0, 0.0, 0.0,   0.0,   0.0],   # 填充字符
        }
        
        # 转换为张量
        self.feature_matrix = torch.tensor([
            self.aa_features[aa] for aa in self.aa_features.keys()
        ], dtype=torch.float32)
        
        # 氨基酸到索引的映射
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.aa_features.keys())}

class MHCPeptideModel(nn.Module):
    def __init__(
        self,
        bert_model_path,
        hidden_dim=1024,
        dropout_rate=0.3,
        freeze_bert=True,
        device=None
    ):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_bert = freeze_bert
        
        # 氨基酸特征
        self.aa_features = AminoAcidFeatures()
        self.aa_feature_dim = 5  # 物理化学特征维度
        
        # BERT特征提取
        self.bert_model = ProteinBertModel(bert_model_path, device=self.device)
        self.set_bert_training_mode(not freeze_bert)
        
        # 物理化学特征处理
        self.physchem_conv = nn.Sequential(
            nn.Conv1d(self.aa_feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 计算融合后的特征维度
        physchem_dim = 64 * 2  # MHC和peptide的物理化学特征
        total_dim = hidden_dim + physchem_dim  # BERT特征 + 物理化学特征
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def get_physchem_features(self, sequences):
        """获取序列的物理化学特征"""
        batch_features = []
        for seq in sequences:
            # 将序列转换为索引
            indices = torch.tensor([
                self.aa_features.aa_to_idx.get(aa, self.aa_features.aa_to_idx['X']) 
                for aa in seq
            ], device=self.device)
            # 获取特征
            features = F.embedding(indices, self.aa_features.feature_matrix.to(self.device))
            batch_features.append(features)
        
        # 填充到相同长度
        padded_features = torch.nn.utils.rnn.pad_sequence(
            batch_features, 
            batch_first=True,
            padding_value=0
        )
        return padded_features
    
    def forward(self, mhc_sequences, peptide_sequences):
        # BERT特征
        bert_features = self.bert_model.get_pooler_output(
            [f"{mhc}{peptide}" for mhc, peptide in zip(mhc_sequences, peptide_sequences)]
        )
        
        # 物理化学特征
        mhc_physchem = self.get_physchem_features(mhc_sequences)
        pep_physchem = self.get_physchem_features(peptide_sequences)
        
        # 处理物理化学特征
        mhc_physchem = self.physchem_conv(mhc_physchem.transpose(1, 2))  # [batch, 64, seq_len]
        pep_physchem = self.physchem_conv(pep_physchem.transpose(1, 2))  # [batch, 64, seq_len]
        
        # 全局池化
        mhc_physchem = F.adaptive_max_pool1d(mhc_physchem, 1).squeeze(-1)  # [batch, 64]
        pep_physchem = F.adaptive_max_pool1d(pep_physchem, 1).squeeze(-1)  # [batch, 64]
        
        # 组合物理化学特征
        physchem_features = torch.cat([mhc_physchem, pep_physchem], dim=1)  # [batch, 128]
        
        # 打印维度信息以便调试
        print(f"BERT features shape: {bert_features.shape}")
        print(f"Physchem features shape: {physchem_features.shape}")
        
        # 特征融合
        combined_features = torch.cat([bert_features, physchem_features], dim=1)
        print(f"Combined features shape: {combined_features.shape}")
        
        fused_features = self.fusion_layer(combined_features)
        
        # 分类
        output = self.classifier(fused_features)
        return output
    
    def get_trainable_params(self):
        """获取需要训练的参数"""
        # 如果BERT被冻结，则不包含BERT的参数
        if self.freeze_bert:
            return [p for n, p in self.named_parameters() 
                   if not n.startswith('bert_model')]
        # 如果BERT没有被冻结，返回所有参数
        return self.parameters()
    
    def set_bert_training_mode(self, mode=False):
        """设置BERT层是否参与训练"""
        self.freeze_bert = not mode
        for param in self.bert_model.parameters():
            param.requires_grad = mode

class MHCPeptideTrainer:
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6
        )
    
    def train_step(self, mhc_sequences, peptide_sequences, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 确保标签数据在正确的设备上
        labels = labels.to(self.device)
        
        outputs = self.model(mhc_sequences, peptide_sequences)
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mhc_sequences, peptide_sequences, labels):
        self.model.eval()
        with torch.no_grad():
            # 确保标签数据在正确的设备上
            labels = labels.to(self.device)
            
            outputs = self.model(mhc_sequences, peptide_sequences)
            loss = self.criterion(outputs, labels)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == labels).float().mean()
        
        return loss.item(), accuracy.item()

    def unfreeze_bert(self, learning_rate=1e-5):
        """解冻BERT层并重新初始化优化器"""
        self.model.set_bert_training_mode(True)
        # 为BERT层使用较小的学习率
        bert_params = {'params': self.model.bert_model.parameters(), 'lr': learning_rate}
        other_params = {'params': [p for n, p in self.model.named_parameters() 
                                 if not n.startswith('bert_model')]}
        self.optimizer = torch.optim.AdamW([bert_params, other_params])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6
        )
