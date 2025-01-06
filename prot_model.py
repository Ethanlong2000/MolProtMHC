import torch
import torch.nn as nn
from prot import ProteinBertModel

class MHCPeptideModel(nn.Module):
    def __init__(
        self,
        bert_model_path,
        hidden_dim=1024,
        gru_hidden_dim=512,
        num_gru_layers=2,
        dropout_rate=0.3,
        freeze_bert=True,
        device=None,
        max_length=49
    ):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.gru_hidden_dim = gru_hidden_dim
        
        # ProtBERT特征提取器
        self.bert_model = ProteinBertModel(bert_model_path, device=self.device)
        self.freeze_bert = freeze_bert
        self.set_bert_training_mode(not freeze_bert)
        
        # 添加BERT输出的BN层
        self.bert_bn = nn.BatchNorm1d(hidden_dim)
        
        # BiGRU层
        self.mhc_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_gru_layers > 1 else 0
        )
        
        # GRU输出的BN层
        self.gru_bn = nn.BatchNorm1d(gru_hidden_dim * 2)
        
        # 特征融合和分类层
        combined_dim = gru_hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),  # 添加BN
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(combined_dim // 2, combined_dim // 4),
            nn.BatchNorm1d(combined_dim // 4),  # 添加BN
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(combined_dim // 4, 1),
            nn.Sigmoid()
        )

    def combine_sequences(self, mhc_seq, peptide_seq):
        """将MHC和peptide序列组合，并添加填充"""
        combined = mhc_seq + peptide_seq
        # 使用X进行填充到指定长度
        padded = combined + 'X' * (self.max_length - len(combined))
        # 不需要手动添加[CLS]和[SEP]，让tokenizer来处理
        return padded

    def forward(self, mhc_sequences, peptide_sequences):
        # 组合序列
        combined_sequences = [
            self.combine_sequences(mhc, pep) 
            for mhc, pep in zip(mhc_sequences, peptide_sequences)
        ]
        
        # 获取BERT特征
        features = self.bert_model.get_last_hidden_state(combined_sequences)
        
        # 对BERT特征进行BN（需要调整维度）
        batch_size, seq_len, hidden_dim = features.shape
        features = features.reshape(-1, hidden_dim)
        features = self.bert_bn(features)
        features = features.reshape(batch_size, seq_len, hidden_dim)
        
        # BiGRU特征提取
        gru_out, _ = self.mhc_gru(features)
        
        # 获取最后一个时间步的隐藏状态
        final_features = torch.cat([
            gru_out[:, -1, :self.gru_hidden_dim],
            gru_out[:, 0, self.gru_hidden_dim:]
        ], dim=1)
        
        # 对GRU输出进行BN
        final_features = self.gru_bn(final_features)
        
        # 分类预测
        output = self.classifier(final_features)
        return output

    def set_bert_training_mode(self, trainable=True):
        """设置BERT层是否可训练"""
        for param in self.bert_model.parameters():
            param.requires_grad = trainable
        self.freeze_bert = not trainable
    
    def get_trainable_params(self):
        """获取需要训练的参数"""
        if self.freeze_bert:
            # 如果BERT被冻结，只返回其他层的参数
            return [p for n, p in self.named_parameters() if not n.startswith('bert_model')]
        else:
            # 如果BERT未被冻结，返回所有参数
            return self.parameters()

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
