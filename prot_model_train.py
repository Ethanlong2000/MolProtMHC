import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from prot_model import MHCPeptideModel, MHCPeptideTrainer

class MHCPeptideDataset(Dataset):
    def __init__(self, mhc_sequences, peptide_sequences, labels):
        self.mhc_sequences = mhc_sequences
        self.peptide_sequences = peptide_sequences
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)
    
    def __len__(self):
        return len(self.mhc_sequences)
    
    def __getitem__(self, idx):
        return {
            'mhc_seq': self.mhc_sequences[idx],
            'peptide_seq': self.peptide_sequences[idx],
            'label': self.labels[idx]
        }

def plot_metrics(train_losses, val_losses, val_accuracies, save_dir, timestamp):
    """绘制训练过程中的指标"""
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_curve_{timestamp}.png'))
    plt.close()
    
    # 绘制accuracy曲线
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'accuracy_curve_{timestamp}.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred, save_dir, timestamp):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'roc_curve_{timestamp}.png'))
    plt.close()
    
    return roc_auc

def train_model(
    train_data_path,
    val_data_path,
    bert_model_path,
    output_dir,
    epochs=50,
    batch_size=256,
    learning_rate=1e-4,
    weight_decay=1e-5,
    seed=42
):
    # 在函数开始处定义device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建带时间戳的输出目录
    model_output_dir = os.path.join(output_dir, f'run_{timestamp}')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(model_output_dir, f'training_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 读取训练集和验证集
    logger.info("Loading datasets...")
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    logger.info(f"数据列名: {train_df.columns.tolist()}")
    
    # 创建数据集（不指定device）
    train_dataset = MHCPeptideDataset(
        train_df['HLA_sequence'].values,
        train_df['peptide'].values,
        train_df['label'].values
    )
    
    val_dataset = MHCPeptideDataset(
        val_df['HLA_sequence'].values,
        val_df['peptide'].values,
        val_df['label'].values
    )
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 初始化模型和训练器
    logger.info(f"Initializing model from {bert_model_path}...")
    model = MHCPeptideModel(bert_model_path, device=device)
    trainer = MHCPeptideTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device  # 传入device
    )
    
    # 记录训练过程的指标
    train_losses = []
    val_losses = []
    val_accuracies = []
    all_val_labels = []
    all_val_preds = []
    
    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_losses = []
        train_loop = tqdm(train_loader, 
                         desc=f'Epoch {epoch+1}/{epochs} [Train]',
                         ncols=100)  # 设置进度条宽度
        
        for batch in train_loop:
            # 在这里将数据移到GPU
            labels = batch['label'].to(device)
            loss = trainer.train_step(
                batch['mhc_seq'],
                batch['peptide_seq'],
                labels
            )
            epoch_train_losses.append(loss)
            train_loop.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{trainer.optimizer.param_groups[0]["lr"]:.2e}'  # 显示当前学习率
            })
        
        # 验证阶段
        model.eval()
        epoch_val_losses = []
        epoch_val_accuracies = []
        epoch_val_labels = []
        epoch_val_preds = []
        
        val_loop = tqdm(val_loader, 
                       desc=f'Epoch {epoch+1}/{epochs} [Val]',
                       ncols=100)
        
        for batch in val_loop:
            with torch.no_grad():
                # 将数据移到GPU
                labels = batch['label'].to(device)
                
                outputs = model(batch['mhc_seq'], batch['peptide_seq'])
                loss = trainer.criterion(outputs, labels)  # 现在labels在GPU上
                
                # 在CPU上进行指标计算
                epoch_val_labels.extend(batch['label'].cpu().numpy())
                epoch_val_preds.extend(outputs.cpu().numpy())
                
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == labels).float().mean()  # labels已经在GPU上
                
                epoch_val_losses.append(loss.item())
                epoch_val_accuracies.append(accuracy.item())
                
                val_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy.item():.4f}'
                })
        
        # 计算平均指标
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        avg_val_accuracy = np.mean(epoch_val_accuracies)
        
        # 记录指标
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        all_val_labels.extend(epoch_val_labels)
        all_val_preds.extend(epoch_val_preds)
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 记录日志
        logger.info(
            f'Epoch {epoch+1}/{epochs} - '
            f'Train Loss: {avg_train_loss:.4f} - '
            f'Val Loss: {avg_val_loss:.4f} - '
            f'Val Accuracy: {avg_val_accuracy:.4f} - '
            f'LR: {trainer.optimizer.param_groups[0]["lr"]:.2e}'
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(model_output_dir, f'best_model_{timestamp}.pth')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                },
                model_save_path
            )
            logger.info(f'Saved best model to {model_save_path} with validation loss: {best_val_loss:.4f}')
    
    # 训练结束后绘制指标图表
    logger.info("Plotting training metrics...")
    plot_metrics(train_losses, val_losses, val_accuracies, model_output_dir, timestamp)
    roc_auc = plot_roc_curve(all_val_labels, all_val_preds, model_output_dir, timestamp)
    
    logger.info(f'Training completed. Final ROC AUC: {roc_auc:.4f}')
    logger.info(f'All results saved in: {model_output_dir}')

if __name__ == '__main__':
    # 训练参数
    config = {
        'train_data_path': '/work/longyh/mhc/Data/train100k_0.csv',
        'val_data_path': '/work/longyh/mhc/Data/val100k_0.csv',
        'bert_model_path': '/home/longyh/software/prot_bert/',
        'output_dir': '/work/longyh/mhc/train_output/prot_output',
        'epochs': 50,
        'batch_size': 512,
        'learning_rate': 2e-4,
        'weight_decay': 1e-5,
        'seed': 42
    }
    
    # 开始训练
    train_model(**config) 