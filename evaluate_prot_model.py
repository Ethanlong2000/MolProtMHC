import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)
from prot_model import MHCPeptideModel
from prot_model_train import MHCPeptideDataset
import logging
from tqdm import tqdm

class ModelEvaluator:
    def __init__(
        self,
        model_path,
        bert_model_path,
        test_data_path,
        output_dir,
        batch_size=256,
        device=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.bert_model_path = bert_model_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        
        # 从模型路径中提取模型信息
        model_name = os.path.basename(model_path)
        model_date = model_name.split('_')[2] if '_' in model_name else 'unknown'
        
        # 从测试数据路径中提取数据集信息
        test_set_name = os.path.basename(test_data_path).split('.')[0]
        
        # 创建更具描述性的输出目录名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            output_dir,
            f'eval_{test_set_name}_model{model_date}_{self.timestamp}'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 加载模型和数据
        self._load_model()
        self._load_data()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """加载预训练模型"""
        self.logger.info("Loading model...")
        self.model = MHCPeptideModel(self.bert_model_path, device=self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def _load_data(self):
        """加载测试数据"""
        self.logger.info("Loading test data...")
        test_df = pd.read_csv(self.test_data_path)
        test_dataset = MHCPeptideDataset(
            test_df['HLA_sequence'].values,
            test_df['peptide'].values,
            test_df['label'].values
        )
        # test_dataset = MHCPeptideDataset(
        # test_df['mhc_sequence'].values,
        # test_df['sequence'].values,
        # test_df['binding300'].values
        # )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
    def evaluate(self):
        """执行完整的评估"""
        self.logger.info("Starting evaluation...")
        
        # 收集预测结果
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                labels = batch['label'].to(self.device)
                outputs = self.model(batch['mhc_seq'], batch['peptide_seq'])
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
        
        # 转换为numpy数组
        self.labels = np.array(all_labels)
        self.preds = np.array(all_preds)
        
        # 计算各种指标
        self._calculate_metrics()
        
        # 绘制图表
        self._plot_curves()
        
        # 生成评估报告
        self._generate_report()
        
        self.logger.info(f"Evaluation completed. Results saved in: {self.output_dir}")
        
    def _calculate_metrics(self):
        """计算各种评估指标"""
        # 将预测概率转换为二进制标签
        pred_labels = (self.preds > 0.5).astype(int)
        
        # 基础指标
        self.metrics = {
            'accuracy': accuracy_score(self.labels, pred_labels),
            'precision': precision_score(self.labels, pred_labels),
            'recall': recall_score(self.labels, pred_labels),
            'f1': f1_score(self.labels, pred_labels),
            'auc_roc': roc_curve(self.labels, self.preds),
            'auc_score': auc(*roc_curve(self.labels, self.preds)[:2]),
            'precision_recall': precision_recall_curve(self.labels, self.preds),
            'auc_pr': average_precision_score(self.labels, self.preds)
        }
        
    def _plot_curves(self):
        """绘制ROC曲线和PR曲线"""
        # ROC曲线
        plt.figure(figsize=(10, 5))
        fpr, tpr, _ = self.metrics['auc_roc']
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["auc_score"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {os.path.basename(self.test_data_path)}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(
            self.output_dir, 
            f'roc_curve_{os.path.basename(self.test_data_path)}_{self.timestamp}.png'
        ))
        plt.close()
        
        # PR曲线
        plt.figure(figsize=(10, 5))
        precision, recall, _ = self.metrics['precision_recall']
        plt.plot(recall, precision, label=f'PR curve (AP = {self.metrics["auc_pr"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {os.path.basename(self.test_data_path)}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(
            self.output_dir, 
            f'pr_curve_{os.path.basename(self.test_data_path)}_{self.timestamp}.png'
        ))
        plt.close()
        
    def _generate_report(self):
        """生成评估报告"""
        report = (
            "Model Evaluation Report\n"
            "=====================\n\n"
            f"Evaluation Time: {self.timestamp}\n"
            f"Model Path: {self.model_path}\n"
            f"Model Date: {os.path.basename(self.model_path).split('_')[2]}\n"
            f"Test Data: {self.test_data_path}\n"
            f"Test Set Name: {os.path.basename(self.test_data_path).split('.')[0]}\n\n"
            "Metrics:\n"
            f"- Accuracy: {self.metrics['accuracy']:.4f}\n"
            f"- Precision: {self.metrics['precision']:.4f}\n"
            f"- Recall: {self.metrics['recall']:.4f}\n"
            f"- F1 Score: {self.metrics['f1']:.4f}\n"
            f"- ROC AUC: {self.metrics['auc_score']:.4f}\n"
            f"- PR AUC: {self.metrics['auc_pr']:.4f}\n"
        )
        
        # 保存报告时使用更具描述性的文件名
        report_filename = f'evaluation_report_{os.path.basename(self.test_data_path)}_{self.timestamp}.txt'
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        # 同时输出到日志
        self.logger.info("\n" + report)

def main(config):
    evaluator = ModelEvaluator(**config)
    evaluator.evaluate()

if __name__ == '__main__':
    # 评估配置示例
    config = {
        'model_path': '/work/longyh/mhc/train_output/prot_output/run_20250107_133441/best_model_20250107_133441.pth',  # 需要填写训练好的模型路径
        'bert_model_path': '/home/longyh/software/prot_bert/',
        
        'test_data_path': '/work/longyh/mhc/Data/val_data_fold2.csv',  # 需要填写测试集路径
        
        'output_dir': '/work/longyh/mhc/evaluation_output',
        'batch_size': 2048
    }
    
    main(config)
