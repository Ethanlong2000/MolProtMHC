import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from transformers import BertModel, BertTokenizer
from prot_model import ProteinBertBiGRUClassifier
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    peptides = data['peptide'].tolist()
    HLA_sequences = data['HLA_sequence'].tolist()
    labels = data['label'].tolist()
    return peptides, HLA_sequences, labels

def evaluate_model(model, val_set, batch_size=64, device=None):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    val_peptides, val_HLA_sequences, val_labels = load_data(val_set)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(val_peptides), batch_size):
            batch_peptides = val_peptides[i:i + batch_size]
            batch_HLA_sequences = val_HLA_sequences[i:i + batch_size]
            batch_labels = val_labels[i:i + batch_size]
            
            outputs = model(batch_peptides, batch_HLA_sequences)
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels)
    
    return np.array(all_labels), np.array(all_preds)

def plot_roc_curve(labels, preds, output_dir):
    roc_auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_pr_curve(labels, preds, output_dir):
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

if __name__ == '__main__':
    model_path = "/home/longyh/software/prot_bert/"
    val_set = '/work/longyh/MolProtMHC/Data/val100k.csv'
    output_dir = "/work/longyh/MolProtMHC/eval_output"
    
    classifier = ProteinBertBiGRUClassifier(model_path, hidden_dim=256, output_dim=2, n_layers=2, bidirectional=True, dropout=0.5)
    classifier.load_state_dict(torch.load('/work/longyh/MolProtMHC/train_output/prot_output/model_latest.pt'))
    
    labels, preds = evaluate_model(classifier, val_set)
    
    plot_roc_curve(labels, preds, output_dir)
    plot_pr_curve(labels, preds, output_dir)
    
    print("Evaluation completed. ROC and PR curves saved.")
