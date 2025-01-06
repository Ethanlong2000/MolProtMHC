import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import yaml
import argparse
import os
import time
from transformers import BertModel, BertTokenizer
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ProteinBertModel:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
    
    def preprocess_sequence(self, peptide, HLA_sequence, max_length=51):
        sequence = f"{self.tokenizer.cls_token}{peptide}{self.tokenizer.sep_token}{HLA_sequence}"
        encoded_input = self.tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(self.device)
        return encoded_input
    
    def preprocess_sequences(self, peptides, HLA_sequences, batch_size=64, max_length=51):
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(peptides), batch_size):
            batch_peptides = peptides[i:i + batch_size]
            batch_HLA_sequences = HLA_sequences[i:i + batch_size]
            batch_sequences = [f"{self.tokenizer.cls_token}{pep}{self.tokenizer.sep_token}{hla}" for pep, hla in zip(batch_peptides, batch_HLA_sequences)]
            encoded_inputs = self.tokenizer(batch_sequences, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(self.device)
            all_input_ids.append(encoded_inputs['input_ids'])
            all_attention_masks.append(encoded_inputs['attention_mask'])
        
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def get_last_hidden_state(self, sequences):
        encoded_inputs = self.preprocess_sequences(sequences)
        output = self.model(**encoded_inputs)
        last_hidden_state = output['last_hidden_state']
        return last_hidden_state.detach()
    
    def get_pooler_output(self, sequences):
        encoded_inputs = self.preprocess_sequences(sequences)
        output = self.model(**encoded_inputs)
        pooler_output = output['pooler_output']
        return pooler_output.detach()
    
    def parameters(self):
        return self.model.parameters()

class ProteinBertBiGRUClassifier(nn.Module):
    def __init__(self, model_path, hidden_dim, output_dim, n_layers, bidirectional, dropout, device=None):
        super(ProteinBertBiGRUClassifier, self).__init__()
        self.bert_model = ProteinBertModel(model_path, device)
        self.gru = nn.GRU(self.bert_model.model.config.hidden_size, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, peptides, HLA_sequences):
        encoded_inputs = self.bert_model.preprocess_sequences(peptides, HLA_sequences)
        with torch.no_grad():
            bert_outputs = self.bert_model.model(**encoded_inputs)
        hidden_states = bert_outputs.last_hidden_state
        gru_output, _ = self.gru(hidden_states)
        pooled_output = torch.cat((gru_output[:, -1, :self.gru.hidden_size], gru_output[:, 0, self.gru.hidden_size:]), dim=1)
        output = self.fc(self.dropout(pooled_output))
        return output

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_params = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_params = model.state_dict()
        self.val_loss_min = val_loss

    def load_best_model(self, model):
        if self.best_model_params:
            model.load_state_dict(self.best_model_params)

class Trainer:
    def __init__(self, model, train_set, val_set, epochs=10, batch_size=512, learning_rate=1e-5, device=None):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5, verbose=True)
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        peptides = data['peptide'].tolist()
        HLA_sequences = data['HLA_sequence'].tolist()
        labels = data['label'].tolist()
        return peptides, HLA_sequences, labels
    
    def plot_metrics(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_metrics_{timestamp}.png"))
        plt.close()
    
    def save_model(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(output_dir, f"model_{timestamp}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def train(self):
        train_peptides, train_HLA_sequences, train_labels = self.load_data(self.train_set)
        val_peptides, val_HLA_sequences, val_labels = self.load_data(self.val_set)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(train_peptides), self.batch_size):
                batch_peptides = train_peptides[i:i + self.batch_size]
                batch_HLA_sequences = train_HLA_sequences[i:i + self.batch_size]
                batch_labels = torch.tensor(train_labels[i:i + self.batch_size], dtype=torch.long).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_peptides, batch_HLA_sequences)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_peptides)
            self.train_losses.append(avg_train_loss)
            
            self.model.eval()
            total_val_loss = 0
            correct = 0
            with torch.no_grad():
                for i in range(0, len(val_peptides), self.batch_size):
                    batch_peptides = val_peptides[i:i + self.batch_size]
                    batch_HLA_sequences = val_HLA_sequences[i:i + self.batch_size]
                    batch_labels = torch.tensor(val_labels[i:i + self.batch_size], dtype=torch.long).to(self.device)
                    
                    outputs = self.model(batch_peptides, batch_HLA_sequences)
                    loss = self.criterion(outputs, batch_labels)
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == batch_labels).sum().item()
            
            avg_val_loss = total_val_loss / len(val_peptides)
            val_accuracy = correct / len(val_peptides)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)
            
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        self.plot_metrics()
        self.save_model()


model_path = "/home/longyh/software/prot_bert/"

train_set='/work/longyh/MolProtMHC/Data/train100k.csv'
val_set='/work/longyh/MolProtMHC/Data/val100k.csv'

output_dir = "/work/longyh/MolProtMHC/train_output/prot_output"

if __name__=='__main__':
    #当前时间
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Example usage:
    classifier = ProteinBertBiGRUClassifier(model_path, hidden_dim=256, output_dim=2, n_layers=2, bidirectional=True, dropout=0.5)
    trainer = Trainer(classifier, train_set, val_set, epochs=20, batch_size=64, learning_rate=1e-4)
    trainer.train()
    #结束时间
    timestamp2 = time.strftime("%Y%m%d-%H%M%S")
    #耗费时间
    print('time:', time.mktime(time.strptime(timestamp2)) - time.mktime(time.strptime(timestamp)))
