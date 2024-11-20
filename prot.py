from transformers import BertModel, BertTokenizer
import torch
import re
import pandas as pd

class ProteinBertModel:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
    
    def preprocess_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", sequence)
        encoded_input = self.tokenizer(sequence, return_tensors='pt').to(self.device)
        return encoded_input
    
    def get_last_hidden_state(self, sequence):
        encoded_input = self.preprocess_sequence(sequence)
        output = self.model(**encoded_input)
        last_hidden_state = output['last_hidden_state']
        return last_hidden_state.cpu().detach().numpy()
    
    def get_pooler_output(self, sequence):
        encoded_input = self.preprocess_sequence(sequence)
        output = self.model(**encoded_input)
        pooler_output = output['pooler_output']
        return pooler_output.cpu().detach().numpy()
    
    def fine_tune(self, train_dataloader, num_epochs=3, learning_rate=1e-5):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                pooled_output = outputs['pooler_output']
                
                # 假设有一个分类层
                logits = self.classifier(pooled_output)
                
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()
                
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def process_csv(self, csv_file, column_name):
        df = pd.read_csv(csv_file)
        sequences = df[column_name].tolist()
        return sequences

# 示例用法
if __name__ == "__main__": 
    model_path = "/home/longyh/software/prot_bert/"
    csv_file = "path/to/your/csvfile.csv"
    column_name = "sequence_column"
    
    protein_bert_model = ProteinBertModel(model_path)
    sequences = protein_bert_model.process_csv(csv_file, column_name)
    
    for sequence in sequences:
        last_hidden_state = protein_bert_model.get_last_hidden_state(sequence)
        pooler_output = protein_bert_model.get_pooler_output(sequence)
        
        print("Last Hidden State Shape:", last_hidden_state.shape)
        print("Last Hidden State:", last_hidden_state)
        print("Pooler Output Shape:", pooler_output.shape)
        print("Pooler Output:", pooler_output)
