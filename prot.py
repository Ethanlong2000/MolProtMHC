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
    
    def preprocess_sequences(self, sequences, batch_size=128):
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            encoded_inputs = self.tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True).to(self.device)
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

    def parameters(self):
        return self.model.parameters()

# 示例用法
if __name__ == "__main__": 
    model_path = "/home/longyh/software/prot_bert/"
    csv_file = "/work/longyh/MolProtMHC/Data/test1k.csv"
    column_name = "combine"
    
    protein_bert_model = ProteinBertModel(model_path)
    sequences = protein_bert_model.process_csv(csv_file, column_name)
    
    last_hidden_states = protein_bert_model.get_last_hidden_state(sequences)
    pooler_outputs = protein_bert_model.get_pooler_output(sequences)
    
    print("Last Hidden States Shape:", last_hidden_states.shape)
    print("Last Hidden States:", last_hidden_states)
    print("Pooler Outputs Shape:", pooler_outputs.shape)
    print("Pooler Outputs:", pooler_outputs)
