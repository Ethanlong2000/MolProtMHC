import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import re
import os
import time  # 添加计时模块

# 定义分类器模型
class ProteinClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return logits

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("/home/longyh/software/prot_bert/", do_lower_case=False)
model = BertModel.from_pretrained("/home/longyh/software/prot_bert/")

# 实例化分类器模型
num_classes = 2  # 假设有两个类别
classifier = ProteinClassifier(model, num_classes)

# 读取训练数据
data = pd.read_csv('/work/longyh/mhc/testdata.csv')

# 准备输入数据和标签
data['seq'] = data['mhc_sequence'] + data['sequence']
sequences = data['seq'].apply(lambda x: re.sub(r"[UZOB]", "X", x))
labels = torch.tensor(data['binding300'].values)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-5)

# 定义训练轮数和早停机制参数
num_epochs = 10  # 设定训练轮数
early_stopping_patience = 3  # 设定早停机制的耐心值
best_loss = float('inf')
patience_counter = 0

# 训练模型
classifier.train()
start_time = time.time()  # 开始计时
for epoch in range(num_epochs):
    epoch_loss = 0
    for sequence, label in zip(sequences, labels):
        encoded_input = tokenizer(sequence, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        optimizer.zero_grad()
        logits = classifier(input_ids, attention_mask)
        loss = criterion(logits, label.unsqueeze(0))  # 需要将标签调整为与logits匹配的形状
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(labels)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

    # 早停机制
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

end_time = time.time()  # 结束计时
print(f"Training Time: {end_time - start_time} seconds")  # 输出训练时间

# 保存微调后的模和参数
output_dir = "/work/longyh/mhc/ptest_output"  # 指定保存模型的目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存模型
model_save_path = os.path.join(output_dir, "test_bert_model.pth")
torch.save(classifier.state_dict(), model_save_path)

# 保存分词器
tokenizer.save_pretrained(output_dir)

