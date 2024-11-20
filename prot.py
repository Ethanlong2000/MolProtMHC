from transformers import BertModel, BertTokenizer
import torch
import re

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("/home/longyh/software/prot_bert/", do_lower_case=False)
model = BertModel.from_pretrained("/home/longyh/software/prot_bert/").to(device)  # 将模型移动到 GPU

sequence_Example = "YSAMYQENVAHTDENTLYIIYEHYTWARLAYTWYEVQLVESGGGL"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt').to(device)  # 将输入数据移动到 GPU

# 获取模型输出
output = model(**encoded_input)

# 提取 last_hidden_state
last_hidden_state = output['last_hidden_state']

# 如果需要，可以将特征移回 CPU 进行进一步处理
last_hidden_state = last_hidden_state.cpu().detach().numpy()

print("Last Hidden State Shape:", last_hidden_state.shape)
print("Last Hidden State:", last_hidden_state)
