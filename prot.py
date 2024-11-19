from transformers import BertModel, BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained("/home/longyh/software/prot_bert/", do_lower_case=False)
model = BertModel.from_pretrained("/home/longyh/software/prot_bert/")
sequence_Example = "A E T C Z A O"
sequence_Example = "YYAMYREISTNTYESNLYLRYDSYTWAEWAYLWY"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')
output = model(**encoded_input)
print(output)