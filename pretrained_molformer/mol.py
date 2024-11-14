from argparse import Namespace
import yaml
import torch
from fast_transformers.masking import LengthMask as LM
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule

class MolFormer:
    def __init__(self, config_path, ckpt_path, vocab_path):
        with open(config_path, 'r') as f:
            self.config = Namespace(**yaml.safe_load(f))
        
        self.tokenizer = MolTranBertTokenizer(vocab_path)
        self.lm = LightningModule(self.config, self.tokenizer.vocab).load_from_checkpoint(ckpt_path, config=self.config, vocab=self.tokenizer.vocab)
    
    def batch_split(self, data, batch_size=64):
        i = 0
        while i < len(data):
            yield data[i:min(i+batch_size, len(data))]
            i += batch_size

    def embed(self, smiles, batch_size=64):
        self.lm.eval()
        embeddings = []
        for batch in self.batch_split(smiles, batch_size=batch_size):
            batch_enc = self.tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
            idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
            with torch.no_grad():
                token_embeddings = self.lm.blocks(self.lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            # average pooling over tokens
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())
        return torch.cat(embeddings)

    def canonicalize(self, s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    def peptide_to_smiles(self, peptide_sequence):
        mol = AllChem.MolFromSequence(peptide_sequence)
        return Chem.MolToSmiles(mol)

# 示例用法
config_path = '/home/longyh/database/molformer/PretrainedMoLFormer/hparams.yaml'
ckpt_path = '/home/longyh/database/molformer/PretrainedMoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
vocab_path = 'bert_vocab.txt'

molformer = MolFormer(config_path, ckpt_path, vocab_path)

df = pd.read_csv('/work/longyh/mhc/testdata.csv')

# 选取 df 的 sequence 和 mhc_sequence 列拼接成新的列（smiles），并应用 peptide_to_smiles 函数
df['smiles'] = df['sequence'] + df['mhc_sequence']
df['smiles'] = df['smiles'].apply(molformer.peptide_to_smiles)

smiles = df.smiles.apply(molformer.canonicalize)

X = molformer.embed(smiles).numpy()
# y = df.Class
y = df.binding300

print(X.shape, y.shape)
input("按空格键继续...")  # 添加输入提示
print(X)
input("按空格键继续...")  # 添加输入提示
print(y)

#torchmetrics-1.5.2
#numpy-1.24.4
#PyYAML-6.0.2;botocore-1.35.58;mmcv;opencv
#pip install  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
#pydantic-1.10.19-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
