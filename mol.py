import yaml
import torch
from tqdm import tqdm  # 添加进度条库

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from argparse import Namespace
from fast_transformers.masking import LengthMask as LM

from lightModule import LightningModule,MolTranBertTokenizer
import process_csv


class MolFormer:
    def __init__(self, config_path, ckpt_path, vocab_path, device='cuda'):
        with open(config_path, 'r') as f:
            self.config = Namespace(**yaml.safe_load(f))
        
        self.tokenizer = MolTranBertTokenizer(vocab_path)
        self.lm = LightningModule(self.config, self.tokenizer.vocab).load_from_checkpoint(ckpt_path, config=self.config, vocab=self.tokenizer.vocab)
        self.device = torch.device(device)
        self.lm.to(self.device)
        print(f"Using device: {self.device}")

    def batch_split(self, data, batch_size=64):
        i = 0
        while i < len(data):
            yield data[i:min(i+batch_size, len(data))]
            i += batch_size
    def embed(self, smiles, batch_size=64):
        self.lm.eval()
        embeddings = []
        # 使用tqdm包裹batch_split生成器以显示进度条
        for batch in tqdm(self.batch_split(smiles, batch_size=batch_size), total=(len(smiles) + batch_size - 1) // batch_size, desc="Embedding batches"):
            batch_enc = self.tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
            idx, mask = torch.tensor(batch_enc['input_ids']).to(self.device), torch.tensor(batch_enc['attention_mask']).to(self.device)
            with torch.no_grad():
                token_embeddings = self.lm.blocks(self.lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            # average pooling over tokens
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach())
        return torch.cat(embeddings)
    
    # def embed(self, smiles, batch_size=64):
    #     self.lm.eval()
    #     embeddings = []
    #     for batch in self.batch_split(smiles, batch_size=batch_size):
    #         batch_enc = self.tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
    #         idx, mask = torch.tensor(batch_enc['input_ids']).to(self.device), torch.tensor(batch_enc['attention_mask']).to(self.device)
    #         with torch.no_grad():
    #             token_embeddings = self.lm.blocks(self.lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))
    #         # average pooling over tokens
    #         input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    #         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    #         embedding = sum_embeddings / sum_mask
    #         embeddings.append(embedding.detach())
    #     return torch.cat(embeddings)

    # def canonicalize(self, s):
    #     return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

    # def peptide_to_smiles(self, peptide_sequence):
    #     mol = AllChem.MolFromSequence(peptide_sequence)
    #     return Chem.MolToSmiles(mol)

    # def parameters(self):
    #     return self.lm.parameters()

# datapath='./Data/test1k.csv'
# config_path = './Data/mol/hparams.yaml'
# ckpt_path = '/home/longyh/database/molformer/PretrainedMoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
# vocab_path = './Data/mol/bert_vocab.txt'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Selected device: {device}") 

# molformer = MolFormer(config_path, ckpt_path, vocab_path, device=device)


# # 调用模型用下面的代码
# df = process_csv.process_csv_file(datapath)
# smiles = df.smiles.apply(process_csv.canonicalize)

# # 训练模型时省略处理smiles的步骤(csv文件中已经处理好)
# df=pd.read_csv(datapath)
# smiles = df.canonical_smiles

# start_time = time.time()
# X = molformer.embed(smiles).cpu().numpy()
# end_time = time.time()
# input("ENTER...")  # ���加输入提示
# print(f"Embedding time: {end_time - start_time} seconds")
# print(X.shape)


