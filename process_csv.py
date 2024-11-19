import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    # 选取 df 的 sequence 和 mhc_sequence 列拼接成新的列（smiles），并应用 peptide_to_smiles 函数
    df['smiles'] = df['sequence'] + df['mhc_sequence']
    df['smiles'] = df['smiles'].apply(peptide_to_smiles)
    return df

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

def peptide_to_smiles(peptide_sequence):
    mol = AllChem.MolFromSequence(peptide_sequence)
    return Chem.MolToSmiles(mol)