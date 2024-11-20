import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    # 选取 df 的 sequence 和 mhc_sequence 列拼接成新的列（smiles），并应用 peptide_to_smiles 函数
    df['combine'] = df['sequence'] + df['mhc_sequence']
    df['smiles'] = df['combine'].apply(peptide_to_smiles)
    
    return df
def peptide_to_smiles(peptide_sequence):
    mol = AllChem.MolFromSequence(peptide_sequence)
    return Chem.MolToSmiles(mol)
def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)



if __name__ == "__main__":

    
    input_file = "/work/longyh/MolProtMHC/Data/test1k.csv"
    
    df = process_csv_file(input_file)
    df['canonical_smiles'] = df['smiles'].apply(canonicalize)
    df.to_csv(input_file, index=False)

