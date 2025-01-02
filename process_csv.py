import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
import os

def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    # 选取 df 的 sequence 和 mhc_sequence 列拼接成新的列（smiles），并应用 peptide_to_smiles 函数
    with ProcessPoolExecutor() as executor:
        df['smiles_peptide'] = list(tqdm(executor.map(peptide_to_smiles, df['peptide']), total=len(df), desc="Processing peptide"))
        df['smiles_HLA'] = list(tqdm(executor.map(peptide_to_smiles, df['HLA_sequence']), total=len(df), desc="Processing HLA"))
    return df

def peptide_to_smiles(peptide_sequence):
    mol = AllChem.MolFromSequence(peptide_sequence)
    return Chem.MolToSmiles(mol)

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

def process_and_save_csv(input_file, dir):
    print(f"Processing {input_file}")
    df = process_csv_file(os.path.join(dir, input_file))
    with ProcessPoolExecutor() as executor:
        df['smiles_peptide'] = list(tqdm(executor.map(canonicalize, df['smiles_peptide']), total=len(df), desc="Processing peptide canonical"))
        df['smiles_HLA'] = list(tqdm(executor.map(canonicalize, df['smiles_HLA']), total=len(df), desc="Processing HLA canonical"))
    df.to_csv(os.path.join(dir, input_file), index=False)

if __name__ == "__main__":
    dir = '/work/longyh/mhc'
    input_files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    for input_file in input_files:
        process_and_save_csv(input_file, dir)





