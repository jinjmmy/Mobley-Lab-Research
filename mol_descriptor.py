import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

class MolecularDescriptorCalculator:
    def __init__(self, sdf_file=None, csv_file=None):
        self.sdf_file = sdf_file
        self.csv_file = csv_file

    def calculate_ecfp4_from_sdf(self):
        """Calculates ECFP4 fingerprints from an SDF file."""
        suppl = Chem.SDMolSupplier(self.sdf_file)
        names_list = []
        smiles_list = []
        ecfp4_list = []
        
        for mol in suppl:
            if mol is not None:
                name = mol.GetProp('_Name')
                smiles = Chem.MolToSmiles(mol)
                ecfp4_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
                
                names_list.append(name)
                smiles_list.append(smiles)
                ecfp4_list.append(ecfp4_fingerprint)
        
        ecfp4_binary_digits = [list(fingerprint.ToBitString()) for fingerprint in ecfp4_list]
        return names_list, smiles_list, ecfp4_binary_digits

    def calculate_ecfp4_from_csv(self):
        """Calculates ECFP4 fingerprints from a CSV file containing SMILES."""
        smiles_data = {'Name': [], 'SMILES': []}
        with open(self.csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                smiles_data['Name'].append(row[0])
                smiles_data['SMILES'].append(row[1])
        
        ecfp4_list = self.smiles_to_ecfp4(smiles_data['SMILES'])
        ecfp4_binary_digits = [list(fingerprint.ToBitString()) for fingerprint in ecfp4_list]
        return smiles_data['Name'], smiles_data['SMILES'], ecfp4_binary_digits
    
    def smiles_to_ecfp4(self, smiles_list, radius=3, nBits=2048):
        """Calculates ECFP4 fingerprints for a list of SMILES strings."""
        ecfp4_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                ecfp4_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                ecfp4_list.append(ecfp4_fingerprint)
        return ecfp4_list
    
    def filter_docked_cmpds(self, file1, file2, docked_output_file, unsuccessful_output_file):
        """Filters docked compounds and saves them to separate files."""
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        docked_compounds = pd.merge(df1[['Name']], df2, on='Name', how='inner')
        unsuccessful_compounds = df2[~df2['Name'].isin(docked_compounds['Name'])]
        
        docked_compounds.to_csv(docked_output_file, index=False)
        unsuccessful_compounds.to_csv(unsuccessful_output_file, index=False)

    def create_binders_file(self, docking_file, output_file, top_n=2000):
        """Creates a file with the top N compounds based on docking scores."""
        data = pd.read_csv(docking_file)
        sorted_data = data.sort_values(by='Score', ascending=True)
        binders = sorted_data.head(top_n)
        binders.to_csv(output_file, index=False)
        print(f'Active compound list saved to {output_file}')
