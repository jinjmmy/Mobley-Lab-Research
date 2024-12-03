import time
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

class CompoundDescriptorProcessor:
    def __init__(self, sdf_file_path=None, smiles_file_path=None):
        self.sdf_file_path = sdf_file_path
        self.smiles_file_path = smiles_file_path
    
    def count_headers(self, csv_file):
        """Counts the number of headers in a CSV file."""
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            first_row = next(reader, None)
            if first_row:
                return len(first_row)
            return 0

    def check_row_lengths(self, csv_file):
        """Checks if all rows in a CSV file have the same length as headers."""
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader, None)
            if headers is None:
                return False
            num_headers = len(headers)
            for row in reader:
                if len(row) != num_headers:
                    return False
        return True

    def smiles_to_ecfp4(self, smiles_list, radius=3, nBits=2048):
        """Converts SMILES list to ECFP4 fingerprints."""
        ecfp4_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                ecfp4_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                ecfp4_list.append(ecfp4_fingerprint)
        return ecfp4_list
    
    def process_sdf(self, output_file_path='all_ecfp4.csv'):
        """Process SDF file to calculate ECFP4 fingerprints and save to CSV."""
        suppl = Chem.SDMolSupplier(self.sdf_file_path)
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

        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Name", "SMILES"] + [f"Morgan_bit{i}" for i in range(2048)]
            writer.writerow(header)
            for name, smiles, fingerprint in zip(names_list, smiles_list, ecfp4_binary_digits):
                writer.writerow([name, smiles] + fingerprint)

        print(f"ECFP4 fingerprints, names, and SMILES saved to {output_file_path}")

    def process_smiles_file(self, input_file_path, output_file_path='output_ecfp4.csv'):
        """Process a CSV file with SMILES strings to calculate ECFP4 fingerprints and save to CSV."""
        smiles_data = {'Name': [], 'SMILES': []}
        with open(input_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                smiles_data['Name'].append(row[0])
                smiles_data['SMILES'].append(row[1])

        ecfp4_list = self.smiles_to_ecfp4(smiles_data['SMILES'])
        ecfp4_binary_digits = [list(fingerprint.ToBitString()) for fingerprint in ecfp4_list]

        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Name", "SMILES"] + [f"Morgan_bit{i}" for i in range(2048)]
            writer.writerow(header)
            for name, smiles, fingerprint in zip(smiles_data['Name'], smiles_data['SMILES'], ecfp4_binary_digits):
                writer.writerow([name, smiles] + fingerprint)

        print(f"ECFP4 fingerprints, names, and SMILES saved to {output_file_path}")
    
    def filter_docked_cmpds(self, file1, file2, docked_output_file, unsuccessful_output_file):
        """Filters docked compounds and saves them to separate files."""
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        docked_compounds = pd.merge(df1[['Name']], df2, on='Name', how='inner')
        unsuccessful_compounds = df2[~df2['Name'].isin(docked_compounds['Name'])]

        docked_compounds.to_csv(docked_output_file, index=False)
        unsuccessful_compounds.to_csv(unsuccessful_output_file, index=False)
    
    def create_binders_file(self, csv_file_path, output_csv_path='binders_docking.csv'):
        """Creates a binders/actives file with the top 2000 compounds based on most negative docking scores."""
        data = pd.read_csv(csv_file_path)
        sorted_data = data.sort_values(by='Score', ascending=True)
        binders = sorted_data.head(2000)
        binders.to_csv(output_csv_path, index=False)
        print(f'Active compound list saved to {output_csv_path}')
