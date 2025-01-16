import io
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

def smiles_to_mol(smiles):
    # https://www.rdkit.org/docs/GettingStartedInPython.html#writing-molecules
    # https://github.com/rdkit/rdkit/issues/2996
    try:
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)#
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)##
        AllChem.MMFFOptimizeMolecule(mol)
        # AllChem.UFFOptimizeMolecule(mol)
    except:
        mol = None
    return mol

def sdf_to_mol(sdf_bin):
    try:
        mol = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_bin), sanitize=False))
    except:
        mol = None
    return mol

def sdf_to_smiles(sdf_bin):
    try:
        mol = next(Chem.ForwardSDMolSupplier(io.BytesIO(sdf_bin), sanitize=False))
        smi = Chem.MolToSmiles(mol, canonical=False)
    except:
        smi = None
    return smi

def mol_to_smiles(mol):
    try:
        smi = Chem.MolToSmiles(mol, canonical=False)
    except:
        smi = None
    return smi
