import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.rdBase import BlockLogs

sys.path.append(str(Path(Chem.RDConfig.RDContribDir)/ 'SA_Score'))
from sascorer import calculateScore  # pyright: ignore[reportMissingImports]

# from .sascorer import calculateScore

class QedSa():
    def __init__(self, smiles=None, rdmol=None) -> None:
        block = BlockLogs()
        try:
            if smiles:
                rdmol = Chem.MolFromSmiles(smiles)
            self.rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
        except:
            self.rdmol = None
        del block

    def qed(self) -> float:
        block = BlockLogs()
        try:
            assert self.rdmol
            _qed = Descriptors.qed(self.rdmol)
        except:
            _qed = None
        del block
        return _qed

    def sa(self, is_norm=True) -> float:
        block = BlockLogs()
        try:
            assert self.rdmol
            _sa = calculateScore(self.rdmol)
            if is_norm:
                _sa = round((10 - _sa) / 9, 2)
        except:
            _sa = None
        del block
        return _sa

    def lipinski(self, n_rules=5, return_rules=False):
        block = BlockLogs()
        try:
            assert self.rdmol
            rules = (
                Descriptors.ExactMolWt(self.rdmol) < 500,
                Lipinski.NumHDonors(self.rdmol) <= 5,
                Lipinski.NumHAcceptors(self.rdmol) <= 10,
                -2 <= Crippen.MolLogP(self.rdmol) <= 5,
                Chem.rdMolDescriptors.CalcNumRotatableBonds(self.rdmol) <= 10
            )
            if n_rules == 5:
                _lipinski = sum(rules)
            elif n_rules == 4:
                _lipinski = sum((rules[0], rules[1], rules[2], rules[3]))
            elif n_rules == 3:
                _lipinski = sum((rules[0], rules[1], rules[3]))
            else:
                _lipinski = None
        except:
            _lipinski = None
            rules = None
        del block

        if return_rules:
            return _lipinski, rules
        else:
            return _lipinski

    def logp(self) -> float:
        block = BlockLogs()
        try:
            assert self.rdmol
            logp = Crippen.MolLogP(self.rdmol)
        except:
            logp = None
        del block
        return logp
        