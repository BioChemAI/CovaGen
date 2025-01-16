import os
import subprocess
import random
import string
import tempfile
import shutil
import time
import traceback
from easydict import EasyDict
from rdkit import Chem, RDLogger
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from openbabel import pybel

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def load_pdb(path):
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path):
    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            print('attention: mol is none')
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
        }))

    return results

def parse_pdbqt_outputs(docked_pdbqt_path):
    suppl = pybel.readfile('pdbqt',docked_pdbqt_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.data['REMARK'].splitlines()[0].split()[2:]
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
        }))

    return results


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)
    @classmethod
    def from_input_data(cls, ligand_path, protein_path, **kwargs):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, conda_env='adt', tmp_dir='./tmp', use_uff=True, center=None, size=None):
        super().__init__(pdb_block, ligand_rdmol)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(os.path.join(tmp_dir, str(os.getpid()) + '_' + get_random_id()))
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        # ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        # if use_uff:
        #     UFFOptimizeMolecule(ligand_rdmol) # TODO:2.1 oped
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        if center is None:
            pos = ligand_rdmol.GetConformer(0).GetPositions()
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center
        if size is None:
            self.size = [20,20,20]
        else:
            self.size = size


        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None

    def run(self, exhaustiveness=32, cpu=1, seed=20221023):#
        commands = """
# eval "$(conda shell.bash hook)"
# conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
/workspace/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor -r {receptor_id}.pdb -o {receptor_id}.pdbqt -A 'hydrogen'
# Prepare ligand    
obabel {ligand_id}.sdf -O {ligand_id}.pdbqt
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x {size_x:.4f} --size_y {size_y:.4f} --size_z {size_z:.4f} \
    --exhaustiveness {exhaust} \
    --cpu {cpu} \
    --seed {seed}
obabel {ligand_id}_out.pdbqt -O {ligand_id}_out.sdf
        """.format(
            receptor_id=self.receptor_id,
            ligand_id=self.ligand_id,
            env=self.conda_env,
            tmp=self.tmp_dir,
            # exhaust=exhaustiveness,
            exhaust=32,
            center_x=self.center[0],
            center_y=self.center[1],
            center_z=self.center[2],#
            size_x=self.size[0],
            size_y=self.size[1],
            size_z=self.size[2],
            cpu=cpu,
            seed=seed
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)
        self.docked_pdbqt_path = os.path.join(self.tmp_dir, '%s_out.pdbqt' % self.ligand_id)

        self.proc = subprocess.Popen(
            '/bin/bash',
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))
        self.proc.stdin.close()

        # return commands

    def run_sync(self, **kwargs):
        self.run(**kwargs)
        while self.get_results() is None:
            time.sleep(0.1)
        results = self.get_results()
        # print('Best affinity:', results[0]['affinity'])
        return results

    def get_results(self):
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                # try:
                #     self.results = parse_qvina_outputs(self.docked_sdf_path)
                # except Exception as error:
                #     print('[Error] Vina output error: %s' % self.docked_sdf_path)
                #     print('[Error] Vina output error: %s' % error)
                #     return []
                try:
                    self.results = parse_pdbqt_outputs(self.docked_pdbqt_path)
                except Exception as error:
                    print('[Error] Vina output error: %s' % self.docked_pdbqt_path)
                    print('[Error] Vina output error: %s' % error)
                    return []
            return self.results

    def clean(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

def docking_score(receptor_bin, ligand_mol, box=None, use_uff=False, _hash=None, **kwargs) -> float:
    try:
        pybel.ob.obErrorLog.StopLogging()
        RDLogger.DisableLog('rdApp.*')
        receptor_str = receptor_bin.decode()

        if box is not None:
            center, box_size = box
        else:
            # Cal Edge
            with tempfile.NamedTemporaryFile() as fp:
                fp.write(receptor_bin)
                mol = Chem.MolFromPDBFile(fp.name, sanitize=False)

            coords_list = mol.GetConformer(0).GetPositions()
            max_coords = coords_list.max(axis=0) + 5
            min_coords = coords_list.min(axis=0) - 5
            center = (max_coords + min_coords) / 2
            box_size = max_coords - min_coords
            # print(f'--> [src] center: {center}, box_size: {box_size}')

        gen_qv = QVinaDockingTask(receptor_str, ligand_mol, use_uff=use_uff, center=center, size=box_size)
        gen_aff = gen_qv.run_sync(**kwargs)[0]['affinity']

        if gen_aff:
            with open(gen_qv.docked_pdbqt_path, 'rb') as f:
                docked_pdbqt = f.read()
        else:
            docked_pdbqt = None

        gen_qv.clean()

    except Exception as error:
        print(f"docking_score | _hash: {_hash}, error: {error}")
        traceback.print_exc()
        gen_aff = None
        docked_pdbqt = None

    return gen_aff, docked_pdbqt
