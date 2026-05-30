# =============================================================================
# From: https://github.com/guanjq/targetdiff with minor modifications

# MIT License

# Copyright (c) 2023 Jiaqi Guan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import os
import subprocess
import random
import string
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from utils.reconstruct import reconstruct_from_generated

from MolKit import Read
import MolKit.molecule
import MolKit.protein
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation


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
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        xyz = data.ligand_pos.clone().cpu().tolist()
        atomic_nums = data.ligand_element.clone().cpu().tolist()
        # print('atomic nums: ', atomic_nums)
        ligand_rdmol = reconstruct_from_generated(xyz, atomic_nums)
        # print('smiles: ', Chem.MolToSmiles(ligand_rdmol))
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)

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

    def __init__(self, pdb_block, ligand_rdmol, conda_env='SBDD', tmp_dir='./tmp', use_uff=True, center=None,
                 size_factor=1.):
        super().__init__(pdb_block, ligand_rdmol)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            UFFOptimizeMolecule(ligand_rdmol)
        # print('after uff smiles: ', Chem.MolToSmiles(ligand_rdmol))
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

        # print(f'!!!!!!!!!{self.receptor_path}!!!!!!!!!!')

    def run(self, exhaustiveness=16):
    #         commands = """
    # eval "$(conda shell.bash hook)"
    # conda activate {env}
    # cd {tmp}
    # # Prepare receptor (PDB->PDBQT)
    # prepare_receptor4.py -r {receptor_id}.pdb
    # # Prepare ligand
    # obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
    # /fs/ess/PCON0041/gruoxi/qvina/bin/qvina2.1 \
    #     --receptor {receptor_id}.pdbqt \
    #     --ligand {ligand_id}.pdbqt \
    #     --center_x {center_x:.4f} \
    #     --center_y {center_y:.4f} \
    #     --center_z {center_z:.4f} \
    #     --size_x {size_x} --size_y {size_y} --size_z {size_z} \
    #     --exhaustiveness {exhaust}
    # obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
    #         """.format(
    #             receptor_id=self.receptor_id,
    #             ligand_id=self.ligand_id,
    #             env=self.conda_env,
    #             tmp=self.tmp_dir,
    #             exhaust=exhaustiveness,
    #             center_x=self.center[0],
    #             center_y=self.center[1],
    #             center_z=self.center[2],
    #             size_x=self.size_x,
    #             size_y=self.size_y,
    #             size_z=self.size_z
    #         )

        commands = """
            eval "$(conda shell.bash hook)" || echo "Failed: conda initialization"
            conda activate {env} || echo "Failed: conda activate"
            cd {tmp} || echo "Failed: cd {tmp}"
            python {prepare_receptor_path} -r {receptor_id}.pdb || echo "Failed: prepare_receptor4.py"
            obabel {ligand_id}.sdf -O{ligand_id}.pdbqt || echo "Failed: obabel ligand preparation"
            {qvina_path} \
                --receptor {receptor_id}.pdbqt \
                --ligand {ligand_id}.pdbqt \
                --center_x {center_x:.4f} \
                --center_y {center_y:.4f} \
                --center_z {center_z:.4f} \
                --size_x {size_x} --size_y {size_y} --size_z {size_z} \
                --exhaustiveness {exhaust} 
                --no_refine
                --score_only || echo "Failed: qvina"
            obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h || echo "Failed: obabel output"
        """.format(
            receptor_id=self.receptor_id,
            ligand_id=self.ligand_id,
            env=self.conda_env,
            tmp=self.tmp_dir,
            prepare_receptor_path='/fs/ess/PCON0041/gruoxi/AutoDockTools_py3/AutoDockTools/Utilities24/prepare_receptor4.py',
            qvina_path='/fs/ess/PCON0041/gruoxi/qvina/bin/qvina2.1',
            exhaust=exhaustiveness,
            center_x=self.center[0],
            center_y=self.center[1],
            center_z=self.center[2],
            size_x=self.size_x,
            size_y=self.size_y,
            size_z=self.size_z
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)
        # print(f"!!!!!!!!!{self.docked_sdf_path}!!!!!!!!!!")

        # print(f"!!!!!!!!!{self.receptor_id}!!!!!!!!!!{self.ligand_id}!!!!!!!")

        # print(commands)

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

    def run_sync(self, exhaustiveness=16):
        self.run(exhaustiveness)
        while self.get_results() is None:
            pass
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
                self.error_output = self.proc.stderr.readlines()
                try:
                    self.results = parse_qvina_outputs(self.docked_sdf_path)
                except:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)
                    return []
            return self.results

