# =============================================================================
# From: https://github.com/guanjq/targetdiff with minor modifications
#
# MIT License
#
# Copyright (c) 2023 Jiaqi Guan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np

if not hasattr(np, 'int'):
    np.int = int

from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils
from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib
import pdb
import hashlib
import shutil
import sys
from utils.reconstruct import reconstruct_from_generated
from utils.docking_qvina import get_random_id, BaseDockingTask


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string()
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        pdb2pqr_cmd = shutil.which('pdb2pqr30') or shutil.which('pdb2pqr')
        if pdb2pqr_cmd is None:
            raise RuntimeError(
                "Could not find 'pdb2pqr30' or 'pdb2pqr' in PATH while preparing receptor."
            )
        result = subprocess.run(
            [pdb2pqr_cmd, '--ff=AMBER', self.prot, self.prot_pqr],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(self.prot_pqr):
            raise RuntimeError(
                "Failed to generate receptor PQR file. "
                f"command={pdb2pqr_cmd} | input={self.prot} | output={self.prot_pqr} | "
                f"returncode={result.returncode} | stderr={result.stderr.strip()} | stdout={result.stdout.strip()}"
            )

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        result = subprocess.run(
            [sys.executable, prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(prot_pdbqt):
            raise RuntimeError(
                "Failed to generate receptor PDBQT file. "
                f"python={sys.executable} | script={prepare_receptor} | input={self.prot_pqr} | output={prot_pdbqt} | "
                f"returncode={result.returncode} | stderr={result.stderr.strip()} | stdout={result.stdout.strip()}"
            )


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
                # print(self.lig_pdbqt)
                
                v.write_poses(self.lig_pdbqt, n_poses=1, overwrite=True)
            else:
                raise ValueError
            return score, pose, self.lig_pdbqt


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, ligand_filename, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        ligand_path = ligand_filename
        protein_path = protein_root
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_filename=None, protein_root='../data/crossdocked', test=True, **kwargs):
        # load original pdb
        if protein_filename == None:
            if test:
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:4] + '_protein.pdb'  
                )
            else:
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:-4] + '_pocket10.pdb'  
                )
        else:
            protein_fn = protein_filename
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_other_mol(cls, ligand_filename, protein_filename, **kwargs):
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_filename)))
        return cls(protein_filename, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='/fs/scratch/PCON0041/Ziqi/tmp', center=None,
                 size_factor=1., buffer=5.0, receptor_cache_dir=None):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        if receptor_cache_dir is None:
            receptor_cache_dir = os.path.join(self.tmp_dir, 'receptor_cache')
        self.receptor_cache_dir = os.path.realpath(receptor_cache_dir)
        os.makedirs(self.receptor_cache_dir, exist_ok=True)
        
        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

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
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def _get_receptor_cache_paths(self):
        receptor_base = os.path.splitext(os.path.basename(self.receptor_path))[0]
        receptor_hash = hashlib.md5(self.receptor_path.encode('utf-8')).hexdigest()[:12]
        cache_prefix = os.path.join(self.receptor_cache_dir, f"{receptor_base}_{receptor_hash}")
        return cache_prefix + '.pqr', cache_prefix + '.pdbqt'

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr, protein_pdbqt = self._get_receptor_cache_paths()
        
        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        
        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        else:
            prot.prot_pqr = protein_pqr
            
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)
        
        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        try:
            score, pose, docked_path = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        except subprocess.CalledProcessError as e:
            return []


        return [{'affinity': score, 'pose': pose}]
    
    def qvina(self, exhaustiveness=16, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr, protein_pdbqt = self._get_receptor_cache_paths()
        
        lig = PrepLig(self.ligand_path, 'sdf')
        #if not os.path.exists(ligand_pdbqt):
        lig.get_pdbqt(ligand_pdbqt)
        
        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        else:
            prot.prot_pqr = protein_pqr
            
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        scores = calculate_qvina2_score(self.receptor_path, protein_pdbqt, self.ligand_path, self.tmp_dir, exhaustiveness=exhaustiveness, **kwargs)

        return scores

    def remove_tmp_file(self):
        os.remove(self.ligand_path)
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        os.remove(ligand_pdbqt)



# from posebusters import PoseBusters
# from posecheck.posecheck import PoseCheck
from pathlib import Path

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(
        f"obabel {sdf_file} -O {pdbqt_outfile} " f"-f {mol_id + 1} -l {mol_id + 1}"
    ).read()
    return pdbqt_outfile


def write_sdf_file(sdf_path, molecules, extract_mol=False):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if extract_mol:
            if m.rdkit_mol is not None:
                w.write(m.rdkit_mol)
        else:
            if m is not None:
                w.write(m)
    w.close()

def calculate_qvina2_score(
    pdb_file,
    receptor_file,
    sdf_file,
    out_dir,
    buster_dict=None,
    violin_dict=None,
    posecheck_dict=None,
    size=20,
    exhaustiveness=16,
    return_rdmol=False,
    filtering=False,
    qvina_bin=None,
):
    """
    Calculate the QuickVina2 score

    Parameters:
    - receptor_file (str): The receptor in pdbqt-format
    - sdf_file (str): The ligand(s) in sdf-format
    - out_dir (str): The directory the docked molecules shall be saved to

    Returns:
    - tuple: (docking scores, PoseCheck dictionary, RDKit molecules [docked])

    """

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)
    pdb_file = Path(pdb_file)

    if receptor_file.suffix == ".pdb":
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(
            os.path.join(out_dir, "docked"), receptor_file.stem + ".pdbqt"
        )
        os.popen(f"prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}")
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols

    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    ligand_name = sdf_file.stem
    ligand_pdbqt_file = Path(os.path.join(out_dir), ligand_name + ".pdbqt")
    out_sdf_file = Path(os.path.join(out_dir), ligand_name + "_out.sdf")

    if filtering:
        valid_ids = []

    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        smiles_in = Chem.MolToSmiles(mol)
        sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

        # center box at ligand's center of mass
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

        # Resolve QuickVina2 from an explicit argument, env var, or PATH so this works outside one filesystem.
        qvina_bin = qvina_bin or os.environ.get("CONDITAR_QVINA_BIN") or shutil.which("qvina2.1") or shutil.which("qvina")
        qvina_bin = shutil.which(qvina_bin) or qvina_bin
        if not qvina_bin or not os.path.exists(qvina_bin):
            raise FileNotFoundError(
                f"QuickVina2 binary not found: {qvina_bin}. "
                "Set CONDITAR_QVINA_BIN or pass --qvina-bin to enable qvina modes."
            )
        # Capture stdout/stderr explicitly so container jobs surface actionable QVina failures.
        result = subprocess.run(
            [
                qvina_bin,
                "--receptor", str(receptor_pdbqt_file),
                "--ligand", str(ligand_pdbqt_file),
                "--center_x", f"{cx:.4f}",
                "--center_y", f"{cy:.4f}",
                "--center_z", f"{cz:.4f}",
                "--size_x", str(size),
                "--size_y", str(size),
                "--size_z", str(size),
                "--exhaustiveness", str(exhaustiveness),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out = result.stdout + result.stderr
        # clean up
        ligand_pdbqt_file.unlink()

        if result.returncode != 0:
            raise RuntimeError(f"QuickVina2 failed with exit code {result.returncode}: {out.strip()}")

        if "-----+------------+----------+----------" not in out:
            continue

        out_split = out.splitlines()
        best_idx = out_split.index("-----+------------+----------+----------") + 1
        best_line = out_split[best_idx].split()
        assert best_line[0] == "1"
        scores.append(float(best_line[1]))

    return scores
