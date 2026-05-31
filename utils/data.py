# =============================================================================
# From: https://github.com/guanjq/targetdiff
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

import pdb
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit import Geometry
from openbabel import openbabel as ob

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
RADIUS = 10

class PDBProteinBase(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    def query_residues_ligand(self, ligand_pos, radius=RADIUS, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand_pos:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected



# =============================================================================
# conDitar is copyrighted by the Ohio State University and covered by US 64/023,113. 
#
# conDitar may be licensed solely for educational and research purposes by non-profit institutions and US government agencies only. 
#
# For other proposed uses, contact tlcip@osu.edu. 
#
# The software may not be sold or redistributed without prior approval. 
#
# You may not use the software to train or process or input the software into or make it accessible to: automated software, services or tools, including, but not limited to, artificial intelligence solutions, algorithms, machine learning, large language models, robots, spiders, crawlers, search engines, text or data mining or any other aggregation functionality.   
#
# One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. 
#
# These terms and conditions are subject to change at any time without prior notice.
# =============================================================================

def center_mol(mol, pre_center=None):
    conformer = mol.GetConformers()[0]
    pos = conformer.GetPositions()
    if pre_center is None: center = np.mean(pos, axis=0)
    else: center = pre_center
    conformer = mol.GetConformers()[0]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)

        p = Geometry.Point3D(float(pos.x - center[0]), float(pos.y - center[1]), float(pos.z - center[2]))
        conformer.SetAtomPosition(idx, p)
    return center

def load_ligand_mol(path, removeHs=True):
    mol = None
    last_exc = None
    for sanitize in (True, False):
        try:
            suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)
            mol = next(iter(suppl))
        except Exception as e:
            last_exc = e
            mol = None
        if mol is not None:
            return mol
    if last_exc is not None:
        raise last_exc
    raise ValueError(f'Failed to read ligand molecule from {path}')

def parse_sdf_file(path, center_ligand=False, pre_center=None):
    rdmol = load_ligand_mol(path, removeHs=True)
    if center_ligand:
        center = center_mol(rdmol, pre_center=pre_center)
    else:
        center = None
    return parse_rdkit_mol(rdmol, center=center), rdmol


def convert_sdf_to_pdb(sdf_path, pdb_path):
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("sdf", "pdb")
    mol = ob.OBMol()
    ob_conversion.ReadFile(mol, sdf_path)
    ob_conversion.WriteFile(mol, pdb_path)


def parse_rdkit_mol(rdmol, center=None):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Remove Hydrogens (if needed)
    rd_num_atoms = rdmol.GetNumAtoms()

    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Hybridization
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybridization.append((atom.GetIdx(), str(atom.GetHybridization())))
    hybridization = [v[1] for v in sorted(hybridization)]

    # Positions
    pos = np.array(rdmol.GetConformers()[-1].GetPositions(), dtype=np.float32)

    # Element + center of mass
    ptable = Chem.GetPeriodicTable()
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight

    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int32)

    # Bonds
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'center': center
    }

    return data


class PDBProtein(PDBProteinBase):

    def residues_to_atom_dict(self, residues, name='POCKET'):
        element, pos, is_backbone, atom_name, atom_to_aa_type = [], [], [], [], []
        for residue in residues:
            for atom_idx in residue['atoms']:
                atom = self.atoms[atom_idx]
                atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
                element.append(atomic_number)
                pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
                atom_name.append(atom['atom_name'])
                is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
                atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

        return {
            'element': np.array(element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(pos, dtype=np.float32),
            'is_backbone': np.array(is_backbone, dtype=np.bool_),
            'atom_name': atom_name,
            'atom_to_aa_type': np.array(atom_to_aa_type, dtype=np.int64)
        }


def parse_pdb(protein_path, ligand_path):
    protein = PDBProtein(protein_path)
    if ligand_path == None:
        pocket_dict = protein.residues_to_atom_dict(protein.residues)
    else:
        ligand_dict, rdmol = parse_sdf_file(ligand_path, center_ligand=True)
        # pocket_dict = protein.residues_to_atom_dict(protein.residues)
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'] + ligand_dict['center'])
        pocket_dict = protein.residues_to_atom_dict(pocket_residues)
    return pocket_dict, ligand_dict


def save_residues_around_reference_ligand(protein_path, ligand_path, out_path, radius=10.0,
                                          criterion='center_of_mass', name='POCKET'):
    protein = PDBProtein(protein_path)
    ligand_dict, _ = parse_sdf_file(ligand_path, center_ligand=False)
    ligand_pos = np.asarray(ligand_dict['pos'], dtype=np.float32)
    if ligand_pos.size == 0:
        raise ValueError(f'Ligand has no coordinates: {ligand_path}')

    pocket_residues = protein.query_residues_ligand(
        ligand_pos=ligand_pos,
        radius=radius,
        criterion=criterion,
    )
    pdb_block = protein.residues_to_pdb_block(pocket_residues, name=name)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(pdb_block)
    return pocket_residues

def parse_sdf_mol_with_pdb(index, raw_path, center_ligand=True):
    protein_fn, ligand_fn = index[:2]
    tmp = "/"+"/".join(ligand_fn.split("/")[-2:])
    ligand_dict, rdmol = parse_sdf_file(raw_path+tmp, center_ligand=center_ligand)
    tmp = "/"+"/".join(protein_fn.split("/")[-2:])
    protein_path = raw_path + tmp
    protein = PDBProtein(protein_path)
    if center_ligand:
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'] + ligand_dict['center'])
    else:
        pocket_residues = protein.query_residues_ligand(ligand_dict['pos'])
    pocket_dict = protein.residues_to_atom_dict(pocket_residues)
    return (protein_fn, ligand_fn, ligand_dict, pocket_dict, rdmol)
