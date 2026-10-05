#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

import yaml
import numpy as np
from rdkit import Chem

adapter_path = Path(__file__).resolve().parents[1] / "runtime" / "pocket_generate.py"
spec = importlib.util.spec_from_file_location("diffsmol_pocket_adapter", adapter_path)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sdf_path = work / "reference.sdf"
        pdb_path = work / "pocket.pdb"
        sdf_path.write_text("adapter test\n")
        pdb_path.write_text("adapter test\n")
        args = argparse.Namespace(guidance=True, device="cpu", num_samples=1)

        raw_path = adapter.prepare_raw_pair(work, sdf_path, pdb_path)
        config_path = adapter.build_config(work, raw_path, args)
        config = yaml.safe_load(config_path.read_text())

        assert (work / "processed").is_dir()
        assert config["data"]["dataset"] == "crossdocked"
        assert config["data"]["processed_path"] == str(work / "processed")
        assert config["sample"]["use_pocket"] is True
        assert config["sample"]["pocket_threshold"] == 0.5

        output = work / "outputs"
        output.mkdir()
        result = {
            "pred_ligand_pos": [np.zeros((2, 3)), np.ones((2, 3))],
            "pred_ligand_v": [np.array([0, 0]), np.array([0, 0])],
        }
        args.input = sdf_path
        args.protein = pdb_path
        with patch.object(adapter.torch, "load", return_value=result), \
             patch.object(adapter.transforms, "get_atomic_number_from_index", return_value=[6, 6]), \
             patch.object(adapter.transforms, "is_aromatic_from_index", return_value=[False, False]), \
             patch.object(adapter.reconstruct, "reconstruct_from_generated", side_effect=lambda *_: Chem.MolFromSmiles("CC")):
            adapter.export_sdf(work, output, args, 0)
        assert sorted(path.name for path in output.glob("*.sdf")) == [
            "generated_0000.sdf", "generated_0001.sdf",
        ]

    print("DiffSMol pocket adapter checks passed.")


if __name__ == "__main__":
    main()
