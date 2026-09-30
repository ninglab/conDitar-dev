#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.backend.diffsmol_pocket_generate import build_config, prepare_raw_pair


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sdf_path = work / "reference.sdf"
        pdb_path = work / "pocket.pdb"
        sdf_path.write_text("adapter test\n")
        pdb_path.write_text("adapter test\n")
        args = argparse.Namespace(guidance=True, device="cpu", num_samples=1)

        raw_path = prepare_raw_pair(work, sdf_path, pdb_path)
        config_path = build_config(work, raw_path, args)
        config = yaml.safe_load(config_path.read_text())

        assert (work / "processed").is_dir()
        assert config["data"]["dataset"] == "crossdocked"
        assert config["data"]["processed_path"] == str(work / "processed")
        assert config["sample"]["use_pocket"] is True
        assert config["sample"]["pocket_threshold"] == 0.5

    print("DiffSMol pocket adapter checks passed.")


if __name__ == "__main__":
    main()
