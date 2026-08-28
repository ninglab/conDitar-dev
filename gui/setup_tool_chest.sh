#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found." >&2
  echo "Install Miniconda/Mambaforge, then rerun this script." >&2
  exit 2
fi

echo "Installing GUI Tool Chest dependencies"
conda env update -f environment.yml

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git was not found." >&2
  echo "Install git to fetch Lilly Medchem Rules, then rerun this script." >&2
  exit 2
fi

if ! command -v make >/dev/null 2>&1; then
  echo "ERROR: make was not found." >&2
  echo "Install build tools to compile Lilly Medchem Rules, then rerun this script." >&2
  exit 2
fi

mkdir -p .tool_chest
if [[ ! -d .tool_chest/Lilly-Medchem-Rules/.git ]]; then
  rm -rf .tool_chest/Lilly-Medchem-Rules
  git clone --depth 1 https://github.com/IanAWatson/Lilly-Medchem-Rules.git .tool_chest/Lilly-Medchem-Rules
else
  git -C .tool_chest/Lilly-Medchem-Rules pull --ff-only
fi
make -C .tool_chest/Lilly-Medchem-Rules
echo
echo "Done. The GUI launchers will use the conditar-gui-dev environment automatically."
