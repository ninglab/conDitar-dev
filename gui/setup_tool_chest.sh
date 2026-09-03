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

if ! command -v make >/dev/null 2>&1; then
  echo "ERROR: make was not found." >&2
  echo "Install build tools to compile Lilly Medchem Rules, then rerun this script." >&2
  exit 2
fi

if [[ ! -d vendor/Lilly-Medchem-Rules ]]; then
  echo "ERROR: vendor/Lilly-Medchem-Rules was not found." >&2
  echo "Restore the vendored Lilly Medchem Rules source, then rerun this script." >&2
  exit 2
fi

mkdir -p .tool_chest
rm -rf .tool_chest/Lilly-Medchem-Rules
cp -R vendor/Lilly-Medchem-Rules .tool_chest/Lilly-Medchem-Rules
make -C .tool_chest/Lilly-Medchem-Rules
echo
echo "Done. The GUI launchers will use the conditar-gui-dev environment automatically."
