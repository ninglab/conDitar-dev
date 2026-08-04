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
echo
echo "Done. The GUI launchers will use the conditar-gui-dev environment automatically."
