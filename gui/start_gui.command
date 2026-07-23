#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "conDitar GUI launcher"
echo

if command -v conda >/dev/null 2>&1; then
  if ! conda run -n conditar-gui-dev python -c "import sys" >/dev/null 2>&1; then
    echo "Setting up the optional GUI Tool Chest environment..."
    ./setup_tool_chest.sh
    echo
  fi
else
  echo "Conda was not found; starting with system Python."
  echo "Optional Tool Chest dependencies such as Lilly may be unavailable."
  echo
fi

./start_cpu_gui.sh
