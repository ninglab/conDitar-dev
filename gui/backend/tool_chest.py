from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class ToolChest:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tools_root = project_root / "tools"

    def list_tools(self) -> list[dict[str, Any]]:
        tools = []
        for path in sorted(self.tools_root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = self._load_module(path)
                metadata = self._metadata(module, path)
                tools.append(metadata)
            except Exception as error:
                tools.append({
                    "id": path.stem,
                    "name": path.stem.replace("_", " ").title(),
                    "description": "Tool could not be loaded.",
                    "available": False,
                    "error": str(error),
                    "inputs": [],
                    "outputs": [],
                })
        return tools

    def run_tool(self, tool_id: str, job_root: Path, options: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._tool_path(tool_id)
        module = self._load_module(path)
        metadata = self._metadata(module, path)
        if not metadata.get("available", True):
            raise ValueError(metadata.get("error") or f"Tool is not available: {tool_id}")
        run = {
            "id": f"{tool_id}-{utc_stamp()}",
            "tool_id": tool_id,
            "tool_name": metadata.get("name") or tool_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
            "options": options or {},
            "outputs": metadata.get("outputs") or [],
        }
        run_root = job_root / "tool_runs" / run["id"]
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            options_path = run_root / "options.json"
            result_path = run_root / "run_result.json"
            stderr_path = run_root / "stderr.log"
            options_path.write_text(json.dumps(options or {}))
            with (run_root / "stdout.log").open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--run-tool", str(path),
                     str(job_root), str(run_root), str(options_path)],
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if process.returncode != 0:
                details = stderr_path.read_text(errors="replace").strip().splitlines()
                reason = details[-1] if details else "No error details were written."
                raise RuntimeError(f"Tool process exited with status {process.returncode}: {reason}")
            result = json.loads(result_path.read_text())
            run["status"] = "completed"
            run["result"] = result or {}
        except Exception as error:
            run["status"] = "failed"
            run["error"] = str(error)
            (run_root / "traceback.txt").write_text(traceback.format_exc())
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_root / "tool_run.json").write_text(json.dumps(run, indent=2))
        return run

    def read_runs(self, job_root: Path) -> list[dict[str, Any]]:
        runs = []
        for path in sorted((job_root / "tool_runs").glob("*/tool_run.json")):
            try:
                runs.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(runs, key=lambda item: item.get("started_at") or "")

    def _tool_path(self, tool_id: str) -> Path:
        if not tool_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid tool id.")
        path = self.tools_root / f"{tool_id}.py"
        if not path.exists():
            raise ValueError(f"Unknown tool: {tool_id}")
        return path

    def _load_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"conditar_tool_{path.stem}", path)
        if not spec or not spec.loader:
            raise ValueError(f"Unable to load tool: {path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "describe") or not hasattr(module, "run"):
            raise ValueError("Tool must define describe() and run(job_root, run_root, options).")
        return module

    def _metadata(self, module: ModuleType, path: Path) -> dict[str, Any]:
        metadata = module.describe()
        if not isinstance(metadata, dict):
            raise ValueError("describe() must return a dictionary.")
        metadata.setdefault("id", path.stem)
        metadata.setdefault("name", metadata["id"].replace("_", " ").title())
        metadata.setdefault("description", "")
        metadata.setdefault("inputs", [])
        metadata.setdefault("outputs", [])
        metadata.setdefault("available", True)
        return metadata


def _run_tool_worker(args: list[str]) -> None:
    tool_path, job_root, run_root, options_path = map(Path, args)
    tool = ToolChest(tool_path.parent.parent)._load_module(tool_path)
    result = tool.run(str(job_root), str(run_root), json.loads(options_path.read_text()))
    (run_root / "run_result.json").write_text(json.dumps(result or {}))


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "--run-tool":
        raise SystemExit("Expected --run-tool PATH JOB_ROOT RUN_ROOT OPTIONS_PATH")
    _run_tool_worker(sys.argv[2:])
