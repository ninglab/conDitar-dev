# conDitar GUI Tool Chest

Drop Python tool modules in this folder to make them available in the Results tab.

Each tool file must define:

```python
def describe() -> dict:
    return {
        "id": "my_tool",
        "name": "My Tool",
        "description": "Annotates generated molecules with custom properties.",
        "available": True,
        "inputs": [],
        "outputs": [{"name": "MY_PROPERTY", "type": "text"}],
    }


def run(job_root: str, run_root: str, options: dict) -> dict:
    ...
```

`job_root` is the completed job folder, and generated SDF files live under `job_root/outputs/`.
`run_root` is a new folder under `job_root/tool_runs/` for logs, summaries, and temporary files.

Tools should write any properties they produce back into the generated SDF files. The GUI reloads those SDFs after a tool finishes, so new properties can be shown in the table, selected-molecule metrics, and CSV export.

Filtering thresholds and export requirements are configured in the Results view,
not in the tool module.
