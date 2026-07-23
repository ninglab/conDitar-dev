# conDitar GUI Tool Chest

Drop Python tool modules in this folder to make them available in the Results tab.

## Fast path

1. Copy `_tool_template.py` to a new file name, for example `my_filter.py`.
2. Edit `TOOL_ID`, `TOOL_NAME`, and `OUTPUTS` near the top of the file.
3. Replace the example code in `calculate_properties()` with your own logic.
4. Restart the GUI, or refresh the page if the backend is already running.
5. Open a completed job in **Results** and click **Run tool**.

The leading underscore matters: `_tool_template.py` is ignored by the GUI, but
`my_filter.py` will be loaded as a real tool.

## Outputs

Each output becomes an SDF property. The GUI reads those properties after the
tool runs and makes them available in the table, CSV export, distribution chart,
and export filters.

Use these output types:

- `number`: threshold/range filter, histogram distribution.
- `boolean`: Pass/Fail filter.
- `text`: dropdown filter for short categorical values.

Example:

```python
OUTPUTS = [
    {"name": "MY_FILTER_PASS", "label": "My Filter", "type": "boolean"},
    {"name": "MY_SCORE", "label": "My Score", "type": "number"},
    {"name": "MY_REASON", "label": "My Reason", "type": "text", "filterable": False},
]
```

Set `"filterable": false` for notes, long text, raw command output, or anything
that should not appear in export requirements.

## Tool API

Each tool file must define:

```python
def describe() -> dict:
    return {
        "id": "my_tool",
        "name": "My Tool",
        "description": "Annotates generated molecules with custom properties.",
        "available": True,
        "inputs": [],
        "outputs": [{"name": "MY_PROPERTY", "label": "My Property", "type": "text"}],
    }


def run(job_root: str, run_root: str, options: dict) -> dict:
    ...
```

`job_root` is the completed job folder, and generated SDF files live under `job_root/outputs/`.
`run_root` is a new folder under `job_root/tool_runs/` for logs, summaries, and temporary files.

Tools should write any properties they produce back into the generated SDF files.
The GUI reloads those SDFs after a tool finishes, so new properties can be shown
in the table, selected-molecule metrics, and CSV export.

Output `type` can be `number`, `boolean`, or `text`. Boolean values are shown
as Pass/Fail filters, numeric values get threshold/range controls, and short
text/categorical values get dropdown filters. Fields ending in `_STATUS`,
`_REASONS`, or `_OUTPUT` are treated as supporting details and are not exposed
as export filters. Set `"filterable": false` on any output that should stay out
of the filter and distribution controls.

Filtering thresholds and export requirements are configured in the Results view,
not in the tool module.

## Dependency Tips

Keep tools as lightweight as possible. If a tool needs an external package or
command-line program, install it in the GUI environment, not in the conDitar
sampling container. The GUI will still start if optional tools are missing; mark
the tool unavailable from `describe()` and return a short `error` message that
tells the user what to install.
