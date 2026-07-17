"""Serve the conDitar frontend preview with no project dependencies."""
from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import threading
import webbrowser

from backend.jobs import LocalJobManager


PROJECT_ROOT = Path(__file__).resolve().parent
JOB_MANAGER = LocalJobManager(PROJECT_ROOT)
REQUIRED_PATHS = (
    "index.html",
    "src/app.js",
)


def validate_project() -> None:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Required application files are missing:\n{formatted}")


class ConDitarRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 50 * 1024 * 1024:
            raise ValueError("Request body is too large.")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if not self.path.startswith("/api/"):
            return super().do_GET()
        parts = self.path.split("?")[0].strip("/").split("/")
        try:
            if parts == ["api", "health"]:
                self._send_json({
                    "ok": True,
                    "container_backend": JOB_MANAGER.container_runtime_kind,
                    "container_runtime": JOB_MANAGER.container_runtime,
                    "gpu_available": bool(Path("/dev/nvidia0").exists()),
                    "docker_image": JOB_MANAGER.docker_image,
                    "docker_tar": JOB_MANAGER.docker_tar,
                    "slurm": {
                        "sbatch": JOB_MANAGER.sbatch_bin,
                        "squeue": JOB_MANAGER.squeue_bin,
                        "sacct": JOB_MANAGER.sacct_bin,
                        "defaults": JOB_MANAGER.slurm_defaults,
                    },
                })
            elif parts == ["api", "jobs"]:
                self._send_json({"jobs": JOB_MANAGER.list_jobs()})
            elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                job = JOB_MANAGER.get_job(parts[2])
                self._send_json({"job": job} if job else {"error": "Job not found."}, 200 if job else 404)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "logs":
                self._send_json(JOB_MANAGER.logs(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "results":
                self._send_json(JOB_MANAGER.results(parts[2]))
            else:
                self._send_json({"error": "Unknown API endpoint."}, 404)
        except Exception as error:
            self._send_json({"error": str(error)}, 500)

    def do_POST(self) -> None:
        if not self.path.startswith("/api/"):
            return super().do_POST()
        parts = self.path.split("?")[0].strip("/").split("/")
        try:
            if parts == ["api", "jobs"]:
                self._send_json({"job": JOB_MANAGER.submit(self._read_json())}, 201)
            elif parts == ["api", "jobs", "batch"]:
                self._send_json(JOB_MANAGER.submit_batch(self._read_json()), 201)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                self._send_json({"job": JOB_MANAGER.cancel(parts[2])})
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "export":
                self._send_json(JOB_MANAGER.export_job(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in {"archive", "cleanup"}:
                self._send_json({"job": JOB_MANAGER.archive_job(parts[2])})
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "rerun":
                self._send_json({"job": JOB_MANAGER.rerun_job(parts[2])}, 201)
            elif len(parts) == 4 and parts[:3] == ["api", "jobs", "rerun"]:
                self._send_json({"job": JOB_MANAGER.rerun_job(parts[3])}, 201)
            else:
                self._send_json({"error": "Unknown API endpoint."}, 404)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:
            self._send_json({"error": str(error)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve conDitar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--open", action="store_true", help="Open the GUI in the default browser")
    args = parser.parse_args()
    validate_project()
    handler = partial(
        ConDitarRequestHandler,
        directory=str(PROJECT_ROOT),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"conDitar: {url}")
    print("Press Ctrl+C to stop the server.")
    if args.open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
