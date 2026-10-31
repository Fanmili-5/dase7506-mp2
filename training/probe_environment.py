"""Read-only hardware/dependency probe; works without third-party packages."""

import argparse
from datetime import datetime, timezone, timedelta
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command_output(command):
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"exit_code": process.returncode,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def collect(probe_torch=False):
    packages = {}
    for package in ("numpy", "torch", "xgboost", "scipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python_executable": sys.executable,
        "python": sys.version,
        "python_3_12": sys.version_info[:2] == (3, 12),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": packages,
        "official_numpy_requirement": "2.5.3",
        "official_numpy_matches": packages["numpy"] == "2.5.3",
        "thread_environment": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    nvidia_smi = shutil.which("nvidia-smi")
    result["nvidia_smi"] = command_output([
        nvidia_smi or "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits"])
    if platform.system() == "Windows":
        result["windows_hardware"] = command_output([
            "powershell.exe", "-NoProfile", "-Command",
            "$c=Get-CimInstance Win32_ComputerSystem; "
            "$p=Get-CimInstance Win32_Processor; "
            "[pscustomobject]@{CPU=($p.Name -join '; '); "
            "RAM_bytes=$c.TotalPhysicalMemory}|ConvertTo-Json -Compress"])
    if probe_torch and packages["torch"] is not None:
        try:
            import torch

            result["torch_runtime"] = {"cuda_build": torch.version.cuda,
                                       "cuda_available": torch.cuda.is_available()}
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                result["torch_runtime"].update({"device": props.name,
                                               "total_memory_bytes": props.total_memory})
                a = torch.ones((16, 16), device="cuda")
                result["torch_runtime"]["cuda_matmul_ok"] = bool((a @ a == 16).all().item())
        except Exception as error:
            result["torch_runtime"] = {"error": repr(error)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-torch", action="store_true")
    args = parser.parse_args()
    result = collect(args.probe_torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
