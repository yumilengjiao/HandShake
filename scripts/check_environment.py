"""Dependency import/CPU smoke probes; does not open cameras or train models.

Run with the project's interpreter. Individual subprocess timeouts keep native
library initialization from hanging the entire verification command.
"""

import argparse
import glob
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


PROBES = {
    "numpy": "import numpy as np; assert np.zeros((2,3)).shape == (2,3); print(np.__version__)",
    "pydantic": "import pydantic; print(pydantic.__version__)",
    "PyYAML": "import yaml; assert yaml.safe_load('value: 1')['value'] == 1; print(yaml.__version__)",
    "opencv-contrib-python": "import cv2, numpy as np; assert cv2.cvtColor(np.zeros((2,2,3),dtype=np.uint8),cv2.COLOR_RGB2GRAY).shape == (2,2); print(cv2.__version__)",
    "mediapipe": "import mediapipe as mp; assert hasattr(mp.tasks.vision, 'HandLandmarker'); print(mp.__version__)",
    "torch": "import torch, numpy as np; t=torch.from_numpy(np.ones((2,3),dtype=np.float32)); assert t.sum().item() == 6; print(torch.__version__, 'CPU tensor + NumPy bridge OK')",
    "scikit-learn": "import sklearn; from sklearn.model_selection import GroupKFold; print(sklearn.__version__, 'GroupKFold import OK')",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Avoid global ROS packages shadowing this project's environment.
    env["MPLCONFIGDIR"] = "/tmp/handshake-mpl"
    result = dict(python=platform.python_version(), os=platform.system(), kernel=platform.release(), architecture=platform.machine(), desktop_session=env.get("XDG_SESSION_TYPE"), camera_nodes=glob.glob("/dev/video*"), probes={}, scope="imports_and_synthetic_CPU_operations_only")
    for distribution, code in PROBES.items():
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result["probes"][distribution] = dict(status="not_installed")
            continue
        try:
            probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout, env=env)
            detail = dict(status="passed" if probe.returncode == 0 else "failed", version=installed_version, stdout=probe.stdout.strip(), stderr=probe.stderr.strip(), returncode=probe.returncode)
        except subprocess.TimeoutExpired:
            detail = dict(status="timeout", version=installed_version, timeout_seconds=args.timeout)
        result["probes"][distribution] = detail
        print(f"{distribution}: {detail['status']}", file=sys.stderr, flush=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return int(any(p["status"] in {"failed", "timeout"} for p in result["probes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
