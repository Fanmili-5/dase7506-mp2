"""Process-local audit hook excluding test data throughout round two."""

import os
from pathlib import Path
import sys

ALLOW_VALIDATION = False
EVENTS = {"test_read_attempts": 0, "validation_read_attempts_before_gate": 0}


def install(allow_validation=False, offline=False):
    global ALLOW_VALIDATION
    ALLOW_VALIDATION = allow_validation

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).replace("\\", "/").lower()
            # Release manifests are permitted; candidate/reference/array files
            # of disallowed splits are not. Official tools themselves are fixed.
            base = Path(name).name
            if "gsm8k_test" in base:
                EVENTS["test_read_attempts"] += 1
                raise PermissionError("Round two excludes all official test input/reference/array files")
            if "gsm8k_validation" in base and not ALLOW_VALIDATION:
                EVENTS["validation_read_attempts_before_gate"] += 1
                raise PermissionError("Validation cannot open before the registered shortlist gate")
        if offline and event in ("socket.connect", "subprocess.Popen"):
            raise PermissionError("Offline inference cannot connect or launch processes")

    sys.addaudithook(audit)


if __name__ == "__main__":
    import argparse
    import runpy

    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=("predict.py", "score_predictions.py"))
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    install(allow_validation=True, offline=True)
    sys.path.insert(0, str(root / "code"))
    sys.argv = [str(root / "code" / args.tool)] + remaining
    runpy.run_path(sys.argv[0], run_name="__main__")
