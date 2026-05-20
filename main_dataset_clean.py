"""main_dataset_clean.py — standalone dataset cleaning entry point."""
import sys, subprocess
from pathlib import Path
if __name__ == "__main__":
    args = ["python", str(Path(__file__).parent / "main_pipeline.py"),
            "--stage", "dataset_clean"] + sys.argv[1:]
    sys.exit(subprocess.call(args))
