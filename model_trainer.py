"""
model_trainer.py — standalone training entry point.
Delegates to main_pipeline.py with stage=model_training.
"""
import sys, subprocess
from pathlib import Path
if __name__ == "__main__":
    args = ["python", str(Path(__file__).parent / "main_pipeline.py"),
            "--stage", "model_training"] + sys.argv[1:]
    sys.exit(subprocess.call(args))
