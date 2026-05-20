"""main_train_tokenizer.py — standalone tokenizer training entry point."""
import sys, subprocess
from pathlib import Path
if __name__ == "__main__":
    args = ["python", str(Path(__file__).parent / "main_pipeline.py"),
            "--stage", "train_tokenizer"] + sys.argv[1:]
    sys.exit(subprocess.call(args))
