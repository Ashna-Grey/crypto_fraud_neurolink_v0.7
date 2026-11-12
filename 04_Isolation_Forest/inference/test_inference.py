"""Compatibility wrapper retained for consistency with neighboring modules."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests" / "test_pipeline.py"),
        run_name="__main__",
    )
