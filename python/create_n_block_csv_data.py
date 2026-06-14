#!/usr/bin/env python3
"""
create_n_block_csv_data.py

Unified block-CSV pipeline for 5-window block shapefiles.

This script runs both block CSV generators in sequence so all block outputs
are produced from a single command:

1) Area pipeline (create_n_block_area_csv_data.py)
   - Coverage, observed area, projected area, AOI totals, and total_summary_raw
2) Volume pipeline (create_n_block_volume_csv_data.py)
   - Observed/projected/total lake-volume CSVs
   - Populates lake-volume columns in total_summary_raw

Run:
    python create_n_block_csv_data.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_main_from_file(filename: str):
    """
    Load and return the `main` callable from a sibling .py script.

    This avoids dependency on the current working directory / sys.path,
    which makes the unified runner robust in Spyder runfile sessions.
    """
    script_path = Path(__file__).resolve().with_name(filename)
    if not script_path.exists():
        raise FileNotFoundError(
            f"Required sibling script not found: {script_path}"
        )

    spec = spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {script_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path.name} does not define a main() function")

    return module.main


def main() -> None:
    run_area_pipeline = _load_main_from_file("create_n_block_area_csv_data.py")
    run_volume_pipeline = _load_main_from_file("create_n_block_volume_csv_data.py")

    print("=== Block CSV pipeline: AREA ===")
    run_area_pipeline()
    print("\n=== Block CSV pipeline: VOLUME ===")
    run_volume_pipeline()
    print("\nDone: all block CSV files generated.")


if __name__ == "__main__":
    main()
