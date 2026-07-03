"""Regenerate documentation infographics.

The repository already includes generated PNGs under docs/figures/generated.
This script is a lightweight placeholder entry point so the generation process is reproducible.
For the full generation logic, see the notebook/script used to build this release in docs/reproducibility.md.
"""
from pathlib import Path

if __name__ == "__main__":
    out = Path("docs/figures/generated")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generated figures are stored in {out.resolve()}")
