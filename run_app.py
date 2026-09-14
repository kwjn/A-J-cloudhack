"""Compatibility launcher for the Streamlit-only deployed application."""

from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = PROJECT_ROOT / "app.py"


def run() -> None:
    """Replace this process with Streamlit using the current interpreter."""
    if not STREAMLIT_APP.is_file():
        raise FileNotFoundError(f"Streamlit app not found: {STREAMLIT_APP}")

    print(
        "Starting Streamlit with inline WebRTC camera and recognition...",
        flush=True,
    )
    os.chdir(PROJECT_ROOT)
    os.execv(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", "app.py"],
    )


if __name__ == "__main__":
    run()
