#!/usr/bin/env python3
"""
Setup script for macOS M-series (Apple Silicon) Dolphin compatibility.

The bundled Dolphin has Python 3.13.5 embedded but is missing the standard library.
This script downloads Python 3.13.5 from python.org and copies the stdlib into the
dolphin0 framework bundle.

Run this BEFORE clone_dolphins.py so the stdlib gets copied to all instances automatically.

Usage:
    1. python scripts/download_dolphin.py
    2. python scripts/setup_macos_dolphin.py   <-- this script
    3. python scripts/clone_dolphins.py
    4. python BTR.py --device mps
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

PYTHON_VERSION = "3.13.5"
PYTHON_PKG_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-macos11.pkg"


def main():
    if sys.platform != "darwin":
        print("This script is only for macOS.")
        return

    project_dir = Path(__file__).parent.parent
    dolphin0 = project_dir / "dolphin0"

    if not dolphin0.exists():
        print("dolphin0 not found. Run download_dolphin.py first.")
        return

    framework_path = dolphin0 / "DolphinQt.app" / "Contents" / "Frameworks" / "Python.framework" / "Versions" / "3.13"
    if not framework_path.exists():
        print(f"Error: Python framework not found at {framework_path}")
        return

    lib_path = framework_path / "lib"
    dynload_path = lib_path / "python3.13" / "lib-dynload"

    if dynload_path.exists() and any(dynload_path.glob("*.so")):
        print("Python stdlib already installed in dolphin0.")
        response = input("Reinstall? [y/N] ").strip().lower()
        if response != 'y':
            print("Skipping installation.")
            return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download Python pkg
        pkg_path = tmpdir / f"python-{PYTHON_VERSION}.pkg"
        print(f"\nDownloading Python {PYTHON_VERSION} from python.org...")
        subprocess.run(
            ["curl", "-L", "-o", str(pkg_path), PYTHON_PKG_URL],
            check=True
        )

        # Extract pkg
        print("Extracting package...")
        expanded_dir = tmpdir / "python-expanded"
        subprocess.run(
            ["pkgutil", "--expand", str(pkg_path), str(expanded_dir)],
            check=True
        )

        # Extract Python_Framework.pkg payload
        framework_pkg = expanded_dir / "Python_Framework.pkg"
        payload_dir = tmpdir / "payload"
        payload_dir.mkdir()

        print("Extracting Python framework...")
        subprocess.run(
            f"cd {payload_dir} && cat {framework_pkg}/Payload | gunzip -dc | cpio -i 2>/dev/null",
            shell=True,
            check=True
        )

        # Find the lib directory
        src_lib = payload_dir / "Versions" / "3.13" / "lib"
        if not src_lib.exists():
            print(f"Error: Could not find lib directory at {src_lib}")
            return

        # Remove existing lib if present
        if lib_path.exists():
            shutil.rmtree(lib_path)

        print("Copying stdlib to dolphin0...")
        shutil.copytree(src_lib, lib_path)

    print("\nDone! Python stdlib installed in dolphin0.")
    print("Now run clone_dolphins.py to copy to all instances, then run BTR.py --device mps")


if __name__ == "__main__":
    main()
