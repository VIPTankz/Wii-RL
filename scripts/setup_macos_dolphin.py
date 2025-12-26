#!/usr/bin/env python3
"""
Setup script for macOS M-series (Apple Silicon) Dolphin compatibility.

The bundled Dolphin has Python 3.13.5 embedded but is missing the standard library.
This script downloads Python 3.13.5 from python.org and copies the stdlib into each
Dolphin instance's framework bundle.

Run this after clone_dolphins.py and before running BTR.py or BTR_test.py on macOS.
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
    dolphin_dirs = sorted(project_dir.glob("dolphin*"))

    if not dolphin_dirs:
        print("No dolphin directories found. Run download_dolphin.py and clone_dolphins.py first.")
        return

    print(f"Found {len(dolphin_dirs)} Dolphin instance(s): {[d.name for d in dolphin_dirs]}")

    # Check if stdlib is already installed
    first_dolphin = dolphin_dirs[0]
    lib_path = first_dolphin / "DolphinQt.app" / "Contents" / "Frameworks" / "Python.framework" / "Versions" / "3.13" / "lib"
    dynload_path = lib_path / "python3.13" / "lib-dynload"

    if dynload_path.exists() and any(dynload_path.glob("*.so")):
        print("Python stdlib already installed in Dolphin bundles.")
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

        # Copy to each Dolphin instance
        for dolphin_dir in dolphin_dirs:
            framework_path = dolphin_dir / "DolphinQt.app" / "Contents" / "Frameworks" / "Python.framework" / "Versions" / "3.13"

            if not framework_path.exists():
                print(f"Warning: {framework_path} does not exist, skipping {dolphin_dir.name}")
                continue

            dst_lib = framework_path / "lib"

            # Remove existing lib if present
            if dst_lib.exists():
                shutil.rmtree(dst_lib)

            print(f"Copying stdlib to {dolphin_dir.name}...")
            shutil.copytree(src_lib, dst_lib)

    print("\nDone! Python stdlib installed in all Dolphin instances.")
    print("You can now run BTR.py or BTR_test.py with --device mps")


if __name__ == "__main__":
    main()
