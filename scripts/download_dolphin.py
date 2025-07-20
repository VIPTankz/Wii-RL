import os
from pathlib import Path

from common import download_file, extract_zip

SAVESTATE_URL = "https://github.com/VIPTankz/Wii-RL/releases/download/dolphin/dolphin0.zip"
ZIP_NAME = "Dolphin.zip"

def main():
    current_directory = Path.cwd()

    download_file(SAVESTATE_URL, ZIP_NAME)
    extract_zip(ZIP_NAME, current_directory)

    os.remove(ZIP_NAME)

    print("Done!")


if __name__ == "__main__":
    main()