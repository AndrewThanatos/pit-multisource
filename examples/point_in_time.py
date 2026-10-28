"""What was knowable at 2023-10-18 12:00 UTC; usage: python point_in_time.py <dataset root>."""

import sys
from pathlib import Path

from pit_multisource.example import point_in_time

if __name__ == "__main__":
    for key, value in point_in_time(Path(sys.argv[1])).items():
        print(f"{key.removeprefix('example.')}: {value}")
