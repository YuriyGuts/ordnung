import argparse
import sys
from importlib.metadata import metadata
from pathlib import Path

from ordnung.organize import organize


def main():
    program_meta = metadata("ordnung")
    parser = argparse.ArgumentParser(
        prog=program_meta["Name"],
        description=program_meta["Summary"],
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="The path to the directory to organize.",
    )
    args = parser.parse_args()
    result = organize(args.directory)
    if not result.is_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
