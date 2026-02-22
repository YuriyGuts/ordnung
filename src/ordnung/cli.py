import argparse
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
    organize(args.directory)


if __name__ == "__main__":
    main()
