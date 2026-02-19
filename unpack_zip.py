import argparse
import zipfile
from pathlib import Path


def unpack_zip(zip_path: str, dest_dir: str | None = None) -> Path:
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if dest_dir is None:
        dest_dir = zip_path.with_suffix("").parent
    else:
        dest_dir = Path(dest_dir).resolve()

    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack a .zip archive")
    parser.add_argument("zip_path", help="Path to the .zip file")
    parser.add_argument(
        "-o",
        "--output",
        dest="dest_dir",
        help="Output directory (default: same name as zip without extension)",
    )
    args = parser.parse_args()

    dest = unpack_zip(args.zip_path, args.dest_dir)
    print(f"Extracted to: {dest}")


if __name__ == "__main__":
    main()
