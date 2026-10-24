"""Copy exactly PUBLIC_FILES.txt into a new directory without Git history."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


ROOT = Path(__file__).resolve().parents[1]


def export(destination):
    names = (ROOT / 'PUBLIC_FILES.txt').read_text().splitlines()
    if not names or names != sorted(set(names)):
        raise ValueError('PUBLIC_FILES.txt must contain sorted, unique relative file paths')
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError(f'Invalid public path: {name}')
        source = ROOT / name
        source.resolve().relative_to(ROOT)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Expected a regular source file: {name}')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for name in names:
        source, target = ROOT / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise RuntimeError(f'Export checksum mismatch: {name}')
        total += target.stat().st_size
    return dict(destination=str(destination), files=len(names), bytes=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are rejected')
    print(json.dumps(export(parser.parse_args().output), indent=2))


if __name__ == '__main__':
    main()
