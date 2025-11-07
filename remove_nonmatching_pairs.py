#!/usr/bin/env python3
"""
remove_nonmatching_pairs.py

Compare two folders and delete files that do NOT have a same-name counterpart
in the other folder.

Features:
- Compare by full filename (including extension) or by stem (without extension).
- Recursive search (default).
- Ignore-case option for name matching.
- Dry-run mode to preview deletions.
- Backup mode: move deleted files to a backup directory instead of permanently
  deleting them.
- Filter by extensions.

Usage examples (PowerShell):
  # Dry-run, recursive, ignore case, match by filename
  python remove_nonmatching_pairs.py "C:\path\to\A" "C:\path\to\B" --dry-run --ignore-case

  # Actually delete (ask for confirmation)
  python remove_nonmatching_pairs.py A B

  # Move non-matching files into a backup folder instead of deleting
  python remove_nonmatching_pairs.py A B --backup-dir "C:\backup_nonmatches" --yes

"""
import argparse
from pathlib import Path
import shutil
import sys
from typing import List, Set, Dict

DEFAULT_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff']


def gather_files(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    if recursive:
        it = root.rglob('*')
    else:
        it = root.glob('*')
    files = [p for p in it if p.is_file() and (not exts or p.suffix.lower() in exts)]
    return files


def name_key(p: Path, match_stem: bool, ignore_case: bool) -> str:
    key = p.stem if match_stem else p.name
    return key.lower() if ignore_case else key


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def confirm(prompt: str) -> bool:
    ans = input(prompt + ' [y/N]: ').strip().lower()
    return ans == 'y' or ans == 'yes'


def main():
    parser = argparse.ArgumentParser(description='Delete files that do not have same-name counterparts in the other folder.')
    parser.add_argument('dir1', help='First directory')
    parser.add_argument('dir2', help='Second directory')
    parser.add_argument('--exts', nargs='+', help='Allowed extensions (e.g. .jpg .png). Default common image extensions', default=DEFAULT_EXTS)
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help='Do not search recursively')
    parser.add_argument('--recursive', dest='recursive', action='store_true', help='Search recursively (default)')
    parser.set_defaults(recursive=True)
    parser.add_argument('--ignore-case', action='store_true', help='Ignore case when comparing names')
    parser.add_argument('--match-stem', action='store_true', help='Match by filename stem (without extension)')
    parser.add_argument('--dry-run', action='store_true', help='Do not delete, only print what would be deleted')
    parser.add_argument('--backup-dir', help='If given, move files to this folder instead of deleting (keeps dir1/dir2 separation)')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    if not dir1.exists() or not dir1.is_dir():
        print(f'Error: {dir1} not found or not a directory')
        sys.exit(2)
    if not dir2.exists() or not dir2.is_dir():
        print(f'Error: {dir2} not found or not a directory')
        sys.exit(2)

    exts = [e.lower() if e.startswith('.') else '.' + e.lower() for e in args.exts] if args.exts else []

    files1 = gather_files(dir1, exts, args.recursive)
    files2 = gather_files(dir2, exts, args.recursive)

    set1: Set[str] = set(name_key(p, args.match_stem, args.ignore_case) for p in files1)
    set2: Set[str] = set(name_key(p, args.match_stem, args.ignore_case) for p in files2)

    only_in_1 = [p for p in files1 if name_key(p, args.match_stem, args.ignore_case) not in set2]
    only_in_2 = [p for p in files2 if name_key(p, args.match_stem, args.ignore_case) not in set1]

    print(f'Found {len(files1)} candidate files in {dir1} and {len(files2)} in {dir2}.')
    print(f'Files only in {dir1}: {len(only_in_1)}')
    print(f'Files only in {dir2}: {len(only_in_2)}')

    if len(only_in_1) == 0 and len(only_in_2) == 0:
        print('Nothing to do. Exiting.')
        return

    if args.dry_run:
        print('\nDRY-RUN: the following files would be removed (or moved to backup):')
        if only_in_1:
            print(f'-- Only in {dir1} ({len(only_in_1)}):')
            for p in only_in_1:
                print(str(p))
        if only_in_2:
            print(f'-- Only in {dir2} ({len(only_in_2)}):')
            for p in only_in_2:
                print(str(p))
        print('\nDRY-RUN complete. No files were changed.')
        return

    if not args.yes:
        print('\nAbout to delete/move these files:')
        print(f'  {len(only_in_1)} files from {dir1}\n  {len(only_in_2)} files from {dir2}')
        if not confirm('Proceed?'):
            print('Aborted by user.')
            return

    # perform deletion or move to backup
    total_deleted = 0
    total_failed = 0

    def process_list(lst: List[Path], src_root: Path, tag: str):
        nonlocal total_deleted, total_failed
        for p in lst:
            try:
                if args.backup_dir:
                    backup_root = Path(args.backup_dir)
                    target_base = backup_root / tag
                    # preserve relative path if inside src_root
                    try:
                        rel = p.relative_to(src_root)
                    except Exception:
                        rel = Path(p.name)
                    target = target_base / rel
                    ensure_dir(target.parent)
                    shutil.move(str(p), str(target))
                    print(f'MOVED: {p} -> {target}')
                else:
                    p.unlink()
                    print(f'DELETED: {p}')
                total_deleted += 1
            except Exception as e:
                print(f'FAILED: {p} -> {e}')
                total_failed += 1

    if only_in_1:
        process_list(only_in_1, dir1, 'only_in_dir1')
    if only_in_2:
        process_list(only_in_2, dir2, 'only_in_dir2')

    print(f'Finished. Deleted/moved: {total_deleted}. Failed: {total_failed}.')


if __name__ == '__main__':
    main()
