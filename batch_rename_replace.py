#!/usr/bin/env python3
"""
batch_rename_replace.py

批量替换文件名中的指定子串。

功能:
- 在指定目录（可递归）中，把文件名中的某个子串替换为另一个字符串
- 仅替换文件名（不更改扩展名）
- 支持忽略大小写匹配、dry-run（模拟）、确认提示
- 支持扩展名过滤（默认常见图片格式）
- 冲突处理策略: skip / overwrite / unique（自动加序号）

示例（PowerShell）:
  # 将目录中所有文件名中的 ir 替换为 _（干运行）
  python batch_rename_replace.py "G:\path\to\dir" ir _ --dry-run --ignore-case

  # 递归替换并在冲突时自动改名
  python batch_rename_replace.py A ir _ --on-conflict unique

"""
from pathlib import Path
import argparse
import sys
import re
from typing import List

DEFAULT_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff']


def gather_files(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    if recursive:
        it = root.rglob('*')
    else:
        it = root.glob('*')
    files = [p for p in it if p.is_file() and (not exts or p.suffix.lower() in exts)]
    return files


def unique_target(path: Path) -> Path:
    """如果目标已存在，则加后缀 _1, _2 ... 直到不冲突"""
    parent = path.parent
    stem = path.stem
    suf = path.suffix
    i = 1
    candidate = path
    while candidate.exists():
        candidate = parent / f"{stem}_{i}{suf}"
        i += 1
    return candidate


def rename_file(src: Path, dst: Path, on_conflict: str, dry_run: bool) -> bool:
    """执行重命名；返回 True 表示实际或模拟成功，False 表示跳过/失败"""
    if src.resolve() == dst.resolve():
        return False
    if dst.exists():
        if on_conflict == 'skip':
            print(f'SKIP (exists): {dst}')
            return False
        elif on_conflict == 'overwrite':
            if dry_run:
                print(f'[DRY-RUN] OVERWRITE: {src} -> {dst}')
                return True
            try:
                dst.unlink()
            except Exception as e:
                print(f'FAIL remove existing {dst}: {e}')
                return False
        elif on_conflict == 'unique':
            dst = unique_target(dst)

    if dry_run:
        print(f'[DRY-RUN] {src} -> {dst}')
        return True

    try:
        src.rename(dst)
        print(f'RENAME: {src} -> {dst}')
        return True
    except Exception as e:
        print(f'FAIL: {src} -> {dst} ({e})')
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch replace substring in filenames')
    parser.add_argument('root', help='Directory to operate on')
    parser.add_argument('old', help='Old substring or regex to replace (literal string by default)')
    parser.add_argument('new', help='Replacement string')
    parser.add_argument('--exts', nargs='+', help='Extensions to include (e.g. .jpg .png). Default common image formats', default=DEFAULT_EXTS)
    parser.add_argument('--no-recursive', dest='recursive', action='store_false', help='Do not search recursively')
    parser.add_argument('--recursive', dest='recursive', action='store_true', help='Search recursively (default)')
    parser.set_defaults(recursive=True)
    parser.add_argument('--ignore-case', action='store_true', help='Ignore case when matching the old substring')
    parser.add_argument('--use-regex', action='store_true', help='Treat old as a regular expression')
    parser.add_argument('--dry-run', action='store_true', help='Only print planned renames')
    parser.add_argument('--on-conflict', choices=['skip', 'overwrite', 'unique'], default='unique', help='How to handle filename conflicts')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f'Error: {root} is not a directory')
        sys.exit(2)

    exts = [e.lower() if e.startswith('.') else '.' + e.lower() for e in args.exts] if args.exts else []

    files = gather_files(root, exts, args.recursive)
    print(f'候选文件数量: {len(files)}')

    flags = re.IGNORECASE if args.ignore_case else 0
    pattern = re.compile(args.old, flags) if args.use_regex else None

    to_rename = []
    for p in files:
        stem = p.stem
        if args.use_regex:
            if pattern.search(stem):
                new_stem = pattern.sub(args.new, stem)
            else:
                continue
        else:
            if args.ignore_case:
                if args.old.lower() in stem.lower():
                    # perform case-insensitive replace while keeping original case aside from replacement
                    # simple strategy: use regex with IGNORECASE to replace
                    new_stem = re.sub(re.escape(args.old), args.new, stem, flags=re.IGNORECASE)
                else:
                    continue
            else:
                if args.old in stem:
                    new_stem = stem.replace(args.old, args.new)
                else:
                    continue

        new_name = new_stem + p.suffix
        dst = p.with_name(new_name)
        if dst.resolve() == p.resolve():
            continue
        to_rename.append((p, dst))

    if not to_rename:
        print('No files to rename.')
        return

    print(f'将要重命名 {len(to_rename)} 个文件')
    if args.dry_run:
        for s, d in to_rename:
            print(f'[DRY-RUN] {s} -> {d}')
        print('DRY-RUN 完成。')
        return

    if not args.yes:
        ans = input(f'确认执行重命名 {len(to_rename)} 个文件？ [y/N]: ').strip().lower()
        if ans not in ('y', 'yes'):
            print('已取消')
            return

    success = 0
    failed = 0
    for s, d in to_rename:
        ok = rename_file(s, d, args.on_conflict, args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1

    print(f'完成。成功: {success}，失败: {failed}')


if __name__ == '__main__':
    main()
