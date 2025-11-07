#!/usr/bin/env python
"""extract_images_by_name.py

从源目录递归查找文件名包含指定子串的图片并复制到目标目录。

特性：
- 支持递归（默认）
- 支持 dry-run（只列出将被复制的文件）
- 可选择保留源目录相对结构（--preserve-structure）或扁平化复制（默认）
- 忽略大小写匹配
- 覆盖行为可选（--overwrite）或自动改名避免冲突

示例：
python extract_images_by_name.py "g:\\2025_yonghao_camera" "e:\\document\\campus\\MMIF-CDDFuse-main\\extracted_ir" --name ir --recursive --dry-run --ignore-case
"""
import argparse
from pathlib import Path
import shutil
import sys

DEFAULT_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff'}


def gather_matches(root: Path, name_substr: str, exts, recursive: bool, case_insensitive: bool):
    if case_insensitive:
        name_substr = name_substr.lower()
    it = root.rglob('*') if recursive else root.iterdir()
    matches = []
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        fname = p.name
        if case_insensitive:
            fname = fname.lower()
        if name_substr in fname:
            matches.append(p)
    return matches


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def unique_target_path(dst: Path):
    """If dst exists, return a new Path with suffix _1, _2 ..."""
    if not dst.exists():
        return dst
    parent = dst.parent
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_files(matches, src_root: Path, dst_root: Path, preserve_structure: bool, overwrite: bool, dry_run: bool):
    copied = 0
    skipped = 0
    details = []
    for src in matches:
        if preserve_structure:
            try:
                rel = src.relative_to(src_root)
            except Exception:
                # fallback: use name only
                rel = Path(src.name)
            dst = dst_root / rel
            ensure_dir(dst.parent)
        else:
            dst = dst_root / src.name
            ensure_dir(dst_root)

        if dst.exists() and not overwrite:
            dst_final = unique_target_path(dst)
        else:
            dst_final = dst

        details.append((src, dst_final))

    # perform copy (or dry-run)
    for src, dst in details:
        if dry_run:
            print(f"[DRY-RUN] {src} -> {dst}")
            copied += 1
            continue
        try:
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"复制失败: {src} -> {dst} : {e}", file=sys.stderr)
            skipped += 1

    return copied, skipped


def main():
    ap = argparse.ArgumentParser(description='复制源目录中所有子目录下文件名包含指定子串的图片到目标目录')
    ap.add_argument('src', help='源目录路径')
    ap.add_argument('dst', help='目标目录路径（将被创建）')
    ap.add_argument('--name', '-n', default='ir', help="要匹配的子串，默认 'ir'")
    ap.add_argument('--exts', nargs='+', help='扩展名列表，例如 .png .jpg（默认常见图片格式）')
    ap.add_argument('--recursive', '-r', action='store_true', help='递归子目录（已启用）')
    ap.add_argument('--no-recursive', dest='recursive', action='store_false', help='不递归（仅当前目录）')
    ap.set_defaults(recursive=True)
    ap.add_argument('--preserve-structure', action='store_true', help='在目标目录中保留源的相对目录结构')
    ap.add_argument('--overwrite', action='store_true', help='如果目标文件已存在则覆盖')
    ap.add_argument('--dry-run', action='store_true', help='仅列出将被复制的文件，不执行实际复制')
    ap.add_argument('--ignore-case', action='store_true', help='忽略大小写匹配')
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists() or not src_root.is_dir():
        print('错误：源目录不存在或不是目录。', file=sys.stderr)
        sys.exit(2)

    exts = set((e.lower() if e.startswith('.') else f'.{e.lower()}') for e in (args.exts or []))
    if not exts:
        exts = DEFAULT_EXTS

    matches = gather_matches(src_root, args.name, exts, args.recursive, args.ignore_case)

    if not matches:
        print('未找到匹配的图片文件。')
        return

    print(f'共找到 {len(matches)} 个匹配文件。')

    # if dry-run, just list and exit after copying simulation
    copied, skipped = copy_files(matches, src_root, dst_root, args.preserve_structure, args.overwrite, args.dry_run)

    if args.dry_run:
        print(f"\nDRY-RUN: 将会复制 {copied} 个文件（未实际复制）。")
    else:
        print(f"\n完成：已复制 {copied} 个文件，失败 {skipped} 个（如果有）。")


if __name__ == '__main__':
    main()
