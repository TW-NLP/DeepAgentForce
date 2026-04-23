#!/usr/bin/env python3
"""
Generate rounded-square application icon assets from images/logo.png.

This script creates a 1024x1024 rounded-square base icon and can emit:
- a PNG preview/base asset
- a macOS .iconset directory and .icns file
- a Windows .ico file
"""

from __future__ import annotations

import argparse
import tempfile
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


try:
    RESAMPLING = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:  # pragma: no cover
    RESAMPLING = Image.LANCZOS


def build_base_icon(source: Path, size: int = 1024) -> Image.Image:
    """Create a rounded-square icon with the logo centered on a light background."""
    logo = Image.open(source).convert("RGBA")

    # Keep the full logo visible and give it breathing room inside the square.
    max_logo_side = int(size * 0.82)
    logo.thumbnail((max_logo_side, max_logo_side), RESAMPLING)

    canvas = Image.new("RGBA", (size, size), (248, 250, 252, 255))
    radius = int(size * 0.22)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)

    # Apply the rounded mask to the background first, then place the logo on top.
    rounded = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rounded.paste(canvas, (0, 0), mask)

    x = (size - logo.width) // 2
    y = (size - logo.height) // 2
    rounded.alpha_composite(logo, dest=(x, y))
    return rounded


def save_png(icon: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    icon.save(target)


def save_ico(icon: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    icon.save(
        target,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def save_iconset(icon: Image.Image, iconset_dir: Path) -> None:
    iconset_dir.mkdir(parents=True, exist_ok=True)
    base_sizes = [16, 32, 128, 256, 512]
    for size in base_sizes:
        for scale in (1, 2):
            actual = size * scale
            suffix = "@2x" if scale == 2 else ""
            filename = f"icon_{size}x{size}{suffix}.png"
            icon.resize((actual, actual), RESAMPLING).save(iconset_dir / filename)


def save_icns(iconset_dir: Path, target: Path) -> None:
    import subprocess

    target.parent.mkdir(parents=True, exist_ok=True)
    if not Path("/usr/bin/iconutil").exists() and not shutil.which("iconutil"):
        raise SystemExit("iconutil is required on macOS to build .icns files.")
    subprocess.run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(target)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rounded-square app icon assets.")
    parser.add_argument("--source", required=True, help="Source logo PNG/SVG rasterized file")
    parser.add_argument("--png", help="Write the rounded-square base PNG")
    parser.add_argument("--ico", help="Write a Windows .ico file")
    parser.add_argument("--iconset", help="Write a macOS .iconset directory")
    parser.add_argument("--icns", help="Write a macOS .icns file")
    args = parser.parse_args()

    source = Path(args.source)
    icon = build_base_icon(source)

    if args.png:
        save_png(icon, Path(args.png))

    if args.ico:
        save_ico(icon, Path(args.ico))

    if args.iconset:
        save_iconset(icon, Path(args.iconset))

    if args.icns:
        iconset_dir = Path(args.iconset) if args.iconset else Path(tempfile.mkdtemp(prefix="deepagentforce-iconset-"))
        if not args.iconset:
            save_iconset(icon, iconset_dir)
        save_icns(iconset_dir, Path(args.icns))


if __name__ == "__main__":
    main()
