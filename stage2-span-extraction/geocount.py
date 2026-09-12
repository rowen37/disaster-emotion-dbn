"""
Stage 2 - auxiliary: geographic heat map of negative-emotion posts.

Reads a span-extraction output CSV, keeps posts whose dominant emotion is
angry/fear/sad with valid coordinates, counts points on a regular
lon/lat grid, and renders a log-scale heat map (geographic footprint of
the negative-emotion sample feeding Stage 3).
"""
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(
        description="Count post coordinates from a CSV on a regular grid and draw a heat map"
    )
    parser.add_argument("csvfile", help="span_output_new/101.csv")
    parser.add_argument(
        "--lon_col", default="lon", help="longitude column name (default: lon)"
    )
    parser.add_argument(
        "--lat_col", default="lat", help="latitude column name (default: lat)"
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.1, help="grid cell size in degrees (default: 0.1)"
    )
    parser.add_argument(
        "--output", default="geocount_grid.png", help="output image file name (default: geocount_grid.png)"
    )
    args = parser.parse_args()

    # 1. load the CSV
    try:
        df = pd.read_csv(args.csvfile)
    except Exception as e:
        print(f"错误：无法读取文件 {args.csvfile}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.lon_col not in df.columns or args.lat_col not in df.columns:
        print(
            f"错误：在CSV中找不到列 '{args.lon_col}' 或 '{args.lat_col}'。"
            f"可用列: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. keep valid coordinates and filter by emotion
    if 'EMO' not in df.columns:
        print("错误：CSV中缺少 'EMO' 列。", file=sys.stderr)
        sys.exit(1)

    # keep rows whose emotion is angry, fear or sad (case/whitespace insensitive)
    valid_emos = {'angry', 'fear', 'sad'}
    emo_mask = df['EMO'].astype(str).str.strip().str.lower().isin(valid_emos)

    lon = pd.to_numeric(df[args.lon_col], errors="coerce")
    lat = pd.to_numeric(df[args.lat_col], errors="coerce")

    # require: matching emotion + valid coordinates
    mask = lon.notna() & lat.notna() & emo_mask
    lon = lon[mask].values
    lat = lat[mask].values

    # 3. grid edges (padded outward so every point is covered)
    grid_size = args.grid_size
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    eps = 1e-9
    lon_bins = np.arange(
        np.floor(lon_min / grid_size) * grid_size,
        np.ceil(lon_max / grid_size) * grid_size + grid_size / 2,
        grid_size,
    )
    lat_bins = np.arange(
        np.floor(lat_min / grid_size) * grid_size,
        np.ceil(lat_max / grid_size) * grid_size + grid_size / 2,
        grid_size,
    )

    # 4. 2-D histogram
    counts, xedges, yedges = np.histogram2d(lon, lat, bins=(lon_bins, lat_bins))

    # 5. heat map
    fig, ax = plt.subplots(figsize=(10, 8))
    # pcolormesh expects C of shape (ny, nx): transpose the counts
    norm = mcolors.LogNorm(vmin=1, vmax=counts.max())  # zero cells fall back to the face color
    mesh = ax.pcolormesh(xedges, yedges, counts.T, cmap='plasma',
                        norm=norm, edgecolors='face')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Point count per {grid_size}°×{grid_size}° grid cell")
    fig.colorbar(mesh, ax=ax, label="Count")
    ax.set_aspect("equal")  # equal aspect for lon/lat

    # 6. save
    try:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"网格图已保存至: {args.output}")
    except Exception as e:
        print(f"错误：保存图像失败 {e}", file=sys.stderr)
        sys.exit(1)

    # plt.show()


if __name__ == "__main__":
    main()
