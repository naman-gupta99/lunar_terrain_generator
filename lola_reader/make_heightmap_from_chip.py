#!/usr/bin/env python3
"""
Create a 1 km × 1 km heightmap PNG from a DEM chip using read_square_by_center.
Saves `heightmap.png` in the same folder.

Usage:
    python make_heightmap_from_chip.py --src PATH --lon 340.0742 --lat -86.8542 --swap_xy

Defaults target the high-res DEM used in this repo.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from read_img import read_square_by_center


DEFAULT_SRC = "Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif"
DEFAULT_CENTER = (340.0742, -86.8542)
DEFAULT_SWAP_XY = True


def save_heightmap_from_chip(chip: np.ndarray, chip_tr, out_path: Path) -> float:
    """Normalize chip (invert like generate_unreal.py) and save as 16-bit PNG.

    Returns elevation range (max - min) used for Z-scale calculation.
    """
    # Convert to float and handle NaNs
    data = chip.astype(float)

    # Invert elevations like in generate_unreal.py
    data = -data

    # Use nan-aware min/max
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    print(f"{out_path.name} | Elevation Range: {min_val:.2f}m to {max_val:.2f}m | \u0394={max_val - min_val:.2f}m")

    elev_range = max_val - min_val
    if elev_range == 0 or np.isnan(elev_range):
        print("Warning: elevation range is zero or NaN. Writing a zeroed heightmap.")
        norm = np.zeros_like(data, dtype=np.uint16)
    else:
        # Normalize to 0..65535
        norm = ((data - min_val) / elev_range * 65535.0)
        # Clip to avoid rounding issues
        norm = np.clip(norm, 0, 65535).astype(np.uint16)

    # Save using PIL (uint16 grayscale)
    img = Image.fromarray(norm)
    img.save(out_path)

    return elev_range


def main():
    p = argparse.ArgumentParser(description="Create 1km heightmap from DEM chip")
    p.add_argument("--src", default=DEFAULT_SRC, help="Path to DEM file")
    p.add_argument("--lon", type=float, default=DEFAULT_CENTER[0], help="Center longitude")
    p.add_argument("--lat", type=float, default=DEFAULT_CENTER[1], help="Center latitude")
    p.add_argument("--swap_xy", action="store_true", default=DEFAULT_SWAP_XY, help="Swap x/y when converting to pixel indices")
    p.add_argument("--out", default="heightmap.png", help="Output filename (PNG) saved inside lola_reader")
    args = p.parse_args()

    src_path = args.src
    center = (args.lon, args.lat)
    out_file = Path(args.out)

    # Read a 1 km x 1 km chip (no resampling)
    chip, chip_tr, chip_crs, win = read_square_by_center(
        src_path, center=center, width_m=1000.0, height_m=1000.0, lon_domain="auto", swap_xy=args.swap_xy
    )

    # Save heightmap implementing same normalization as generate_unreal.py
    elev_range = save_heightmap_from_chip(chip, chip_tr, out_file)

    # Compute X/Y pixel sizes from the transform
    px_w = abs(chip_tr.a)
    px_h = abs(chip_tr.e)

    print(f"Pixel size X (meters/pixel): {px_w:.6f}")
    print(f"Pixel size Y (meters/pixel): {px_h:.6f}")

    # Z-scale like generate_unreal.py
    if elev_range == 0 or np.isnan(elev_range):
        z_scale = 0.0
    else:
        z_scale = elev_range / 65536.0
    print(f"Use Z-scale = {z_scale:.6f} in Unreal for {out_file.name}")

    # Suggested X-Y scale for Unreal (match the repo's previous formula)
    # derive pixels_per_km = 1000 / meters_per_pixel -> X-Y scale = (1 / pixels_per_km) * 100000 = meters_per_pixel * 100
    suggested_xy_x = px_w * 100.0
    suggested_xy_y = px_h * 100.0
    print(f"Suggested X scale in Unreal ≈ {suggested_xy_x:.6f}")
    print(f"Suggested Y scale in Unreal ≈ {suggested_xy_y:.6f}")

    print(f"Saved heightmap to: {out_file}")


if __name__ == "__main__":
    main()
