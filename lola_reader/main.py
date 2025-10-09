import matplotlib.pyplot as plt
import numpy as np

from read_img import read_square_by_center

LOW_RES_CORRECT_ALIGN_DEM = {
    "path": "ldem_85s_10m_float.tif",
    "center": (289.9258, -86.8542),
    "true_crs_src": "ldem_85s_10m_float.tif",
    "suffix": ""
}
HIGH_RES_CORRECT_ALIGN_DEM = {
    "path": "Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif",
    "center": (340.0742, -86.8542),
    "true_crs_src": "ldem_85s_10m_float.tif",
    "suffix": "_1m"
}

plot = HIGH_RES_CORRECT_ALIGN_DEM


def plot_zoom_sequence(
    src_path, true_crs_src, center, center_is_lonlat=True, lon_domain="auto", suffix=""
):
    """
    Generate a sequence of zoomed images centered on the same point.
    Uses appropriate downsampling for larger areas for better performance.
    """

    # Define zoom levels (width in meters) and their max display pixels for downsampling
    zoom_levels = [
        (16000, 2000),  # 16km area, max 2000 pixels
        (8000, 8000),  # 8km area, max 1500 pixels
        (4000, 4000),  # 4km area, max 1500 pixels
        (2000, 2000),  # 2km area, max 1500 pixels
        (1000, 1000),  # 1km area, max 1000 pixels
        (500, 500),  # 500m area, full resolution
    ]

    print("Generating zoom sequence...")
    for i, (width, max_pixels) in enumerate(zoom_levels):
        print(f"   Processing {width}m zoom level ({i+1}/{len(zoom_levels)})...")

        try:
            chip, chip_tr, chip_crs, win = read_square_by_center(
                src_path,
                true_crs_src=true_crs_src,
                center=center,
                width_m=width,
                lon_domain=lon_domain,
            )

            # Apply downsampling if needed
            original_shape = chip.shape
            if max(chip.shape) > max_pixels:
                downsample_factor = max(chip.shape) // max_pixels + 1
                print(
                    f"      Downsampling by factor of {downsample_factor} (from {original_shape} to approx {max_pixels} pixels)"
                )

                # Simple downsampling using numpy slicing (every nth pixel)
                chip = chip[::downsample_factor, ::downsample_factor]

                # Update transform for downsampled data
                chip_tr = rasterio.transform.Affine(  # type: ignore
                    chip_tr.a * downsample_factor,  # pixel width
                    chip_tr.b,
                    chip_tr.c,
                    chip_tr.d,
                    chip_tr.e * downsample_factor,  # pixel height (negative)
                    chip_tr.f,
                )

            # Create the figure
            plt.figure(figsize=(12, 10))

            # Calculate proper extent from the transform
            height, width_px = chip.shape
            # Get the bounds of the chip in the coordinate system
            left = chip_tr.c
            top = chip_tr.f
            right = left + width_px * chip_tr.a
            bottom = top + height * chip_tr.e

            # Create extent [left, right, bottom, top] for imshow
            extent: tuple[float, float, float, float] = (left, right, bottom, top)

            # Calculate local elevation range for this specific image
            local_min = np.nanmin(chip)
            local_max = np.nanmax(chip)
            print(f"      Local elevation range: {local_min:.1f} to {local_max:.1f} m")

            # Use terrain colormap with local elevation range for better contrast
            im = plt.imshow(
                chip,
                cmap="terrain",
                vmin=local_min,
                vmax=local_max,
                extent=extent,
                origin="upper",
            )

            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label("Elevation (m)", fontsize=12)

            # The center should be at the middle of the extent
            center_x = (left + right) / 2
            center_y = (bottom + top) / 2
            plt.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2)
            plt.plot(
                center_x,
                center_y,
                "ro",
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1,
            )

            # Add title and labels
            coord_str = (
                f"{center[1]:.4f}°S, {abs(center[0]-360):.4f}°W"
                if center_is_lonlat
                else f"({center[0]:.1f}, {center[1]:.1f}) m"
            )
            plt.title(
                f"Lunar DEM - {width}m × {width}m area\nCentered at {coord_str}",
                fontsize=14,
                pad=20,
            )

            plt.xlabel("X (meters)", fontsize=12)
            plt.ylabel("Y (meters)", fontsize=12)
            plt.grid(True, alpha=0.3)

            # Save with high DPI for better quality
            output_filename = f"{width}_meters{suffix}.png"
            plt.savefig(output_filename, dpi=200, bbox_inches="tight")

            # Save grayscale version
            grayscale_filename = f"{width}_meters{suffix}_grayscale.png"
            plt.figure(figsize=(12, 10))
            im_gray = plt.imshow(
                chip,
                cmap="gray",
                vmin=local_min,
                vmax=local_max,
                extent=extent,
                origin="upper",
            )

            # Add colorbar for grayscale
            cbar_gray = plt.colorbar(im_gray, shrink=0.8)
            cbar_gray.set_label("Elevation (m)", fontsize=12)

            # Add the same center markers and annotations
            plt.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2)
            plt.plot(
                center_x,
                center_y,
                "ro",
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1,
            )

            coord_str = (
                f"{center[1]:.4f}°S, {abs(center[0]-360):.4f}°W"
                if center_is_lonlat
                else f"({center[0]:.1f}, {center[1]:.1f}) m"
            )
            plt.title(
                f"Lunar DEM - {width}m × {width}m area (Grayscale)\nCentered at {coord_str}",
                fontsize=14,
                pad=20,
            )

            plt.xlabel("X (meters)", fontsize=12)
            plt.ylabel("Y (meters)", fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.savefig(grayscale_filename, dpi=200, bbox_inches="tight")
            plt.close()  # Close to save memory

            print(f"   ✓ Saved: {output_filename}")
            print(f"   ✓ Saved: {grayscale_filename}")

        except Exception as e:
            print(f"   ✗ Error processing {width}m level: {e}")
            continue

    print("4. Zoom sequence complete!")
    color_files = [f"{w}_meters{suffix}.png" for w, _ in zoom_levels]
    grayscale_files = [f"{w}_meters{suffix}_grayscale.png" for w, _ in zoom_levels]
    print(
        f"   Color files generated: full_lunar_dem_with_center{suffix}.png, {', '.join(color_files)}"
    )
    print(f"   Grayscale files generated: {', '.join(grayscale_files)}")


plot_zoom_sequence(
    plot["path"],
    true_crs_src=plot["true_crs_src"],
    center=plot["center"],
    center_is_lonlat=True,
    lon_domain="auto",
    suffix="_1m",
)
