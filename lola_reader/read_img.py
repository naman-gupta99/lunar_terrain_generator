import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from pyproj import CRS, Transformer


def _lunar_geog_from_proj(proj_crs: CRS) -> CRS:
    """
    Build a lunar geographic CRS (long/lat) using the ellipsoid from the dataset's projected CRS.
    Works for spherical & ellipsoidal lunar products.
    """
    # Get semi-major/minor from the dataset CRS
    c = (
        CRS.from_wkt(proj_crs.to_wkt())
        if isinstance(proj_crs, rasterio.crs.CRS)
        else proj_crs
    )
    a = c.ellipsoid.semi_major_metre
    b = c.ellipsoid.semi_minor_metre
    # Construct a long/lat CRS on the same body (no Earth EPSG here!)
    return CRS.from_proj4(f"+proj=longlat +a={a} +b={b} +no_defs +type=crs")


def _lon_to_360(lon_deg):
    """Convert -180..180 to 0..360 if needed."""
    return lon_deg if lon_deg >= 0 else lon_deg + 360.0


def read_square_by_center(
    src_path, center, width_m, height_m=None, lon_domain="auto", swap_xy=False
):
    """
    Read a chip centered at a given coordinate with width/height in meters.
    center: (lon, lat) if center_is_lonlat=True; otherwise (x, y) in dataset CRS.
    lon_domain: "auto" | "360" | "180"  (controls how we normalize input lon if center_is_lonlat=True)
    swap_xy: if True, treats the projected x/y as (row, col) instead of (col, row) when
             converting to pixel indices – useful for rasters whose axis order is flipped.
    """
    height_m = width_m if height_m is None else height_m

    with rasterio.open(src_path, "r") as src:
        dst_crs = src.crs
        if dst_crs is None:
            raise ValueError(f"Dataset {src_path} has no CRS defined.")
        geog = _lunar_geog_from_proj(dst_crs)
        lon, lat = center
        if lon_domain == "360" or (
            lon_domain == "auto" and "0_360" in (dst_crs.to_string() or "")
        ):
            lon = _lon_to_360(lon)
        transformer = Transformer.from_crs(geog, dst_crs, always_xy=True)
        x_c, y_c = transformer.transform(lon, lat)
        print(f"   Center in dataset CRS: ({x_c:.2f}, {y_c:.2f})")

        px_w = abs(src.transform.a)
        px_h = abs(src.transform.e)
        w_px = width_m / px_w
        h_px = height_m / px_h
        
        print(f"   Window size in pixels: ({w_px:.1f}, {h_px:.1f}) at {px_w:.2f}m/px")

        if swap_xy:
            row_c, col_c = src.index(x_c, y_c)
        else:
            col_c, row_c = src.index(x_c, y_c)

        col_off = col_c - w_px / 2.0
        row_off = row_c - h_px / 2.0

        win = (
            Window(col_off=col_off, row_off=row_off, width=w_px, height=h_px)
            .round_offsets()
            .round_lengths()
        )

        chip = src.read(
            1,
            window=win,
            out_dtype="float32",
            boundless=True,
            fill_value=float("nan"),
        )
        chip_transform = window_transform(win, src.transform)

        width = win.width
        height = win.height
        ul_x = chip_transform.c
        ul_y = chip_transform.f
        ur_x = ul_x + chip_transform.a * width
        ur_y = ul_y
        ll_x = ul_x
        ll_y = ul_y + chip_transform.e * height
        lr_x = ur_x
        lr_y = ll_y
        print(
            "   Window corners (projected CRS): "
            f"UL=({ul_x:.2f}, {ul_y:.2f}), "
            f"UR=({ur_x:.2f}, {ur_y:.2f}), "
            f"LL=({ll_x:.2f}, {ll_y:.2f}), "
            f"LR=({lr_x:.2f}, {lr_y:.2f})"
        )

        return chip, chip_transform, dst_crs, win
