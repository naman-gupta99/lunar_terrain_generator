from elevation_map_generator import ElevationMapGenerator
from PIL import Image
import pandas as pd
import numpy as np


def save_heightmap(df, filename="heightmap.png"):
    # Invert
    df = -df

    # Normalize to 0–65535
    min_val = df.values.min()
    max_val = df.values.max()
    print(
        f"{filename} | Elevation Range: {min_val:.2f}m to {max_val:.2f}m | Δ={max_val - min_val:.2f}m"
    )

    norm = ((df.values - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

    img = Image.fromarray(norm)
    img.save(filename)

    # Return elevation delta for Unreal Z-scale
    return max_val - min_val

size_km = 1
pixels_per_km = 8129
for i in range(5, 6):
    generator = ElevationMapGenerator(size_km, pixels_per_km)
    generator.generate_elevation_map(f"craters_{i}.csv")
    generator.add_undulation()
    generator.plot_heatmap(f"heatmap_{i}.png")

    elevation_df = pd.DataFrame(generator.elevation)
    elevation_range = save_heightmap(elevation_df, f"heightmap_{i}.png")

    print(
        f"Use Z-scale = {(elevation_range * 1000) / 65536:.3f} in Unreal for heightmap_{i}",
        f"X-Y scale = {size_km/pixels_per_km * 10_000_000:.3f} in Unreal",
    )
