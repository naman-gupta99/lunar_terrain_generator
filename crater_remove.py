import pandas as pd

# Map dimensions from generate_unreal.py
size_km = 1
pixels_per_km = 10000
total_pixels = size_km * pixels_per_km

# Define the pixel-based bounding box for the slice
px_x_min, px_x_max = 3200, 3600
px_y_min, px_y_max = 3200, 3600

# Convert pixel coordinates to km coordinates
# The crater coordinates are normalized (0 to 1 for a 1km map)
km_x_min = px_x_min / total_pixels
km_x_max = px_x_max / total_pixels
km_y_min = px_y_min / total_pixels
km_y_max = px_y_max / total_pixels

input_file = 'craters_5.csv'
output_file = 'craters_5_sliced.csv'

try:
    # Read the crater data
    craters_df = pd.read_csv(input_file)

    # Filter craters whose centers are within the bounding box in km
    filtered_craters = craters_df[
        (craters_df['x_km'] >= km_x_min) &
        (craters_df['x_km'] < km_x_max) &
        (craters_df['y_km'] >= km_y_min) &
        (craters_df['y_km'] < km_y_max)
    ]

    # Save the filtered craters to a new CSV file
    filtered_craters.to_csv(output_file, index=False)

    print(f"Original number of craters: {len(craters_df)}")
    print(f"Number of craters in slice: {len(filtered_craters)}")
    print(f"Filtered crater data saved to '{output_file}'")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except KeyError as e:
    print(f"Error: Missing expected column in '{input_file}': {e}")
    print("The CSV file is expected to have 'x_km' and 'y_km' columns.")