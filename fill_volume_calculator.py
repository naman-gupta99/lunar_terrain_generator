import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elevation_profile import ProfileFactory
import math
from tqdm import tqdm
import glob

class FillVolumeCalculator:
    def __init__(self, crater_csv, size_km):
        self.pixels_per_km = 10000
        self.size_km = size_km
        self.image_size = int(size_km * self.pixels_per_km)
        self.elevation = np.zeros((self.size_km * self.pixels_per_km, self.size_km * self.pixels_per_km))

    def _meters_to_pixels(self, meters):
        km = meters / 1000
        return int(km * self.pixels_per_km)

    def generate_elevation_map(self, crater_csv, radius_threshold):
        df = pd.read_csv(crater_csv)
        self.elevation = np.zeros((self.image_size, self.image_size))

        for _, crater in df.iterrows():
            center_x = int(crater["x_km"] * self.pixels_per_km)
            center_y = int(crater["y_km"] * self.pixels_per_km)
            if crater["radius_m"] > radius_threshold:
                continue
            crater_radius_px = self._meters_to_pixels(crater["radius_m"])

            # Skip if radius is zero
            if crater_radius_px <= 0:
                continue

            radius_px = int(crater_radius_px * 2.5)

            # Create crater profile
            profile = ProfileFactory.create(
                "small", {"D": crater["radius_m"] * 2, "depth": crater["depth_m"]}
            )  # D is diameter in meters

            # Generate elevation grid
            y_coords, x_coords = np.ogrid[
                -radius_px : radius_px + 1, -radius_px : radius_px + 1
            ]
            dist_from_center = np.sqrt(x_coords**2 + y_coords**2)

            # Convert pixel distances to meters
            dist_meters = dist_from_center * (1000 / self.pixels_per_km)  # Convert to meters

            # Calculate crater depth using profile
            crater_depth = np.zeros_like(dist_from_center)
            valid_points = dist_meters <= (2.5 * crater["radius_m"])  # Profile valid up to 2.5R
            crater_depth[valid_points] = np.vectorize(profile.get_height)(dist_meters[valid_points])

            # Calculate the bounds for the crater
            y_min = max(0, center_y - radius_px)
            y_max = min(self.image_size, center_y + radius_px + 1)
            x_min = max(0, center_x - radius_px)
            x_max = min(self.image_size, center_x + radius_px + 1)

            # Calculate the corresponding bounds in the crater_depth array
            mask_y_min = max(0, -(center_y - radius_px))
            mask_y_max = crater_depth.shape[0] - max(0, (center_y + radius_px + 1) - self.image_size)
            mask_x_min = max(0, -(center_x - radius_px))
            mask_x_max = crater_depth.shape[1] - max(0, (center_x + radius_px + 1) - self.image_size)

            # Ensure the shapes match
            crater_section = crater_depth[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
            target_section = self.elevation[y_min:y_max, x_min:x_max]

            if crater_section.shape == target_section.shape:
                self.elevation[y_min:y_max, x_min:x_max] += crater_section

    def calculate_fill_volume(self, crater_csv, radius_threshold):
        self.generate_elevation_map(crater_csv, radius_threshold)
        
        # Calculate the number of 1000x10000 arrays that fit in the elevation map
        num_arrays_y = self.elevation.shape[0] // 1000
        
        # Initialize a list to store the sums of negative values for each array
        negative_sums = []
        
        # Iterate through each 1000x10000 array
        for y in range(num_arrays_y):
            # Extract the current array
            start_y = y * 1000
            end_y = (y + 1) * 1000
            start_x = 0
            end_x = 10000
            
            current_array = self.elevation[start_y:end_y, start_x:end_x]
            
            # Calculate sum of negative values
            negative_sum = np.sum(current_array[current_array < 0])
            negative_sums.append(-negative_sum*0.01)
        
        return negative_sums

    def plot_elevation_with_strips(self, negative_sums):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot the elevation map as a heatmap
        im = ax.imshow(self.elevation, cmap='terrain', aspect='auto')
        plt.colorbar(im, ax=ax, label='Elevation')
        
        # Add horizontal lines for each strip
        for y in range(self.elevation.shape[0] // 1000):
            y_pos = y * 1000
            ax.axhline(y=y_pos, color='red', linestyle='--', alpha=0.5)
            
            # Add the negative sum value as text
            if y < len(negative_sums):
                ax.text(0, y_pos + 500, f'Sum: {negative_sums[y]:.2f}', 
                       color='white', fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.5))
        
        # Add a final line at the bottom
        ax.axhline(y=self.elevation.shape[0], color='red', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Elevation Map with Strip Volumes')
        
        plt.tight_layout()
        plt.show()

    def plot_negative_sums_boxplot(self, negative_sums):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the box plot
        box = ax.boxplot(negative_sums, 
                        patch_artist=True,
                        showfliers=True,
                        vert=True)
        
        # Customize the box plot colors
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('blue')
        
        for whisker in box['whiskers']:
            whisker.set(color='blue', linewidth=1.5)
        
        for cap in box['caps']:
            cap.set(color='blue', linewidth=1.5)
        
        for median in box['medians']:
            median.set(color='red', linewidth=2)
        
        # Add labels and title
        ax.set_ylabel('Negative Sum Volume')
        ax.set_title('Distribution of Negative Sums Across Strips')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean value as a horizontal line
        mean_value = np.mean(negative_sums)
        ax.axhline(y=mean_value, color='green', linestyle='--', 
                  label=f'Mean: {mean_value:.2f}')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def analyze_multiple_files(self, crater_files_pattern, radius_thresholds):
        # Get all crater files matching the pattern
        crater_files = glob.glob(crater_files_pattern)
        
        # Initialize DataFrame to store results
        results_df = pd.DataFrame(columns=['file', 'radius_threshold', 'volume'])
        
        # Process each file
        for crater_file in tqdm(crater_files, desc="Processing files"):
            # Process each radius threshold
            for threshold in tqdm(radius_thresholds, desc=f"Processing {crater_file}", leave=False):
                # Calculate negative sums for current file and threshold
                negative_sums = self.calculate_fill_volume(crater_file, threshold)
                
                # Add results to DataFrame
                for volume in negative_sums:
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'file': [crater_file],
                        'radius_threshold': [threshold],
                        'volume': [volume]
                    })], ignore_index=True)
        
        return results_df

    def plot_comparative_boxplots(self, results_df):
        # Create the figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for threshold in sorted(results_df['radius_threshold'].unique()):
            threshold_data = results_df[results_df['radius_threshold'] == threshold]['volume']
            data.append(threshold_data)
            if threshold == math.inf:
                labels.append('R ≤ ∞')
            else:
                labels.append(f'R ≤ {threshold}m')
        
        # Create the box plot
        box = ax.boxplot(data, 
                        patch_artist=True,
                        showfliers=True,
                        vert=True,
                        labels=labels)
        
        # Customize the box plot colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('blue')
            patch.set_alpha(0.7)
        
        for whisker in box['whiskers']:
            whisker.set(color='blue', linewidth=1.5)
        
        for cap in box['caps']:
            cap.set(color='blue', linewidth=1.5)
        
        for median in box['medians']:
            median.set(color='red', linewidth=2)
        
        # Add labels and title
        ax.set_ylabel('Volume')
        ax.set_title('Distribution of Volumes for Different Radius Thresholds\n(Across All Crater Files)')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean values as points
        for i, threshold_data in enumerate(data):
            mean_value = threshold_data.mean()
            ax.plot(i + 1, mean_value, 'g*', markersize=10, 
                   label=f'Mean: {mean_value:.2f}' if i == 0 else None)
        
        # Add legend
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Example usage
fill_volume_calculator = FillVolumeCalculator("craters_1.csv", 1)
radius_thresholds = [1, 1.2, 1.4, 1.6, 1.8, 2]

# Analyze all crater files and create DataFrame
results_df = fill_volume_calculator.analyze_multiple_files("craters_*.csv", radius_thresholds)

# Plot the results
fill_volume_calculator.plot_comparative_boxplots(results_df)

# Save the results to CSV
results_df.to_csv('volume_analysis_results_2.csv', index=False)