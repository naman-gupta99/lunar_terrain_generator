from elevation_profile.small_crater_elevation_profile import SmallCraterElevationProfile
import numpy as np
import matplotlib.pyplot as plt

profile = SmallCraterElevationProfile(
    {"D": 2.65 * 2, "depth": 2.3}
)

# Generate radius values
radius_values = np.linspace(-5.65, 5.65, 100)
# Calculate elevation profile
elevation_arr = [profile.get_height(abs(r))/100 for r in radius_values]
# Calculate slope profile using numerical differentiation of elevation
dr = radius_values[1] - radius_values[0]  # spacing between radius points
slope_arr = np.gradient(elevation_arr, dr)  # dy/dx
slope_arr = np.degrees(np.arctan(slope_arr))  # convert to degrees
slope_arr = np.abs(slope_arr)  # take absolute value for slope

# Create plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Create secondary y-axis for slope first
ax2 = ax1.twinx()
line2 = ax2.plot(radius_values, slope_arr, 'c-', linewidth=2, label='Slope')
ax2.set_ylabel("Slope (degrees)", color='c')
ax2.tick_params(axis='y', labelcolor='c')

# Plot elevation on primary y-axis (on top)
line1 = ax1.plot(radius_values, elevation_arr, 'b-', linewidth=2, label='Height')
ax1.set_xlabel("Radius (m)")
ax1.set_ylabel("Height (m)", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_zorder(ax2.get_zorder() + 1)  # Bring ax1 to front
ax1.patch.set_visible(False)  # Make ax1 background transparent

# Add title and legend
plt.title("Small Crater Elevation and Slope Profile")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.show()