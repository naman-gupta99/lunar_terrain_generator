import numpy as np
import matplotlib.pyplot as plt
from elevation_profile.small_crater_elevation_profile import SmallCraterElevationProfile

def main():
    # Create a range of crater diameters to analyze with exponential distribution
    # Starting from a small positive value instead of 0 to avoid potential issues
    diameters = np.logspace(0, 4, 40)  # Exponential range from 1 to 10000
    
    angles = np.arange(100, 185, 10)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define a colormap for the lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    
    # For each angle, calculate and plot a line
    for i, angle in enumerate(angles):
        # Calculate max traversable radius for each diameter at this angle
        max_traversable_radii = []
        for diameter in diameters:
            try:
                profile = SmallCraterElevationProfile({'D': diameter})
                max_traversable_radii.append(profile.get_min_traversable_normalized_radius(angle))
            except Exception as e:
                print(f"Error with diameter {diameter} at angle {angle}: {e}")
                max_traversable_radii.append(np.nan)  # Use NaN for failed calculations
        
        # Plot this angle's line
        plt.semilogx(diameters, max_traversable_radii, '-', 
                    linewidth=2, color=colors[i], 
                    label=f'{angle}°')
    
    # Add labels and title
    plt.xlabel('Crater Diameter (meters) - Log Scale', fontsize=12)
    plt.ylabel('Minimum Traversable Normalized Radius', fontsize=12)
    plt.title('Crater Traversability at Different Angles', fontsize=14)
    
    # Add legend
    plt.legend(title='Traversable Angle', loc='best')
    
    # Add grid and improve appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show the plot
    plt.savefig('crater_traversability_multiple_angles.png')
    plt.show()
    
    # Print example values for reference (just for the first angle)
    angle = angles[0]
    max_traversable_radii = []
    for diameter in diameters:
        try:
            profile = SmallCraterElevationProfile({'D': diameter})
            max_traversable_radii.append(profile.get_min_traversable_normalized_radius(angle))
        except Exception:
            max_traversable_radii.append(np.nan)
    
    print(f"Example values for angle {angle}°:")
    print("Diameter (m) | Max Traversable Normalized Radius")
    print("-" * 45)
    for i in range(0, len(diameters), len(diameters)//5):
        print(f"{diameters[i]:.1f} | {max_traversable_radii[i]:.4f}")
    
if __name__ == "__main__":
    main()