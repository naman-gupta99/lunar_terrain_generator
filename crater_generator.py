import numpy as np
import pandas as pd

class CraterGenerator:
    def __init__(self, size_km=10, distribution_variance=0.2):
        self.size_km = size_km
        self.distribution_variance = distribution_variance
        self.area = size_km * size_km
        
        # Crater size distribution (craters per km²)
        self.size_distribution = {
            0: 10000,  # 1m
            1: 100,    # 10m
            2: 10,     # 100m
            3: 0.1,    # 1km
            4: 0.001,  # 10km
        }

    def _apply_distribution_variance(self, base_frequency):
        variance_factor = 1 + np.random.uniform(
            -self.distribution_variance,
            self.distribution_variance
        )
        return base_frequency * variance_factor

    def _calculate_crater_counts(self):
        crater_counts = {}
        for size_exp, freq_per_km2 in self.size_distribution.items():
            base_count = int(freq_per_km2 * self.area)
            varied_count = int(self._apply_distribution_variance(base_count))
            crater_counts[size_exp] = varied_count
        return crater_counts

    def _generate_crater_size(self, size_exp):
        base_size = 10 ** size_exp
        variation = np.random.uniform(0.1, 1.0)
        return (base_size * variation) / 2

    def _get_crater_depth(self, radius):
        d = radius * 2
        return 0.03736 * (d ** 1.069)

    def generate_craters(self):
        crater_data = []
        crater_counts = self._calculate_crater_counts()
        
        for size_exp, count in crater_counts.items():
            print(f"Generating {count} craters of size 10^{size_exp}m")
            for _ in range(count):
                x = np.random.uniform(0, self.size_km)
                y = np.random.uniform(0, self.size_km)
                radius = self._generate_crater_size(size_exp)
                depth = self._get_crater_depth(radius)
                crater_data.append({
                    'x_km': x,
                    'y_km': y,
                    'radius_m': radius,
                    'depth_m': depth
                })
        
        return pd.DataFrame(crater_data)

    def save_craters(self, filename):
        df = self.generate_craters()
        df.to_csv(filename, index=False)
