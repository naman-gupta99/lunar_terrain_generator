"""
Based on the small crater profile described in:
https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021GL095537

"""

import numpy as np
from .elevation_profile import ElevationProfile

class SmallCraterElevationProfile(ElevationProfile):
    """Small crater profile based on https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021GL095537"""
    
    def __init__(self, parameters):
        super().__init__(parameters)
        if 'D' not in parameters:
            raise ValueError("Parameter 'D' (diameter) is required")
        
        self.D = parameters['D']  # Diameter in meters
        self.R = self.D / 2      # Radius in meters
        
        # Constants for small crater profile
        self.a = -2.8567
        self.b = 5.8270
        self.alpha = -3.1906
        
        # Diameter-dependent parameters
        self.d_0 = 0.114 * (self.D ** -0.002)
        self.h_t = 0.02513 * (self.D ** -0.0757)
        self.C = self.d_0 * (np.exp(self.a) + 1) / (np.exp(self.b) - 1)
    
    def get_height(self, r: float) -> float:
        """Get height at radius r.
        
        Args:
            r: Radius in meters from crater center
            
        Returns:
            Height in meters at radius r
        """
        # Convert to normalized radius for formula
        r_norm = r / self.R
        if r_norm <= 1:
            temp = self.C * (np.exp(self.b * r_norm) - np.exp(self.b)) / (1 + np.exp(self.a + self.b * r_norm))
            return (self.D * self.h_t + temp * self.R) * 1000
        else:
            return self.D * self.h_t * ((r_norm ** self.alpha)) * 1000
    
    def slope(self, r_norm: float) -> float:
        """Get slope at normalized radius r_norm.
        
        Args:
            r_norm: Normalized radius (r / R)
            
        Returns:
            Slope at normalized radius r_norm
        """
        if r_norm <= 1:
            return self.C * self.b * np.exp(self.b * r_norm) / (1 + np.exp(self.a + self.b * r_norm))**2
        else:
            return self.D * self.h_t * self.alpha * (r_norm ** (self.alpha - 1))
    
    def get_min_traversable_normalized_radius(self, traversable_slope_degrees: int) -> float:
        """Get the maximum traversable radius.
        
        Returns:
            The r_norm at which the slope is cos(80°) and r_norm > 1
        """
        cos_slope = np.cos(np.radians(traversable_slope_degrees))
        l, r = 1, 100
        while r - l > 0.01:
            mid = (l + r) / 2
            s = self.slope(mid)
            if s < cos_slope:
                l = mid
            else:
                r = mid
        return l