import numpy as np
from .elevation_profile import ElevationProfile

class BasicCraterElevationProfile(ElevationProfile):
    """Basic crater profile using parabolic shape.
    
    Uses a simple parabolic function: h(r) = depth * (1 - (r/R)^2) for r <= R,
    where R is the crater radius and depth is the maximum depth at center.
    """
    
    def __init__(self, parameters):
        super().__init__(parameters)
        if 'D' not in parameters:
            raise ValueError("Parameter 'D' (diameter) is required")
        if 'depth' not in parameters:
            raise ValueError("Parameter 'depth' (center depth) is required")
        
        self.D = parameters['D']        # Diameter in meters
        self.R = self.D / 2            # Radius in meters
        self.depth = parameters['depth'] # Depth at center in meters
    
    def get_height(self, r: float) -> float:
        """Get height at radius r.
        
        Args:
            r: Radius in meters from crater center
            
        Returns:
            Height in meters at radius r (negative for depression)
        """
        if r <= self.R:
            # Parabolic profile inside crater
            return self.depth * min(1, max(0, (1 - (r/self.R)**2))) * 1000
        else:
            # Zero elevation outside crater
            return 0
