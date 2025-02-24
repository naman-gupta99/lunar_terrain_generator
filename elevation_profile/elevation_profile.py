from abc import ABC, abstractmethod
from typing import Dict, Union

class ElevationProfile(ABC):
    """Base class for crater elevation profiles.
    
    Parameters Dictionary:
        Required parameters depend on the specific profile type.
        Common parameters include:
        - D: Crater diameter in meters
        - depth: Maximum depth at crater center in meters
    """
    
    def __init__(self, parameters: Dict[str, float]):
        """Initialize elevation profile with parameters.
        
        Args:
            parameters: Dictionary containing profile parameters
        """
        self.parameters = parameters
    
    @abstractmethod
    def get_height(self, r: float) -> float:
        """Get height at normalized radius r.
        
        Args:
            r: Normalized radius (r/R where R is crater radius)
            
        Returns:
            Height at normalized radius r
        """
        pass