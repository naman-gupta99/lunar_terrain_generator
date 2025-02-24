from typing import Dict, Type
from .elevation_profile import ElevationProfile
from .small_crater_elevation_profile import SmallCraterElevationProfile
from .basic_crater_elevation_profile import BasicCraterElevationProfile

class ProfileFactory:
    """Factory for creating crater elevation profiles."""
    
    _profiles = {
        'small': SmallCraterElevationProfile,
        'basic': BasicCraterElevationProfile
    }
    
    @classmethod
    def create(cls, profile_type: str, parameters: Dict[str, float]) -> ElevationProfile:
        """Create a crater elevation profile.
        
        Args:
            profile_type: Type of profile ('small' for SmallCraterElevationProfile)
            parameters: Dictionary of parameters for the profile
            
        Returns:
            ElevationProfile instance
            
        Raises:
            ValueError: If profile_type is not recognized
        """
        if profile_type not in cls._profiles:
            raise ValueError(f"Unknown profile type: {profile_type}. Available types: {list(cls._profiles.keys())}")
        
        return cls._profiles[profile_type](parameters)
    
    @classmethod
    def available_profiles(cls) -> list:
        """Get list of available profile types."""
        return list(cls._profiles.keys())
