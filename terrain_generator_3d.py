import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

class TerrainGenerator3D:
    def __init__(self, elevation_data, size_km):
        self.elevation = elevation_data
        self.size_km = size_km
        self.image_size = elevation_data.shape[0]

    def generate_3d_terrain(self, filename=None):
        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        
        # Handle any NaN or inf values in elevation data
        elevation_km = np.nan_to_num(self.elevation / 1000, nan=0.0, posinf=0.0, neginf=0.0)

        # Create the bilinear interpolation function with bounds_error=False to handle edge cases
        interp_func = RegularGridInterpolator(
            (x, y), 
            elevation_km, 
            bounds_error=False, 
            fill_value=0.0
        )

        # Generate grid points
        xi, yi = np.linspace(0, self.size_km, self.image_size), np.linspace(0, self.size_km, self.image_size)
        XI, YI = np.meshgrid(xi, yi)
        
        # Reshape points for interpolation and handle any potential numerical issues
        points = np.vstack((XI.ravel(), YI.ravel())).T
        with np.errstate(invalid='ignore', divide='ignore'):
            ZI = interp_func(points).reshape(XI.shape)

        fig = go.Figure(data=[go.Surface(
            z=ZI, 
            x=xi, 
            y=yi,
            colorscale=[
                [0, 'rgb(0,0,0)'],
                [1, 'rgb(225,225,225)']
            ],
            hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Elevation: %{z:.3f} km<extra></extra>'
        )])
        
        fig.update_layout(
            template='plotly',
            scene=dict(
                xaxis=dict(title='Distance (km)', range=[self.size_km, 0]),
                yaxis=dict(title='Distance (km)', range=[self.size_km, 0]),
                zaxis=dict(
                    title='Elevation (km)',
                    range=[-self.size_km//2, self.size_km//2]
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=1000,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='white'
        )

        if filename:
            fig.write_html(filename)
        return fig
