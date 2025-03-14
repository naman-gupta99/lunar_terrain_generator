import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

class CutFillGenerator:
    def __init__(self, elevation_data, size_km):
        self.elevation = elevation_data
        self.size_km = size_km
        self.image_size = elevation_data.shape[0]
        self.max_z = 3
        self.min_z = -4

    def generate_3d_terrain(self, filename=None):
        num_contours = 6
        elevation = self.elevation

        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        
        elevation_km = np.nan_to_num(elevation / 1000, nan=0.0, posinf=0.0, neginf=0.0)

        interp_func = RegularGridInterpolator(
            (x, y), 
            elevation_km, 
            bounds_error=False, 
            fill_value=0.0
        )

        xi, yi = np.linspace(0, self.size_km, self.image_size), np.linspace(0, self.size_km, self.image_size)
        XI, YI = np.meshgrid(xi, yi)
        
        points = np.vstack((XI.ravel(), YI.ravel())).T
        ZI = interp_func(points).reshape(XI.shape)

        fig = go.Figure()

        # Calculate contour parameters
        z_min = np.min(-ZI.T)
        z_max = np.max(-ZI.T)
        contour_size = (z_max - z_min) / num_contours
        
        # Surface Plot with contours
        fig.add_trace(go.Surface(
            z=-ZI.T,
            x=xi,
            y=yi,
            colorscale=[
                [0, 'rgb(0,255,0)'],
                [0.74, 'rgb(200,255,200)'],
                [0.75, 'rgb(255,255,255)'],
                [0.76, 'rgb(255,200,200)'],
                [1, 'rgb(255,0,0)']
            ],
            cmax=13,
            cmin=-34,
            cmid=0,
            contours={
                "z": {
                    "show": True,
                    "usecolormap": False,
                    "highlightcolor": "black",
                    "project_z": True,
                    "width": 2,
                    "start": z_min,
                    "end": z_max,
                    "size": float(contour_size)  # Convert to Python float to avoid numpy.float64
                }
            }
        ))

        fig.update_layout(
            template='plotly',
            scene=dict(
                xaxis=dict(title='Distance (km)', range=[0, self.size_km]),
                yaxis=dict(title='Distance (km)', range=[0, self.size_km]),
                zaxis=dict(
                    title='Elevation (m)',
                    range=[(-self.size_km//2)*1000, (self.size_km//2)*1000]
                ),
                camera=dict(
                    eye=dict(x=0.5, y=0.5, z=0.5)
                )
            ),
            width=1800,
            height=1000,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='white'
        )

        if filename:
            fig.write_html(filename)
        return fig
  
    def generate_cut_fill(self, filename):
        """Generate two views of the terrain: one with path and one without."""
        self.generate_3d_terrain(filename)