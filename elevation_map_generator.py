import numpy as np
import pandas as pd
import plotly.graph_objects as go
from elevation_profile import ProfileFactory

class ElevationMapGenerator:
    def __init__(self, size_km, pixels_per_km):
        self.size_km = size_km
        self.pixels_per_km = pixels_per_km
        self.image_size = int(size_km * pixels_per_km)
        self.elevation = np.zeros((self.image_size, self.image_size))

    def _meters_to_pixels(self, meters):
        km = meters / 1000
        return int(km * self.pixels_per_km)

    def generate_elevation_map(self, crater_csv):
        df = pd.read_csv(crater_csv)
        
        for _, crater in df.iterrows():
            center_x = int(crater['x_km'] * self.pixels_per_km)
            center_y = int(crater['y_km'] * self.pixels_per_km)
            crater_radius_px = self._meters_to_pixels(crater['radius_m'])

            # Skip if radius is zero
            if crater_radius_px <= 0:
                continue
            
            radius_px = int(crater_radius_px * 2.5)
            
            # Create crater profile
            profile = ProfileFactory.create('small', {'D': crater['radius_m'] * 2, 'depth': crater['depth_m']})  # D is diameter in meters

            # Generate elevation grid
            y_coords, x_coords = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            dist_from_center = np.sqrt(x_coords**2 + y_coords**2)
            
            # Convert pixel distances to meters
            dist_meters = dist_from_center * (1000 / self.pixels_per_km)  # Convert to meters
            
            # Calculate crater depth using profile
            crater_depth = np.zeros_like(dist_from_center)
            valid_points = dist_meters <= (2.5 * crater['radius_m'])  # Profile valid up to 2.5R
            crater_depth[valid_points] = np.vectorize(profile.get_height)(dist_meters[valid_points])
            
            # Apply crater to elevation map
            y_min = max(0, center_y-radius_px)
            y_max = min(self.image_size, center_y+radius_px+1)
            x_min = max(0, center_x-radius_px)
            x_max = min(self.image_size, center_x+radius_px+1)
            
            mask_y_min = max(0, -(center_y-radius_px))
            mask_y_max = crater_depth.shape[0] - max(0, (center_y+radius_px+1) - self.image_size)
            mask_x_min = max(0, -(center_x-radius_px))
            mask_x_max = crater_depth.shape[1] - max(0, (center_x+radius_px+1) - self.image_size)
            
            self.elevation[y_min:y_max, x_min:x_max] -= crater_depth[mask_y_min:mask_y_max, mask_x_min:mask_x_max]

    def plot_heatmap(self, filename=None):
        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)

        fig = go.Figure()
        
        # Add heatmap layer
        fig.add_trace(go.Heatmap(
            z=self.elevation,
            x=x,
            y=y,
            zsmooth='best',
            hoverongaps=False,
            colorscale=[
                [0, 'rgb(0,0,0)'],
                [1, 'rgb(225,225,225)']
            ],
            hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Elevation: %{z:.1f} m<extra></extra>'
        ))
        
        # Add contour layer at zero elevation
        fig.add_trace(go.Contour(
            z=self.elevation,
            x=x,
            y=y,
            contours=dict(
                start=0,
                end=0,
                coloring='lines',
                showlabels=True
            ),
            line=dict(color='red', width=2),
            showscale=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            template='plotly',  # Changed from plotly_dark to plotly
            xaxis_title='Distance (km)',
            yaxis_title='Distance (km)',
            xaxis=dict(
                scaleanchor='y',
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor='x',
                scaleratio=1,
            ),
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
            coloraxis_colorbar=dict(
                title=dict(
                    text='Elevation (meters)',
                    side='right'
                ),
                ticks='outside',
                tickformat='.0f'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        if filename:
            fig.write_html(filename)
        return fig
