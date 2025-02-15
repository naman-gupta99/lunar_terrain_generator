import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
            radius_px = self._meters_to_pixels(crater['radius_m'])
            depth = crater['depth_m']

            # Skip if radius is zero
            if radius_px <= 0:
                continue

            y_coords, x_coords = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            dist_from_center = np.sqrt(x_coords**2 + y_coords**2)
            
            crater_mask = dist_from_center <= radius_px
            crater_depth = np.zeros_like(dist_from_center)
            
            # Calculate normalized distances safely
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_dist = np.where(radius_px > 0, 
                                         dist_from_center[crater_mask]/radius_px, 
                                         np.zeros_like(dist_from_center[crater_mask]))
                crater_depth[crater_mask] = depth * np.clip(1 - normalized_dist**2, 0, 1)
            
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

        fig = go.Figure(data=go.Heatmap(
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
