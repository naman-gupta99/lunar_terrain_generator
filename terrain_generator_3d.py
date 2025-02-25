import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

class TerrainGenerator3D:
    def __init__(self, elevation_data, size_km):
        self.elevation = elevation_data
        self.size_km = size_km
        self.image_size = elevation_data.shape[0]
        self.path_points = None
        
    def generate_random_path(self):
        # Generate 2 boundary points
        while True:
            boundary_points = []
            sides = np.random.choice([0, 1, 2, 3], size=2, replace=False)
            for side in sides:
                if side == 0:  # Top
                    boundary_points.append([np.random.uniform(0, self.size_km), 0])
                elif side == 1:  # Right
                    boundary_points.append([self.size_km, np.random.uniform(0, self.size_km)])
                elif side == 2:  # Bottom
                    boundary_points.append([np.random.uniform(0, self.size_km), self.size_km])
                else:  # Left
                    boundary_points.append([0, np.random.uniform(0, self.size_km)])
            
            # Calculate distance between points
            p1, p2 = np.array(boundary_points)
            distance = np.sqrt(np.sum((p2 - p1) ** 2))
            
            # Only accept points if they are at least 2km apart
            if distance >= 2.0:
                break
        
        # Create path with the valid boundary points
        self.path_points = np.array(boundary_points)
        return self.path_points
    
    def modify_elevation_for_path(self, path_width_km=0.10):
        if self.path_points is None:
            self.generate_random_path()
            
        # Create grid points
        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        # For each segment in the path
        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]
            
            # Calculate distances from each grid point to the line segment
            distances = self._point_to_line_distance(grid_points, start, end)
            mask = distances <= path_width_km
            
            # Modify elevation where distance is within threshold
            flat_elevation = self.elevation.ravel()
            flat_elevation[mask] = 0
            self.elevation = flat_elevation.reshape(self.image_size, self.image_size)
    
    def _point_to_line_distance(self, points, start, end):
        # Calculate distances from points to line segment
        line_vec = end - start
        point_vec = points - start
        line_length = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_length
        proj_length = np.dot(point_vec, line_unit_vec)
        
        # Handle points beyond segment ends
        proj_length = np.clip(proj_length, 0, line_length)
        projection = start + np.outer(proj_length, line_unit_vec)
        
        # Calculate distances
        distances = np.sqrt(np.sum((points - projection) ** 2, axis=1))
        return distances

    def generate_3d_terrain(self, filename=None, show_path=True):
        num_contours=6
        
        # Generate and apply path if not already done
        if self.path_points is None and show_path:
            self.generate_random_path()
            self.modify_elevation_for_path()

        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        
        elevation_km = np.nan_to_num(self.elevation / 1000, nan=0.0, posinf=0.0, neginf=0.0)

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

        # Get min and max Z values for contours
        z_min = np.min(ZI)
        z_max = np.max(ZI)
        
        # Calculate contour spacing to ensure 0 is included
        if z_min < 0 and z_max > 0:
            # Ensure symmetrical spacing around 0
            max_abs = max(abs(z_min), abs(z_max))
            z_min = -max_abs
            z_max = max_abs
        
        # Add surface plot
        fig.add_trace(go.Surface(
            z=-ZI.T,
            x=xi,
            y=yi,
            colorscale=[
                [0, 'rgb(0,0,0)'],
                [1, 'rgb(225,225,225)']
            ],
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    project_z=False,
                    start=z_min,
                    end=z_max,
                    size=(z_max - z_min) / num_contours,
                    width=2,     
                    color='rgb(100, 0, 0)'
                )
            ),
            hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Elevation: %{z:.1f} m<extra></extra>',
        ))

        # Add path visualization only if show_path is True
        if show_path and self.path_points is not None:
            path_z = interp_func(self.path_points)
            
            fig.add_trace(go.Scatter3d(
                x=self.path_points[:, 0],
                y=self.path_points[:, 1],
                z=path_z,
                mode='lines+markers',
                line=dict(
                    color='white', 
                    width=5,
                    dash='dash'  # Add this line to make the path dashed
                ),
                name='Path'
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
                    eye=dict(x=0, y=0, z=1),
                    up=dict(x=0, y=1, z=0)
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

    def generate_cross_section(self, y_km=1.5, filename=None):
        """Generate a 2D cross-section view of the terrain at specified y coordinate."""
        # Create interpolation function for the elevation data
        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        elevation_km = np.nan_to_num(self.elevation / 1000, nan=0.0, posinf=0.0, neginf=0.0)
        interp_func = RegularGridInterpolator(
            (x, y), 
            elevation_km, 
            bounds_error=False, 
            fill_value=0.0
        )
        
        # Generate points along the cross-section
        x_points = np.linspace(0, self.size_km, self.image_size)
        cross_section_points = np.column_stack((x_points, np.full_like(x_points, y_km)))
        z_points = interp_func(cross_section_points)  # Convert back to meters
        
        # Create the cross-section plot
        fig = go.Figure()
        mask_top = z_points > 0
        mask_bottom = z_points < 0
        # Add fill above x-axis (red)
        fig.add_trace(go.Scatter(
            x=x_points[mask_top],
            y=-z_points[mask_top],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            name='Above Ground',
            hoverinfo='skip'
        ))
        
        # Add fill below x-axis (green)
        fig.add_trace(go.Scatter(
            x=x_points[mask_bottom],
            y=-z_points[mask_bottom],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.3)',
            name='Below Ground'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_points,
            y=-z_points,
            mode='lines',
            line=dict(color='black', width=2),
            name='Below Ground'
        ))

        fig.update_layout(
            title=f'Terrain Cross-section at y = {y_km} km',
            xaxis_title='Distance (km)',
            yaxis_title='Elevation (m)',
            width=1800,
            height=1000,
            template='plotly_white',
            showlegend=True,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=0.001,  # Since x is in km and y in m
                autorange=True
            )
        )
        
        if filename:
            fig.write_html(filename)
        return fig
    
    def generate_terrain_views(self, filename_with_path=None, filename_without_path=None, cross_section_path=None):
        """Generate two views of the terrain: one with path and one without."""
        fig_without_path = self.generate_3d_terrain(filename_without_path, show_path=False)
        fig_with_path = self.generate_3d_terrain(filename_with_path, show_path=True)
        fig_cross_section = self.generate_cross_section(y_km=1.5, filename=cross_section_path)
        return fig_with_path, fig_without_path, fig_cross_section