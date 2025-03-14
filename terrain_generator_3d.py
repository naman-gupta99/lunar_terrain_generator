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
        self.max_z = 3
        self.min_z = -4
        
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
    
    def generate_offset_paths(self, path_width_km=0.10):
        """
        Generate two offset paths on either side of the main path.
        
        Args:
            path_width_km (float): Width of the path in kilometers.
        
        Returns:
            left_path (np.ndarray): Left-side path points.
            right_path (np.ndarray): Right-side path points.
        """
        offset_distance = path_width_km
        left_path = []
        right_path = []

        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]

            # Compute direction vector
            direction = end - start
            direction /= np.linalg.norm(direction)  # Normalize
            
            # Compute perpendicular vector (rotate 90 degrees)
            perpendicular = np.array([-direction[1], direction[0]])  

            # Compute left and right offset points
            left_start = start + perpendicular * offset_distance
            right_start = start - perpendicular * offset_distance
            left_end = end + perpendicular * offset_distance
            right_end = end - perpendicular * offset_distance

            left_path.append(left_start)
            left_path.append(left_end)
            right_path.append(right_start)
            right_path.append(right_end)

        return np.array(left_path), np.array(right_path)
    
    def modify_elevation_for_path(self, path_width_km=0.10, inverse=False):
        """
        Modify elevation data for path or inverse path.
        
        Args:
            path_width_km (float): Width of the path in kilometers
            inverse (bool): If True, flatten everything except the path
        """
            
        # Create grid points
        x = np.linspace(0, self.size_km, self.image_size)
        y = np.linspace(0, self.size_km, self.image_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Calculate combined mask for all path segments
        combined_mask = np.zeros(len(grid_points), dtype=bool)
        
        # For each segment in the path
        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]
            
            # Calculate distances from each grid point to the line segment
            distances = self._point_to_line_distance(grid_points, start, end)
            segment_mask = distances <= path_width_km
            combined_mask = combined_mask | segment_mask
        
        # Flatten elevation data
        flat_elevation = self.elevation.ravel()
        
        if inverse:
            print("Inverse path")
            # Set elevation to 0 for all points except the path, keep the path unchanged
            new_elevation = np.zeros_like(flat_elevation)
            new_elevation[combined_mask] = flat_elevation[combined_mask]  # Preserve path elevation
            flat_elevation = new_elevation
        else:
            print("Normal path")
            # Flatten the path
            flat_elevation[combined_mask] = 0  # Set elevation to 0 only for path
    
        return flat_elevation.reshape(self.image_size, self.image_size)
    
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

    def generate_3d_terrain(self, filename=None, show_path=True, inverse_path=False):
        num_contours = 7
        elevation = self.elevation
        
        if show_path:
            elevation = self.modify_elevation_for_path(inverse=inverse_path)

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

        # Surface Plot
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
                    start=self.min_z,
                    end=self.max_z,
                    size=(self.max_z - self.min_z) / num_contours,
                    width=2,     
                    color='rgb(100, 0, 0)'
                )
            ),
            hovertemplate='X: %{x:.1f} km<br>Y: %{y:.1f} km<br>Elevation: %{z:.1f} m<extra></extra>',
        ))

        # Add path visualization
        if show_path and self.path_points is not None:
            left_path, right_path = self.generate_offset_paths()
            
            left_z = interp_func(left_path)
            right_z = interp_func(right_path)

            # Left Side White Line
            fig.add_trace(go.Scatter3d(
                x=left_path[:, 0],
                y=left_path[:, 1],
                z=left_z,
                mode='lines',
                line=dict(color='white', width=5, dash='dash'),
                name=''
            ))

            # Right Side White Line
            fig.add_trace(go.Scatter3d(
                x=right_path[:, 0],
                y=right_path[:, 1],
                z=right_z,
                mode='lines',
                line=dict(color='white', width=5, dash='dash'),
                name=''
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
  
    def generate_terrain_views(self, filename_with_path=None, filename_without_path=None, filename_with_inverse_path=None):
        self.generate_random_path()
        """Generate two views of the terrain: one with path and one without."""
        fig_without_path = self.generate_3d_terrain(filename_without_path, show_path=False)
        fig_with_inverse_path = self.generate_3d_terrain(filename_with_inverse_path, show_path=True, inverse_path=True)
        fig_with_path = self.generate_3d_terrain(filename_with_path, show_path=True, inverse_path=False)