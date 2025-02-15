from PIL import Image, ImageDraw
import pandas as pd

class MapGenerator2D:
    def __init__(self, size_km, pixels_per_km):
        self.size_km = size_km
        self.pixels_per_km = pixels_per_km
        self.image_size = int(size_km * pixels_per_km)
        self.image = Image.new('RGB', (self.image_size, self.image_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def _meters_to_pixels(self, meters):
        km = meters / 1000
        return int(km * self.pixels_per_km)

    def generate_map(self, crater_csv):
        df = pd.read_csv(crater_csv)
        
        for _, crater in df.iterrows():
            x_px = int(crater['x_km'] * self.pixels_per_km)
            y_px = int(crater['y_km'] * self.pixels_per_km)
            radius_px = self._meters_to_pixels(crater['radius_m'])
            
            bbox = [
                x_px - radius_px,
                y_px - radius_px,
                x_px + radius_px,
                y_px + radius_px
            ]
            self.draw.ellipse(bbox, outline='black', fill='black')

    def save(self, filename):
        self.image.save(filename)
