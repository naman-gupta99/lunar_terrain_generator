import argparse
from crater_generator import CraterGenerator
from map_generator_2d import MapGenerator2D
from elevation_map_generator import ElevationMapGenerator
from terrain_generator_3d import TerrainGenerator3D

def parse_args():
    parser = argparse.ArgumentParser(description='Moon terrain generator')
    parser.add_argument('-c', '--craters', type=str, help='Input craters CSV file')
    parser.add_argument('-m', '--map', action='store_true', help='Generate 2D map')
    parser.add_argument('-e', '--elevation', action='store_true', help='Generate elevation map')
    parser.add_argument('-t', '--terrain', action='store_true', help='Generate 3D terrain')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parameters
    size_km = 2
    pixels_per_km = 500
    craters_file = 'craters.csv'
    
    # Generate or use existing craters file
    if not args.craters:
        generator = CraterGenerator(size_km=size_km, distribution_variance=0.2)
        generator.save_craters(craters_file)
    else:
        craters_file = args.craters

    # If no flags are provided, generate everything
    generate_all = not (args.map or args.elevation or args.terrain)
    
    # Generate 2D map
    if args.map or generate_all:
        map_2d = MapGenerator2D(size_km, pixels_per_km)
        map_2d.generate_map(craters_file)
        map_2d.save('crater_map_2d.png')
    
    # Generate elevation map
    elevation_data = None
    if args.elevation or args.terrain or generate_all:
        elevation_map = ElevationMapGenerator(size_km, pixels_per_km)
        elevation_map.generate_elevation_map(craters_file)
        elevation_data = elevation_map.elevation
        if args.elevation or generate_all:
            elevation_map.plot_heatmap('elevation_heatmap.html')
    
    # Generate 3D terrain
    if args.terrain or generate_all:
        if elevation_data is None:
            elevation_map = ElevationMapGenerator(size_km, pixels_per_km)
            elevation_map.generate_elevation_map(craters_file)
            elevation_data = elevation_map.elevation
        terrain_3d = TerrainGenerator3D(elevation_data, size_km)
        terrain_3d.generate_3d_terrain('terrain_3d.html')

if __name__ == "__main__":
    main()
