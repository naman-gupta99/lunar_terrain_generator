# Moon Rover Project

A Python-based project for moon rover data analysis and visualization.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:naman-gupta99/lunar_terrain_generator.git
cd moon_rover
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following main packages:
- narwhals (1.26.0)
- numpy (2.2.2)
- pandas (2.2.3)
- pillow (11.1.0)
- plotly (6.0.0)
- scipy (1.15.1)

For a complete list of dependencies, see `requirements.txt`.

## Usage

Run the main script with various options to generate moon terrain data:

```bash
python main.py [options]
```

Options:
- `-c, --craters <file>`: Use existing craters CSV file instead of generating new one
- `-m, --map`: Generate 2D crater map (outputs crater_map_2d.png)
- `-e, --elevation`: Generate elevation heatmap (outputs elevation_heatmap.html)
- `-t, --terrain`: Generate 3D terrain visualization (outputs terrain_3d.html)
- `-cf, --cut_fill`: Generate cut and fill map (outputs cut_fill.html)

If no options are provided, the script will generate all outputs using a new random crater distribution.

Examples:
```bash
# Generate all visualizations with new random craters
python main.py

# Use existing craters file and generate only 2D map
python main.py -c existing_craters.csv -m

# Generate elevation heatmap and 3D terrain
python main.py -e -t
```

The generated files will be saved in the current directory:
- `craters.csv`: Crater distribution data
- `crater_map_2d.png`: 2D visualization of crater positions
- `elevation_heatmap.html`: Interactive elevation heatmap
- `terrain_3d.html`: Interactive 3D terrain visualization
- `terrain_3d_path.html`: 3D terrain with path visualization
- `terrain_3d_inverse_path.html`: 3D terrain with inverse path visualization
- `cut_fill.html`: Cut and fill analysis visualization (when enabled)

## Terrain Features
The generated terrain includes:

- Realistic crater distributions following lunar surface statistics
- Variable crater sizes and depths
- Proper crater rim and ejecta formation
- Optional path visualization
- Interactive 3D terrain visualization with contour lines

## Configuration
The terrain is generated with the following default parameters:

- Size: 2 km x 2 km
- Resolution: 500 pixels per km
- Distribution variance: 0.2 (controls randomness in crater distribution)
- These parameters can be modified in the main.py file.

## License

MIT License
