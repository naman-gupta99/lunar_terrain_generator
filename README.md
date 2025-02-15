# Moon Rover Project

A Python-based project for moon rover data analysis and visualization.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
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

## License

[Add your license information here]
