# starkit-ransac

A Python library for fitting geometric models to noisy point clouds using the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) (Random Sample Consensus) algorithm.

## Supported Models

| Model | Dimensions | Min. samples |
|-------|-----------|--------------|
| Point | 3D | 1 |
| Line | 3D | 2 |
| Plane | 3D | 3 |
| Circle | 2D, 3D | 3 |
| Sphere | 3D | 4 |
| Ellipse | 2D | 5 |
| Ellipsoid | 3D | 9 |
| Mobius Strip | 3D | 4 |
| Staircase | 3D | 90 |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.12+. Core dependency: `numpy >= 1.23.5, < 2`. Optional: `open3d` for visualization.

## Quick Start

```python
import numpy as np
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.generators.sphere import generate_sphere
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.visualisation.visualize import generate_mesh, draw_pretty
import open3d as o3d

# 1. Generate a noisy point cloud around a sphere
true_model = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=5.0)
points = generate_sphere(true_model, noise_sigma=0.1, n_points=1000)

# 2. Fit with RANSAC
ransac = RANSAC(points)
fitted = ransac.fit(Sphere, iter_num=1000, distance_threshold=0.05)

print(fitted.model)  # {'center': array([...]), 'radius': ...}

# 3. Visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
mesh = generate_mesh(fitted, color=[0, 1, 0])
draw_pretty([mesh, pcd])
```

More examples for every supported model are in the [`examples/`](examples/) directory.

## Library Structure

```
starkit_ransac/
├── ransac_3d.py            # RANSAC algorithm
├── abstract_surface.py     # Base class for all models
├── surfaces/               # Geometric model classes
├── generators/             # Noisy point cloud generators
└── visualisation/          # Open3D mesh generation and drawing
```

**`RANSAC`** is the main entry point. Call `.fit(ModelClass, iter_num, distance_threshold)` to run the algorithm. Each model class inherits from `AbstractSurfaceModel` and implements:

- `fit_model(points)` — fit the model to a minimal sample
- `calc_distances(points)` — compute distances from all points to the surface
- `num_samples` — number of points needed to define the model

## Adding a Custom Model

1. Create a class inheriting from `AbstractSurfaceModel` in `starkit_ransac/surfaces/`.
2. Implement `fit_model`, `calc_distances`, `calc_distance_one_point`, and the `num_samples` / `model` properties.
3. (Optional) Add a generator in `generators/` and a mesh generator in `visualisation/`.

See [`starkit_ransac/surfaces/point.py`](starkit_ransac/surfaces/point.py) for a minimal example.

## Testing

```bash
python -m pytest test/
```

## Contributing

1. Clone the repo
2. Create a branch: `feature/<your_shape_name>`
3. Write tests (see `test/test_point.py` for reference) and make sure they pass
4. Open a pull request

Code style: `PascalCase` for classes, `snake_case` for functions and variables.

## License

MIT
