# MyLib

A comprehensive Python library for computer vision, geometry, and photogrammetry tasks. MyLib provides utilities for 3D reconstruction, camera calibration, feature matching, dataset processing, and visualization tools for computer vision research and applications.

## Features

- **Geometry & Projections**: Essential geometric transformations, rotation matrices, and camera projection utilities
- **Computer Vision**: Image processing, feature matching, and camera visualization tools
- **Coordinate Conversions**: Homogeneous/Euclidean conversions and coordinate system transformations
- **I/O Operations**: File input/output utilities for various computer vision data formats
- **Metrics & Evaluation**: Performance evaluation tools for reconstruction and matching tasks
- **Visualization**: Comprehensive plotting utilities for cameras, 3D points, matches, and geometric data

## Installation

### From source (development)
```bash
git clone <your-repo-url>
cd mylib
pip install -e .
```

### Dependencies
- Python >= 3.8
- NumPy
- SciPy
- Matplotlib
- Pandas
- OpenCV
- Kornia (for vision geometry operations)

All dependencies are listed in `pyproject.toml` and will be installed automatically.

## Quick Start

```python
import mylib

# Geometry operations
from mylib import geometry, conversions
R = geometry.rotation_matrix_from_euler(0.1, 0.2, 0.3)
points_3d = conversions.homogeneous_to_euclidean(points_h)

# Camera visualization
from mylib import plot  # camera_visualization is included in plot or as a separate module if present
# plot.plot_cameras(cameras, ax=ax)  # Uncomment if camera_visualization is available
plot.plot_matches(image1, image2, matches)

# Projections and transformations
from mylib import projections
K = projections.intrinsic_matrix(fx, fy, cx, cy)

# Metrics and evaluation
from mylib import metrics
error = metrics.reprojection_error(points_3d, points_2d, camera_matrix)

# File I/O operations
from mylib import io
data = io.load_camera_data(filepath)
```

## Module Overview

### Core Modules

- **`geometry`**: 
  - Rotation matrices and transformations
  - Euler angle conversions
  - Triangulation algorithms
  - Pose estimation utilities

- **`conversions`**: 
  - Homogeneous ↔ Euclidean coordinate conversions
  - Coordinate system transformations
  - Point format conversions

- **`projections`**: 
  - Camera projection models
  - Intrinsic and extrinsic parameter handling
  - Perspective projection utilities

- **`io`**: 
  - File input/output operations
  - Camera parameter loading/saving
  - Data format utilities

- **`matching`**: 
  - Feature matching algorithms
  - Correspondence utilities
  - Match filtering and validation

- **`metrics`**: 
  - Reconstruction quality evaluation
  - Reprojection error computation
  - Pose estimation accuracy metrics

- **`plot`**: 
  - General 2D/3D visualization tools
  - Match visualization
  - Point cloud plotting

## Project Structure
```
mylib/
├── src/
│   └── mylib/
│       ├── __init__.py                 # Main package imports
│       ├── conversions.py              # Coordinate conversions
│       ├── geometry.py                 # Geometric operations
│       ├── io.py                       # File I/O operations
│       ├── matching.py                 # Feature matching
│       ├── metrics.py                  # Evaluation metrics
│       ├── plot.py                     # General plotting tools
│       ├── projections.py              # Camera projections
│       └── test.py                     # Test utilities
├── pyproject.toml                      # Package configuration
├── setup.py                            # Setup script
└── README.md                           # This file
```

## Example Usage

### Basic Geometry Operations
```python
from mylib import geometry, conversions
import numpy as np

# Create rotation matrix from Euler angles
R = geometry.rotation_matrix_from_euler(0.1, 0.2, 0.3)

# Convert between coordinate representations
points_homogeneous = np.array([[1, 2, 3, 1], [4, 5, 6, 1]]).T
points_euclidean = conversions.homogeneous_to_euclidean(points_homogeneous)
```


### Feature Matching and Visualization
```python
from mylib import matching, plot

# Perform feature matching
matches = matching.match_features(features1, features2)

# Visualize matches
plot.plot_matches(image1, image2, keypoints1, keypoints2, matches)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

See `pyproject.toml` for the complete dependency list. Main requirements:
- numpy
- scipy  
- matplotlib
- pandas
- opencv-python
- kornia

Additional dependencies may be required.

##
This readme was ~~proudly~~ quickly made with Copilot. It might contain errors. 