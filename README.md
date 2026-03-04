# About this library
This is an open source implementation of most common objects for python.

# Requirements
This repo was written on python 3.12. It should work fine with some other
versions too, though.

To install requirements, run:
```
pip install -r requirements.txt
```

To install this module during development:
```
pip install -e .
```

# Modules
## Surfaces
This module represents all the surfaces included in this library.
In each file, there's a class that encapsulates one surface e.g. Cylinder,
Ellipsoid etc.

## Generators
This module contains all the generation functions for surfaces provided in
this library.
In each file, there's a function like: 
```
generate_cylinder(cylinder:Cylinder, noise_sigma, n_points)
```
that generates a set of 3d points for a given model.

## Visualization
This module contains mesh generators for each surface, like:
```
generate_cylinder_mesh(cylinder:Cylinder, color=...)
```
That creates an `open3d.LineSet` object to be visualized.

# Writing your modules
To write a custom surface detector, create a child class of SurfaceModel and
place it in `ransac3d/surfaces/`. An example can be found at
`ransac3d/surfaces/point.py`.

# Testing
To test your module, please write a pytest file for it. Example can be found in
`test/test_point.py`. 

To run your tests:
```
python -m pytest test/<your_test_file_name>
```

# Contributing
To add you custom surface to this repo:
1) Clone this repo
2) Create a brahcn called `feature/<your_shape_name>`
3) Write tests. Make sure that they pass.
4) Create a pull request.

## Codestyle
If you want to contribute to this library, make sure that your code adheres to the following code style:
* classes are written in `EachWordStartingWithACapitalLetter` style
* functions and variables are written in `snake_case`
