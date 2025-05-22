# Geodesic Toolbox
This repository contains a collection of tools for geodesic computations, including the calculation of geodesic trajectories and distances on various manifolds. It also supports sampling form manifolds.
Randers metrics are also experimentally supported.
The main assumption is that the manifold admits a global identity chart and thus that the manifold on only as $(\mathbb{R}^d, g)$, where $g$ is a Riemannian metric.

## Installation
To install the package, clone the repository and run the following command in the root directory:
```bash
pip install -e ./
```
It requires the following packages:
- torch
- numpy
- scipy
- scikit-learn
- tqdm

## Usage
The package can be used to compute geodesic trajectories and distances on various manifolds. Examples are provided in the `examples` directory.
