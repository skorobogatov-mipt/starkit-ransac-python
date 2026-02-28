import numpy as np
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
from copy import deepcopy

class Point3D(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = {
            'x' : np.nan,
            'y' : np.nan,
            'z' : np.nan
        }
        self.num_samples = 1

    def fit_model(
            self,
            points:NDArray
            ):
        assert len(points) == self.num_samples
        self.model['x'] = points[0][0]
        self.model['y'] = points[0][1]
        self.model['z'] = points[0][2]

    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        center_point = np.array([
            self.model['x'], 
            self.model['y'], 
            self.model['z']
        ])

        diff_vectors = points - center_point
        distances = np.linalg.norm(
                diff_vectors, 
                axis=1
        )
        return distances

    def calc_distance_one_point(self, point: NDArray) -> float:
        return 0.

