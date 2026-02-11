import numpy as np
from ransac3d.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
from copy import deepcopy

class Point3D(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.k = 1
        self._model = {
            'x' : np.nan,
            'y' : np.nan,
            'z' : np.nan
        }

    def fit_model(
            self,
            points:NDArray
            ):
        assert len(points) == self.k
        self._model['x'] = points[0][0]
        self._model['y'] = points[0][1]
        self._model['z'] = points[0][2]

    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        center_point = np.array([
            self._model['x'], 
            self._model['y'], 
            self._model['z']
        ])

        diff_vectors = points - center_point
        distances = np.linalg.norm(
                diff_vectors, 
                axis=1
        )
        return distances

    def calc_distance_one_point(self, point: NDArray):
        pass

