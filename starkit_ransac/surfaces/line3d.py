import numpy as np
from numpy.typing import ArrayLike, NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from starkit_ransac.utils import normalize

class Line3D(AbstractSurfaceModel):

    def __init__(
            self,
            direction:ArrayLike|None=None,
            point:ArrayLike|None=None
        ) -> None:
        if direction is not None:
            direction = np.array(direction)
            direction = normalize(direction)
        if point is not None:
            point = np.array(point)

        self.direction:ArrayLike = direction
        self.point:ArrayLike = point
        self.num_samples = 2

    def fit_model(self, points: NDArray):
        self.direction = normalize(points[1] - points[0])
        self.point = points[0] + np.dot(points[0], self.direction) * self.direction
        
    def calc_distances(self, points: NDArray) -> NDArray:
        a = self.point
        return (a - points) - np.outer(np.dot((a - points), self.direction), self.direction)

    def calc_distance_one_point(self, point):
        return self.calc_distance_one_point([point])[0]

