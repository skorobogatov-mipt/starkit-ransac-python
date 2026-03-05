import numpy as np
from typing import Iterable
from numpy.typing import NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from starkit_ransac.utils import line_from_2_points, midpoint, normal_to_2d_line, lines_intersection


class Circle2D(AbstractSurfaceModel):
    def __init__(
            self,
            center:Iterable=[np.nan, np.nan],
            radius:float=np.nan
        ) -> None:
        super().__init__()
        self.model = {
            'center' : center,
            'radius' : radius
        }
        self.num_samples = 3

    def fit_model(self, points: NDArray):
        # 1) get a central normal to the line between any two points
        mid1 = midpoint(points[0], points[1])
        k1, b1 = line_from_2_points(points[0], points[1])
        k1, b1 = normal_to_2d_line(k1, b1, mid1)

        # 2) get a central normal to the line between other two points
        mid2 = midpoint(points[0], points[2])
        k2, b2 = line_from_2_points(points[0], points[2])
        k2, b2 = normal_to_2d_line(k2, b2, mid2)

        # 3) get lines intersection, aka center
        center = lines_intersection(k1, b1, k2, b2)

        # 4) get radius
        radius = np.linalg.norm(center - points[0])
        
        self.model['center'] = center
        self.model['radius'] = radius

    def calc_distances(self, points: NDArray) -> NDArray:
        return np.linalg.norm(points - self.model['center'], axis=-1) - self.model['radius']

    def calc_distance_one_point(self, point: NDArray) -> float:
        return np.linalg.norm(point - self.model['center']) - self.model['radius']

