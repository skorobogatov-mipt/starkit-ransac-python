import pdb
import numpy as np
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
from copy import deepcopy

class Plane3D(AbstractSurfaceModel):
    def __init__(
            self,
            a:float=np.nan,
            b:float=np.nan,
            c:float=np.nan,
            d:float=np.nan
        ) -> None:
        self.num_samples = 3
        self.coeffs = np.array([a,b,c,d])

    @property
    def a(self):
        return self.coeffs[0]

    @a.setter
    def a(self, a):
        self.coeffs[0] = a

    @property
    def b(self):
        return self.coeffs[1]

    @b.setter
    def b(self, b):
        self.coeffs[1] = b

    @property
    def c(self):
        return self.coeffs[2]

    @c.setter
    def c(self, c):
        self.coeffs[2] = c

    @property
    def d(self):
        return self.coeffs[3]

    @d.setter
    def d(self, d):
        self.coeffs[3] = d

    def fit_model(
            self,
            points:NDArray
        ):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        self.coeffs[:3] = normal
        self.d = -np.sum(np.multiply(normal, points[0, :]))
        return True
        
    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:

        norm = np.linalg.norm(self.coeffs[:3])
        return np.abs((points @ self.coeffs[:3] + self.d)) / norm

    def calc_distance_one_point(self, point: NDArray):
        return self.calc_distances(np.array([point]))[0]
