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
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def fit_model(
            self,
            points:NDArray
        ):
        assert len(points) == self.num_samples

        x1,y1,z1 = points[0][0],points[0][1],points[0][2]
        x2,y2,z2 = points[1][0],points[1][1],points[1][2]
        x3,y3,z3 = points[2][0],points[2][1],points[2][2]

        self.a = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
        self.b = (x3-x1)*(z2-z1)-(x2-x1)*(z3-z1)
        self.c = (y3-y1)*(x2-x1)-(y2-y1)*(x3-x1)
        norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        if norm == 0:
            return False
        self.a /= norm
        self.b /= norm
        self.c /= norm
        self.d = -self.a*x1 - self.b*y1 -self.c*z1
        return True
        
    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        norm = np.sqrt(a**2 + b**2 + c**2)

        distances = np.array([
            x, y, z, np.ones_like(x)
        ]).T

        multiplier = np.array([[a,b,c,d]]).T
        distances = distances @ multiplier / norm

        return distances

    def calc_distance_one_point(self, point: NDArray):
        return self.calc_distances(np.array([point]))[0]
