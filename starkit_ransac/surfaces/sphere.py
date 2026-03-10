import numpy as np
from numpy.typing import ArrayLike, NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel

class Sphere(AbstractSurfaceModel):
    def __init__(
            self,
            center:ArrayLike|None=None,
            radius:float|None=None
        ) -> None:
        if center is not None:
            center = np.array(center)
        self.center:NDArray = center
        self.radius:float = radius
        self.num_samples = 4

    def fit_model(
            self, 
            points: ArrayLike
        ):
        points = np.array(points)
        # sphere can be described as:
        # (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
        # let's extract a,b,c and r from it:
        # x^2 - 2ax - a^2 + y^2 - 2by + b^2 + z^2 - 2cz + c^2 = r^2
        # x^2 + y^2 + z^2 = 2ax + 2by + 2cz + (r^2 - a^2 - b^2 - c^2)
        # let d = (r^2 - a^2 - b^2 - c^2)
        # then:
        # x^2 + y^2 + z^2 = 2ax + 2by + 2cz + d
        # we have to solve for a,b,c,d
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        rhs = x**2 + y**2 + z**2
        A = np.array([
            2*x, 2*y, 2*z, np.ones_like(x)
        ]).T
        abcd = np.linalg.solve(A, rhs)
        a,b,c,d = abcd
        r = np.sqrt(d + a**2 + b**2 + c**2)

        self.center = np.array([a, b, c])
        self.radius = r
        return True

    def calc_distances(self, points: ArrayLike) -> NDArray:
        return np.abs(np.linalg.norm(np.array(points) - self.center, axis=-1) - self.radius)

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances([point])[0]
