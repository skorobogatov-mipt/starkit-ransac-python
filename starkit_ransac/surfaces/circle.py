from numpy.linalg import norm
from numpy.typing import ArrayLike, NDArray
from typing import Iterable
import numpy as np
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from starkit_ransac.utils import line_from_2_points, lines_intersection, midpoint, normal_to_2d_line, rotate_from_axis_to_axis, rotate_rodrigues
from starkit_ransac.surfaces.circle2d import Circle2D

class Circle3D(AbstractSurfaceModel):
    def __init__(
            self,
            center:ArrayLike|None=None,
            radius:float|None=None,
            normal:ArrayLike|None=None
        ) -> None:
        if center is None:
            center = np.full(3, np.nan)
        if radius is None:
            radius = np.nan

        if normal is None:
            normal = np.full(3, np.nan)

        self.center = center
        self.radius = radius
        self.normal = normal
        self.num_samples = 3
        self.__circle2d = Circle2D(center, radius)

    @property
    def center(self) -> NDArray[np.float64]:
        return self._center

    @center.setter
    def center(self, center:ArrayLike):
        self._center = np.copy(center)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, radius:float):
        self._radius = float(radius)

    @property
    def normal(self) -> NDArray[np.float64]:
        return self._normal

    @normal.setter
    def normal(self, normal:ArrayLike):
        if np.asarray(normal).flatten().shape != (3,):
            raise ValueError("Normal must be a vector of 3 elements")
        self._normal = np.asarray(normal) / np.linalg.norm(normal)

    def fit_model(self, points: NDArray):
        assert len(points) == self.num_samples

        # 0) get two vectors on the circle's plane
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # 1) get normal to the circle
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        self.normal = normal

        # 2) rotate points so that they are flat in 2d
        rotated_points = rotate_from_axis_to_axis(
            points,
            normal,
            [0, 0, 1]
        )
        
        # 3) fit points as a 2d circle
        self.__circle2d.fit_model(rotated_points[..., :2])
        self.radius = self.__circle2d.radius

        # 4) "unrotate" the center
        rotated_center = self.__circle2d.center
        rotated_center = np.append(rotated_center, rotated_points[0, 2])
        center = rotate_from_axis_to_axis(
            [rotated_center],
            [0, 0, 1],
            normal
        )[0]

        self.center = center
        return True

    def calc_distances(self, points: NDArray) -> NDArray:
        # 1) get the circle's plane
        # plane equation from 
        # normal = [A,B,C] and point=[x_p,y_p,z_p] is:
        # A*x + B*y + C*z - (A*xp + B*y_p + C*z_p) = 0
        normal = self.normal
        center = self.center
        d = -np.sum(normal*center)

        # 2) get distance to plane
        # distance from point=[x,y,z] to 
        # plane described above is:
        # (Ax + By + Cz + d)/sqrt(A^2 + B^2 + C^2)
        plane_distances = (np.sum(points*normal, axis=-1) + d)
        plane_distances = plane_distances / np.linalg.norm(normal)
        
        # 3) get distance to a cylinder instead of circle
        radius = self.radius
        cyl_distances = np.cross(normal, points-center)
        cyl_distances = np.linalg.norm(cyl_distances, axis=-1) - radius

        # 4) get actual distance
        return np.sqrt(np.square(cyl_distances) + np.square(plane_distances))

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances(np.array([point]))[0]

    def __repr__(self):
        res = ''
        res += 'radius: ' + str(self.radius) + '\n'
        res += 'center: ' + str(self.center) + '\n'
        res += 'normal: ' + str(self.normal) + '\n'
        return res
