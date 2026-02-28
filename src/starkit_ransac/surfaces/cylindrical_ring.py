import numpy as np
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
import math


class CylindricalRing(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = {
            'center_x': np.nan,
            'center_y': np.nan,
            'center_z': np.nan,
            'inner_radius': np.nan,
            'outer_radius': np.nan,
            'height': np.nan,
            'central_radius': np.nan,
            'thickness': np.nan
        }
        # Ось всегда вертикальная [0,0,1]
        self.model['axis_x'] = 0.0
        self.model['axis_y'] = 0.0
        self.model['axis_z'] = 1.0
        self.num_samples = 2

    def fit_model(self, points: NDArray):
        assert len(points) >= self.num_samples

        p1, p2 = points[0], points[1]

        axis = np.array([0, 0, 1])

        #проекция на плоскость XY
        center = np.array([
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2,
            0.0
        ])

        radii = []
        for pt in [p1, p2]:
            dist = np.sqrt(pt[0] ** 2 + pt[1] ** 2)
            radii.append(dist)

        central_radius = np.mean(radii)

        height = abs(p1[2] - p2[2]) * 1.2
        if height < 0.1:
            height = 4.0  # значение по умолчанию

        # Толщина - 20% от радиуса
        thickness = central_radius * 0.2
        inner_radius = max(0.1, central_radius - thickness / 2)
        outer_radius = central_radius + thickness / 2

        self.model['center_x'] = center[0]
        self.model['center_y'] = center[1]
        self.model['center_z'] = center[2]
        self.model['inner_radius'] = inner_radius
        self.model['outer_radius'] = outer_radius
        self.model['height'] = height
        self.model['central_radius'] = central_radius
        self.model['thickness'] = thickness

    def calc_distances(self, points: NDArray) -> NDArray:
        center_x = self.model['center_x']
        center_y = self.model['center_y']
        center_z = self.model['center_z']

        inner_r = self.model['inner_radius']
        outer_r = self.model['outer_radius']
        height = self.model['height']

        distances = []

        for point in points:
            x, y, z = point

            # Расстояние до оси Z в плоскости XY
            dist_to_axis = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Расстояние по высоте
            dist_height = abs(z - center_z)

            # Расстояние до цилиндрического кольца
            if dist_height <= height / 2:
                if dist_to_axis < inner_r:
                    dist = inner_r - dist_to_axis
                elif dist_to_axis > outer_r:
                    dist = dist_to_axis - outer_r
                else:
                    dist = 0
            else:
                dist = dist_height - height / 2
                if dist_to_axis < inner_r:
                    dist = max(dist, inner_r - dist_to_axis)
                elif dist_to_axis > outer_r:
                    dist = max(dist, dist_to_axis - outer_r)

            distances.append(dist)

        return np.array(distances)

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances(np.array([point]))[0]

    def get_center(self) -> NDArray:
        return np.array([
            self.model['center_x'],
            self.model['center_y'],
            self.model['center_z']
        ])

    def get_radii(self) -> tuple:
        return self.model['inner_radius'], self.model['outer_radius']

    def get_height(self) -> float:
        return self.model['height']

    def get_central_radius(self) -> float:
        return self.model['central_radius']

    def get_thickness(self) -> float:
        return self.model['thickness']

    def get_model(self) -> dict:
        return self.model.copy()