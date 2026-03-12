import numpy as np
from numpy.typing import ArrayLike
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
from copy import deepcopy

class MobiusStrip(AbstractSurfaceModel):
    def __init__(
            self, 
            center=[np.nan, np.nan, np.nan], 
            radius=np.nan, 
            normal=[np.nan, np.nan, np.nan], 
            width=np.nan, 
            orientation=np.nan, 
            start_vector=[np.nan, np.nan, np.nan]
        ) -> None:
        super().__init__()
        self.center = np.array(center).astype(np.float64)
        self.radius = radius
        self.normal = np.array(normal).astype(np.float64)

        self.width = width
        self.orientation = orientation
        self.start_vector = np.array(start_vector).astype(np.float64)
        self.num_samples = 4

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
    def start_vector(self) -> NDArray[np.float64]:
        return self._start_vector

    @start_vector.setter
    def start_vector(self, start_vector:ArrayLike):
        self._start_vector = np.array(start_vector, np.float64)

    @property
    def normal(self) -> NDArray[np.float64]:
        return self._normal

    @normal.setter
    def normal(self, normal:ArrayLike):
        if np.asarray(normal).flatten().shape != (3,):
            raise ValueError("Normal must be a vector of 3 elements")
        self._normal = np.asarray(normal) / np.linalg.norm(normal)

    def fit_model(
            self,
            points:NDArray
            ):
        assert len(points) == self.num_samples
        orientation = 1
        AB = points[1] - points[0]
        AC = points[2] - points[0]
        CB = points[2] - points[1]
        normal = np.cross(AB, AC)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            return False
            # raise ValueError("The Mobius strip cannot be " \
            # "constructed from the given points, since a circle cannot " \
            # "be constructed from the first three of them.")
        normal = normal / n_norm
        if normal[2] < 0:
            normal = -normal

        a = np.linalg.norm(CB)
        b = np.linalg.norm(AC)
        c = np.linalg.norm(AB)
        radius = (a * b * c) / (2.0 * n_norm)
        
        wA = a * a * (b * b + c * c - a * a)
        wB = b * b * (c * c + a * a - b * b)
        wC = c * c * (a * a + b * b - c * c)
        denom = wA + wB + wC
        if abs(denom) < 1e-18:
            return False
            # raise ValueError("The Mobius strip cannot be constructed from the given points.")
        center = (wA * points[0] + wB * points[1] + wC * points[2]) / denom

        # --- проекция P на плоскость окружности ---
        d_plane = np.dot(points[3] - center, normal)
        P_proj = points[3] - d_plane * normal
        r_proj = P_proj - center
        r_proj_norm = np.linalg.norm(r_proj)

        # Радиальное направление (единичное, ортогональное normal)
        if r_proj_norm < 1e-12:
            radial_dir = points[0] - center
            radial_dir -= np.dot(radial_dir, normal) * normal
            radial_dir /= np.linalg.norm(radial_dir)
        else:
            radial_dir = r_proj / r_proj_norm

        closest_point = center + radius * radial_dir
        distance_to_circle = np.linalg.norm(points[3] - closest_point)
        width = 2*distance_to_circle

        
        # --- второй вектор базиса в плоскости окружности ---
        tangent_dir = np.cross(normal, radial_dir)
    
        # twist angle
        v = closest_point - points[3]
        v_norm = np.linalg.norm(v)
        v /= v_norm
        v_proj_on_radial_dir = np.dot(v, radial_dir)
        v_proj_on_normal = np.dot(v, normal)
        if v_norm < 1e-12:
            twist_angle = 0.0
        else:
            twist_angle = np.arccos(np.clip(v_proj_on_normal, -1, 1))
            if v_proj_on_radial_dir < 0:
                twist_angle = np.pi - twist_angle
    
        # circle_angle положительный, если orientation положительная - лента закручивается 
        # против часовой стрелки при обходе окружности против часовой стрелки
        circle_angle = 2*(np.pi/2 - twist_angle)*orientation
    
        # --- точка нулевой скрутки (сечение ленты лежит в плоскости окружности) ---
        start_vector = np.cos(circle_angle) * radial_dir - np.sin(circle_angle) * tangent_dir

        self.center = center
        self.radius = radius
        self.normal = normal
        self.orientation = orientation
        self.width = width
        self.start_vector = start_vector
        return True

    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        
        grid_size = 100
        activate_Brent = False
        
        points_in_mobius_coordinates = self.get_points_coordinates_in_mobius(points)
        
        theta_grid = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
        objective_values = self.objective_function_multi_target(
            theta_grid,
            points_in_mobius_coordinates
        )
    
        if not activate_Brent:
            return np.min(objective_values, axis=1)**0.5
        
        best_indices = np.argmin(objective_values, axis=1)
        initial_thetas = theta_grid[best_indices]
        step = 2.0 * np.pi / grid_size
        num_targets = target_points.shape[0]
        distances = np.empty(num_targets)
        for target_index in range(num_targets):
            bracket = (
                initial_thetas[target_index] - step,
                initial_thetas[target_index] + step
            )
        
            result = minimize_scalar(
                objective_function,
                bracket=bracket,
                args=(points_in_mobius_coordinates[target_index]),
                method="brent"
            )
            distances[target_index] = result.fun
    
        return distances**0.5

    def calc_distance_one_point(self, point: NDArray) -> float:
        grid_size = 100
        activate_Brent = False
        
        point_in_mobius_coordinates = get_point_coordinate_in_mobius(point)
        
        theta_grid = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
        objective_values = objective_function_array(
            theta_grid,
            point_in_mobius_coordinates
        )
    
        if not activate_Brent:
            return np.min(objective_values)**0.5
        
        best_index = np.argmin(objective_values)
        step = 2.0 * np.pi / grid_size
    
        bracket = (
            theta_grid[best_index] - step,
            theta_grid[best_index] + step
        )
    
        result = minimize_scalar(
            objective_function,
            bracket=bracket,
            args=(point_in_mobius_coordinates),
            method="brent"
        )
    
        distance = result.fun
    
        return distance**0.5

    def get_points_coordinates_in_mobius(self, points):
        relative_coordinates = points - self.center
        x_axis = self.start_vector
        z_axis = self.normal
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    
        transformed_points = relative_coordinates @ rotation_matrix
    
        return transformed_points

    def get_point_coordinate_in_mobius(info, point):
        relative_coordinates=point-self.center
        x_axis = self.start_vector.copy()
        z_axis = self.normal.copy()
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        transformed_point = np.array([
            np.dot(relative_coordinates, x_axis),
            np.dot(relative_coordinates, y_axis),
            np.dot(relative_coordinates, z_axis)
        ])
        return transformed_point

    def objective_function(self, theta_value, target_point):
        if self.width == 0:
            optimal_v = 0.0
        else:
            x = target_point[0]
            y = target_point[1]
            z = target_point[2]
            optimal_v = np.clip(
                2* (-(radius - x *np.cos(theta_value) - y *np.sin(theta_value))*
                     np.cos(theta_value*self.orientation/2) +
                     z*np.sin(theta_value*self.orientation/2))
                     /self.width, 
                -1.0, 1.0)

        a_vector = compute_a_vector(theta_value)
        b_vector = compute_b_vector(theta_value, target_point)
    
        residual = b_vector + optimal_v * a_vector
        return np.dot(residual, residual)

    def objective_function_array(self, theta_values, target_point):
        if self.width == 0:
            optimal_v = np.zeros_like(theta_values)
        else:
            x = target_point[0]
            y = target_point[1]
            z = target_point[2]
            half_theta = theta_values * self.orientation / 2.0
            optimal_v = np.clip(2*
                                (-(radius - x *np.cos(theta_values) - y *np.sin(theta_values))*
                                 np.cos(half_theta) + z*np.sin(half_theta))
                                 /self.width, -1.0, 1.0)
    
        a_vectors = compute_a_vector_array(theta_values)
        b_vectors = compute_b_vector_array(theta_values, target_point)
    
        residual_vectors = b_vectors + optimal_v[:, np.newaxis] * a_vectors
        return np.einsum("ij,ij->i", residual_vectors, residual_vectors)

    def objective_function_multi_target(self, theta_values, target_points):
        num_thetas = theta_values.shape[0]
        num_targets = target_points.shape[0]
        if self.width == 0:
            optimal_v = np.zeros((num_targets, num_thetas))
        else:
            x_coords = target_points[:, 0][:, None]
            y_coords = target_points[:, 1][:, None]
            z_coords = target_points[:, 2][:, None]
    
            theta_row = theta_values[None, :] 
            half_theta = theta_row * self.orientation / 2.0
            optimal_v_unclipped = (
                2.0
                * (
                    z_coords * np.sin(half_theta)
                    - np.cos(half_theta)
                    * (
                        self.radius
                        - x_coords * np.cos(theta_row)
                        - y_coords * np.sin(theta_row)
                    )
                )/self.width
            )
            optimal_v = np.clip(optimal_v_unclipped, -1.0, 1.0)
    
        a_vectors = self.compute_a_vector_array(theta_values)
        b_vectors = self.compute_b_vector_multi_target(
            theta_values, target_points
        )
    
        residual_vectors = (
            b_vectors
            + optimal_v[..., None] * a_vectors[None, :, :]
        )
        return np.einsum("mtj,mtj->mt", residual_vectors, residual_vectors)

    def compute_a_vector_array(self, theta_values):
        # helper function for calculating distance

        half_theta = theta_values * self.orientation / 2.0
        scale = self.width / 2.0
    
        return scale * np.column_stack(
            [
                np.cos(theta_values) * np.cos(half_theta),
                np.sin(theta_values) * np.cos(half_theta),
                np.sin(half_theta),
            ]
        )
    
    def compute_b_vector_multi_target(self, theta_values, target_points):
        # helper function for calculating distance:
        theta_row = theta_values[None, :]
        x_coords = target_points[:, 0][:, None]
        y_coords = target_points[:, 1][:, None]
        z_coords = target_points[:, 2][:, None]
    
        return np.stack(
            [
                self.radius * np.cos(theta_row) - x_coords,
                self.radius * np.sin(theta_row) - y_coords,
                -z_coords * np.ones_like(theta_row),
            ],
            axis=2
        )
        

    def compute_b_vector_array(self, theta_values, target_point):
        # helper function for calculating distance
        x_coord = target_point[0]
        y_coord = target_point[1]
        z_coord = target_point[2]
    
        return np.column_stack(
            [
                self.radius * np.cos(theta_values) - x_coord,
                self.radius * np.sin(theta_values) - y_coord,
                -z_coord * np.ones_like(theta_values),
            ]
        )

    def compute_b_vector(self, theta_value, target_point):
        return np.array([
            self.radius * np.cos(theta_value) - target_point[0],
            self.radius * np.sin(theta_value) - target_point[1],
            -target_point[2]
        ])

    def compute_a_vector(self, theta_value):
        half_theta = theta_value * 0.5 * self.orientation
        coefficient = self.width * 0.5
    
        return np.array([
            coefficient * np.cos(theta_value) * np.cos(half_theta),
            coefficient * np.sin(theta_value) * np.cos(half_theta),
            coefficient * np.sin(half_theta)
        ])

