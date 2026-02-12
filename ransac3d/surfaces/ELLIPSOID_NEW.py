import numpy as np
from numpy.typing import NDArray
from ransac3d.abstract_surface import AbstractSurfaceModel
from copy import deepcopy


class EllipsoidModel(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.num_samples = 9  # Минимум 9 точек для эллипсоид
        self._model_data = {
            'coefficients': None,
            'center': None,
            'radii': None,
            'rotation': None
        }

    def fit_model(self, points: NDArray):
        """Строит эллипсоид по 9 точкам."""
        assert len(points) == self.num_samples

        # Аппроксимация эллипсоида
        coefficients = self._fit_ellipsoid(points)
        self._model_data['coefficients'] = coefficients

        # Извлекаем параметры
        if coefficients is not None:
            params = self._ellipsoid_parameters_from_coefficients(coefficients)
            if params is not None:
                center, radii, rotation = params
                self._model_data['center'] = center
                self._model_data['radii'] = radii
                self._model_data['rotation'] = rotation

    def calc_distances(self, points: NDArray) -> NDArray:
        """Вычисляет расстояния от точек до эллипсоида."""
        if self._model_data['coefficients'] is None:
            return np.full(len(points), np.inf)

        distances = np.zeros(len(points))
        for i, point in enumerate(points):
            distances[i] = self._algebraic_distance_3d(point, self._model_data['coefficients'])

        return distances

    def calc_distance_one_point(self, point: NDArray) -> float:
        """Вычисляет расстояние от одной точки до эллипсоида."""
        if self._model_data['coefficients'] is None:
            return float('inf')

        return self._algebraic_distance_3d(point, self._model_data['coefficients'])

    def _fit_ellipsoid(self, points):
        """Аппроксимация эллипсоида по точкам."""
        n = len(points)
        A_mat = np.zeros((n, 10))

        for i, point in enumerate(points):
            x, y, z = point
            A_mat[i] = [x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, 1]

        try:
            _, _, V = np.linalg.svd(A_mat)
            coefficients = V[-1, :]
            norm = np.sqrt(np.sum(coefficients ** 2))

            if norm == 0:
                return None

            return coefficients / norm

        except np.linalg.LinAlgError:
            return None

    def _ellipsoid_parameters_from_coefficients(self, coefficients):
        """Извлекает параметры эллипсоида из коэффициентов."""
        A, B, C, D, E, F, G, H, I, J = coefficients

        Q = np.array([[A, D / 2, E / 2], [D / 2, B, F / 2], [E / 2, F / 2, C]])
        L = np.array([G, H, I]) / 2

        try:
            center = -np.linalg.solve(Q, L)
            J_new = J + center @ Q @ center + 2 * L @ center

            eigenvalues, eigenvectors = np.linalg.eig(Q)

            # Проверяем, что все собственные значения одного знака
            if not np.all(np.sign(eigenvalues) == np.sign(eigenvalues[0])):
                return None

            abc_squared = -J_new / eigenvalues
            if np.any(abc_squared <= 0):
                return None

            abc = np.sqrt(abc_squared)
            idx = abc.argsort()[::-1]
            radii = abc[idx]
            rotation = eigenvectors[:, idx]

            return center, radii, rotation

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _algebraic_distance_3d(self, point, coefficients):
        """Алгебраическое расстояние от точки до эллипсоида."""
        x, y, z = point
        A, B, C, D, E, F, G, H, I, J = coefficients

        distance = (A * x * x + B * y * y + C * z * z + D * x * y + E * x * z + F * y * z +
                    G * x + H * y + I * z + J)

        norm = np.sqrt(A * A + B * B + C * C + D * D + E * E + F * F + G * G + H * H + I * I + J * J)
        if norm == 0:
            return float('inf')

        return abs(distance / norm)

    def __repr__(self):
        result = ''
        if self._model_data['center'] is not None:
            center = self._model_data['center']
            radii = self._model_data['radii']
            result += f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n"
            result += f"Radii: ({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})\n"
        else:
            result = "Ellipsoid not fitted\n"
        return result