import numpy as np
from numpy.typing import NDArray
from ransac3d.abstract_surface import AbstractSurfaceModel

class Ellipsoid3D(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = {
            'axes' : np.ones((3,3)) * np.nan,
            'axes_lengths' : np.ones(3) * np.nan,
            'center' : np.ones(3) * np.nan,
            'polynomial' : np.ones(9) * np.nan
        }
        self.num_samples = 9

    def fit_model(self, points: NDArray):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        eq_matrix = np.array([
            x**2,
            2 * x * y,
            y**2,
            2 * x * z,
            2 * y * z,
            z**2,
            -2 * x,
            -2 * y,
            -2 * z
        ]).T
        intermediate_solution = np.linalg.solve(
                eq_matrix, -np.ones(len(points))
        )
        m11, m12, m22, m13, m23, m33, b1, b2, b3 = intermediate_solution
        M = np.array([
            [m11, m12, m13],
            [m12, m22, m23],
            [m13, m23, m33]
        ])
        b = np.array([[b1, b2, b3]]).T
        # solve for M.v = b
        v = np.linalg.solve(M, b)
        coeff = 1/(v.T @ M @ v - 1)
        A = coeff * M

        # v is the center of the ellipsoid 
        # A is a matrix of parameters of ellipsoid:
        # it's eigenvectors are the axes of an ellipsoid
        # and it's eigenvalues are inverse squares of it's axes' lengths.
        vals, axes = np.linalg.eig(A)
        axes_lengths = 1/np.sqrt(vals)

        print(vals)
        if (vals < 0).any():
            raise ValueError("Could not fit an ellipse to points")

        self.model['center'] = v
        self.model['axes'] = axes
        self.model['axes_lengths'] = axes_lengths
        self.model['polynomial'] = np.array([
            m11,     # x^2
            2 * m12, # xy
            m22,     # y^2
            2 * m13, # xz
            2 * m23, # yz
            m33,     # z^2
            -2 * b1, # x
            -2 * b2, # y
            -2 * b3  # z
        ])

    def calc_distance_one_point(self, point: NDArray) -> float:
        return 0.

    def calc_distances(self, points: NDArray) -> NDArray:
        return np.zeros(1)
