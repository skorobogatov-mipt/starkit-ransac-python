import pdb
from typing import Iterable
import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from starkit_ransac.utils import normalize

class Ellipsoid3D(AbstractSurfaceModel):
    def __init__(
            self,
            axes:Iterable|None=None,
            radii:Iterable|None=None,
            center:Iterable|None=None,
            polynomial:Iterable|None=None
        ) -> None:
        super().__init__()
        if axes is not None:
            axes = np.array(axes)
            if axes.shape != (3,3):
                raise ValueError("'axes' must be a 3x3 array")

            if not np.allclose(np.dot(axes[0], axes[1]), 0) or \
               not np.allclose(np.dot(axes[1], axes[2]), 0) or \
               not np.allclose(np.dot(axes[2], axes[0]), 0):
                   raise ValueError("Axes must be perpendicular to each other.")
            axes = normalize(axes, -1)

        if radii is not None:
            radii = np.array(radii)
            if not (radii > 0).all():
                raise ValueError("All radii must be greater than 0")
            if radii.shape != (3,):
                raise ValueError("There must be 3 radii values")

        if center is not None:
            center = np.array(center)
            if center.shape != (3,):
                raise ValueError("Center must be 3d vector")

        if polynomial is not None:
            polynomial = np.array(polynomial)
            if polynomial.shape != (9,):
                raise ValueError("Polynomial must have 9 coefficients")
    
        if axes is not None and\
           radii is not None and\
           center is not None:
               axes, radii = self.sort_axes_and_radii(axes, radii)
               if polynomial is None:
                    polynomial = self.axes_to_polynomial(axes, radii, center)
               else:
                    raise ValueError(
                            "Polynomial must be None,"
                            "when axes, radii and center are passed in __init__"
                    )
        
        self.model = {
            'axes' : axes,
            'radii' : radii,
            'center' : center,
            'polynomial' : polynomial
        }
        self.num_samples = 9

    def fit_model(self, points: NDArray):
        """
            Fits an ellipsoid based on the following equation:
            Ax^2 + By^2 + Cy^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        eq_matrix = np.array([
            x**2,
            y**2,
            z**2,
            x*y,
            x*z,
            y*z,
            x,
            y,
            z
        ]).T
        polynomial = np.linalg.solve(
                eq_matrix, np.ones(len(points))
        )
        self.model['polynomial'] = polynomial

        try:
            axes, radii, center = self.polynomial_to_axes(polynomial)
        except LinAlgError as lae:
            self.model['center'] = np.inf
            self.model['axes'] = np.inf
            self.model['radii'] = np.inf
            return
        axes, radii = self.sort_axes_and_radii(axes, radii)

        self.model['center'] = center
        self.model['axes'] = axes
        self.model['radii'] = radii

    @staticmethod
    def axes_to_polynomial(axes, radii, center):
        axes = np.asarray(axes)
        radii = np.asarray(radii)
        center = np.asarray(center)

        eigenvalue_matrix = np.diag(1/radii**2)

        M = axes.T @ eigenvalue_matrix @ axes
        k = (1 - center.T @ M @ center)

        Q = M / k
        A = Q[0, 0]
        B = Q[1, 1]
        C = Q[2, 2]
        D = Q[0, 1] * 2
        E = Q[0, 2] * 2
        F = Q[1, 2] * 2

        # get linear coeffs
        b = (-2 * Q @ center).squeeze()
        G,H,I = b

        return np.array([A,B,C,D,E,F,G,H,I])

    @staticmethod
    def polynomial_to_axes(polynomial):
        A,B,C,D,E,F,G,H,I = polynomial

        b = np.array([G,H,I])

        Q = np.array([
            [A, D/2, E/2],
            [D/2, B, F/2],
            [E/2, F/2, C]
        ])
        c = np.linalg.solve(-2*Q, b)

        k = 1/(1 + c.T @ Q @ c)
        M = Q * k

        inv_rad, axes = np.linalg.eig(M)
        radii = np.sqrt(1/inv_rad)
        axes = axes.T

        return axes, radii, c

    @staticmethod
    def sort_axes_and_radii(axes, radii):
        order = np.argsort(radii)
        return axes[:, order], radii[order]
        return axes, radii

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances(np.array([point]))[0]

    def calc_distances(self, points: NDArray) -> NDArray:
        """
            Calculates algebraic distance
        """
        poly = self.model['polynomial']
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        # numpy multiplication for speed
        poly_values = np.dot(
            poly, 
            np.array([
                x**2, 
                y**2, 
                z**2, 
                x*y,
                x*z, 
                y*z, 
                x, 
                y, 
                z
            ])
        )
        return np.abs(poly_values - 1)


