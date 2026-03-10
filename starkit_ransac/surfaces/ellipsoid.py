import pdb
import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from starkit_ransac.utils import normalize

class Ellipsoid3D(AbstractSurfaceModel):
    def __init__(
            self,
            axes:ArrayLike|None=None,
            radii:ArrayLike|None=None,
            center:ArrayLike|None=None,
            polynomial:ArrayLike|None=None
        ) -> None:
        super().__init__()

        if axes is not None and\
           radii is not None and\
           center is not None:
               axes, radii = self.sort_axes_and_radii(
                   np.asarray(axes), 
                   np.asarray(radii)
                )
               if polynomial is not None:
                    raise ValueError(
                            "Either polynomial, or axes, radii and center must"
                            "be passed to __init__."
                    )
               else:
                    polynomial = self.axes_to_polynomial(axes, radii, center)
        else:
            if polynomial is not None:
                axes, radii, center = self.polynomial_to_axes(polynomial)
                axes, radii = self.sort_axes_and_radii(axes, radii)

        if axes is not None:
            self.axes = axes
        else:
            self.axes = np.full((3,3), np.nan)

        if radii is not None:
            self.radii = radii
        else:
            self.radii = np.full(3, np.nan)

        if center is not None:
            self.center = center
        else:
            self.center = np.full(3, np.nan)

        if polynomial is not None:
            self.polynomial = polynomial
        else:
            self.polynomial = np.full(9, np.nan)
        
        self.num_samples = 9

    @property
    def center(self) -> NDArray[np.float64]:
        return self._center

    @center.setter
    def center(self, center:ArrayLike):
        self._center = np.copy(center)

    @property
    def axes(self) -> NDArray[np.float64]:
        return self._axes

    @axes.setter
    def axes(self, axes:ArrayLike):
        axes = np.asarray(axes)
        if axes.shape != (3,3):
            raise ValueError("'axes' must be a 3x3 array")

        if np.isnan(axes).all():
            self._axes = np.copy(axes)
            return

        if not np.allclose(np.dot(axes[0], axes[1]), 0) or \
           not np.allclose(np.dot(axes[1], axes[2]), 0) or \
           not np.allclose(np.dot(axes[2], axes[0]), 0):
               raise ValueError("Axes must be perpendicular to each other.")
        self._axes = normalize(axes, -1)

    @property
    def radii(self) -> NDArray[np.float64]:
        return self._radii
   
    @radii.setter
    def radii(self, radii:ArrayLike):
        radii = np.array(radii)
        if np.isnan(radii).all():
            self._radii = radii
            return
        if not (radii > 0).all():
            raise ValueError("All radii must be greater than 0")
        if radii.shape != (3,):
            raise ValueError("There must be 3 radii values")
        self._radii = np.copy(radii)


    @property
    def polynomial(self) -> NDArray[np.float64]:
        return self._polynomial

    @polynomial.setter
    def polynomial(self, polynomial:ArrayLike):
        polynomial = np.asarray(polynomial)
        if polynomial.shape != (9,):
            raise ValueError("Polynomial must have 9 coefficients")
        self._polynomial = np.copy(polynomial)


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
        self.polynomial = polynomial

        try:
            axes, radii, center = self.polynomial_to_axes(polynomial)
        except LinAlgError as lae:
            self.center = np.inf
            self.axes = np.inf
            self.radii = np.inf
            return False
        axes, radii = self.sort_axes_and_radii(axes, radii)

        try:
            self.center = center
            self.axes = axes
            self.radii = radii
        except ValueError as e:
            return False

        return True


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

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances(np.array([point]))[0]

    def calc_distances(self, points: NDArray) -> NDArray:
        """
            Calculates algebraic distance
        """
        poly = self.polynomial
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

    def __repr__(self):
        res = ''
        res += 'center: ' + str(self.center)
        res += '\n'
        res += 'radii: ' + str(self.radii)
        res += '\n'
        res += 'axes: \n' + str(self.axes)
        res += '\n'
        res += 'polynomial: ' + str(self.polynomial)
        return res


