import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from starkit_ransac.abstract_surface import AbstractSurfaceModel

class Ellipse2D(AbstractSurfaceModel):
    def __init__(
            self,
            rotation:float|None=None,
            radius_1:float|None=None,
            radius_2:float|None=None,
            center:ArrayLike|None=None,
            polynomial:ArrayLike|None=None
        ) -> None:
        self.rotation:float = rotation
        if radius_1 is not None and\
           radius_2 is not None:
                self.major_radius:float = max(radius_1, radius_2)
                self.minor_radius:float = min(radius_1, radius_2)
        if center is not None:
            center = np.copy(center)
            if center.shape != (2,):
                raise ValueError("Center must be an array of length 2")
        if polynomial is not None:
            polynomial = np.copy(polynomial)
            if polynomial.shape != (5,):
                raise ValueError("Ellipse polynomial must contain 5 numbers: Ax^2 + Bxy^ + Cy^2 + Dx + Ey = 1")

        if center is not None and\
           radius_1 is not None and\
           radius_2 is not None and\
           rotation is not None:
                polynomial = self.polynomial_from_rotation_and_radii(
                    rotation, 
                    radius_1,
                    radius_2,
                    center
                )

        self.polynomial = polynomial
        self.center = center
        self.num_samples = 5

    def fit_model(
            self, 
            points: ArrayLike
        ):
        # Ax^2 + Bxy^ + Cy^2 + Dx + Ey + 1 = 0
        points = np.asarray(points)

        x = points[:, 0]
        y = points[:, 1]

        A = np.array([
            x**2,
            x*y,
            y**2,
            x,
            y
        ]).T
        poly = np.linalg.solve(A, -np.ones(len(points))).squeeze()

        # check invariant
        A = poly[0]
        B = poly[1]
        C = poly[2]
        self.polynomial = poly
        invariant = B**2 - 4 * A * C
        if invariant > 0:
            self.rotation = np.inf
            self.minor_radius = np.inf
            self.major_radius = np.inf
            self.center = np.full((2,), np.inf)
            return

        self.rotation,\
        self.major_radius,\
        self.minor_radius,\
        self.center = \
                self.rotation_and_radii_from_polynomial(poly)


    @staticmethod
    def rotation_and_radii_from_polynomial(polynomial):
        A,B,C,D,E = polynomial

        # https://en.wikipedia.org/wiki/Ellipse#Parametric_representation 
        # note that F = 1 for our case
        term_1 = 2 * (A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C))
        plus_minus_term = np.sqrt((A-C)**2 + B**2)
        divisor = B**2 - 4*A*C
        a = -np.sqrt(term_1*((A+C) + plus_minus_term)) / divisor
        b = -np.sqrt(term_1*((A+C) - plus_minus_term)) / divisor
        x0 = (2*C*D - B*E)/divisor
        y0 = (2*A*E - B*D)/divisor
        rotation = 0.5*np.arctan2(-B, C-A)
        center = np.array((x0, y0))
        return rotation, a,b, center

    @staticmethod
    def polynomial_from_rotation_and_radii(
            rotation, 
            radius_1, 
            radius_2,
            center
        ):
        # https://en.wikipedia.org/wiki/Ellipse#Parametric_representation 
        a = max(radius_1, radius_2)
        b = min(radius_1, radius_2)
        x0,y0 = center
        sin_th = np.sin(rotation)
        cos_th = np.cos(rotation)
        A = a**2 * sin_th**2 + b**2 * cos_th**2
        B = 2*(b**2 - a**2)*sin_th*cos_th
        C = a**2 * cos_th**2 + b**2 + sin_th**2
        D = -2*A*x0 - B*y0
        E = -B*x0 - 2 * C*y0
        poly = np.asarray([A,B,C,D,E])
        F = A * x0**2 + B * x0*y0 + C * y0**2 - a**2 * b**2
        # divide so that F = 1
        poly /= F
        return poly

    def calc_distances(self, points: NDArray) -> NDArray:
        points = np.asarray(points)
        poly = self.polynomial
        x = points[:, 0]
        y = points[:, 1]
        poly_values = np.dot(
            poly,
            np.array([
                x**2,
                x*y,
                y**2,
                x,
                y
            ])
        )

        return np.abs(poly_values + 1)

    def calc_distance_one_point(self, point: NDArray) -> float:
        return self.calc_distances([point])[0]

