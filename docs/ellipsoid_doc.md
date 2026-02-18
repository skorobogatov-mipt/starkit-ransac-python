# EllipsoidModel

The EllipsoidModel class is an implementation of an ellipsoid surface in three-dimensional space. It inherits from the abstract base class `AbstractSurfaceModel` and is intended for use in the RANSAC algorithm to fit an ellipsoid to a point cloud.

The model describes a general ellipsoid with the possibility of rotation and displacement relative to the origin. The fitting is performed on 9 points using the algebraic least squares method via singular value decomposition (SVD).

# Model Parameters

The model stores its parameters in the dictionary `_model_data`, which contains the following keys:

# `coefficients`

An array of 10 coefficients describing an ellipsoid in general algebraic form:
$$
A x^2 + B y^2 + C z^2 + D xy + E xz + F yz + G x + H y + I z + J = 0
$$
The coefficients are normalized so that their Euclidean norm is 1.

# `center`

The center of the ellipsoid is a three—dimensional point $(c_x, c_y, c_z)^T$.

# `radii`

The semi—axes of the ellipsoid are a vector $(r_a, r_b, r_c)^T$ ordered in descending order.

# `rotation`

A 3×3 rotation matrix R that defines the orientation of the ellipsoid. The columns of the matrix correspond to the directions of the semi-axes in the order specified in `radii`.

# The process of fitting the model

The model is fitted using the `fit_model` method, which takes a 9x3 matrix of `points` — exactly 9 points needed to define an ellipsoid.

##1. Building the system matrix

For each point $(x_i, y_i, z_i)$ a string is being formed:
$$
\mathbf{a}_i = [x_i^2,\ y_i^2,\ z_i^2,\ x_i y_i,\ x_i z_i,\ y_i z_i,\ x_i,\ y_i,\ z_i,\ 1]
$$
A 9×10 matrix A is constructed from all the rows.

## 2. Solution via SVD

The singular value decomposition is performed:

$A=UΣV^T$

The solution corresponds to the last row of the matrix V (the singular vector corresponding to the smallest singular number). The resulting vector is normalized:

The vector of coefficients is normalized:
$$
\mathbf{c} = \frac{V[-1]}{\|V[-1]\|}
$$

## 3. Extracting parameters
From the vector of coefficients $c=(A,B,C,D,E,F,G,H,I,J)$ are formed:

Matrix of quadratic form:
$$
Q = \begin{bmatrix}
A & D/2 & E/2 \\
D/2 & B & F/2 \\
E/2 & F/2 & C
\end{bmatrix}
$$
Vector of linear terms:
$$
\mathbf{L} = \frac{1}{2} \begin{bmatrix} G \\ H \\ I \end{bmatrix}
$$
The center of the ellipsoid is located as:
$$
\mathbf{c}_{\text{center}} = -Q^{-1} \mathbf{L}
$$
Then the free term in the centered system is calculated:
$$
J' = J + \mathbf{c}_{\text{center}}^T Q\mathbf{c}_{\text{center}}+ 2 \mathbf{L}^T\mathbf{c}_{\text{center}}
$$
The eigenvalues of $\lambda_i$ and the eigenvectors of $Q$ give:

Semi-axes: $r_i = \sqrt{-J' / \lambda_i}$

Orientation: a rotation matrix made up of eigenvectors ordered by descending semi-axes.

# Calculating the distance

The distance from an arbitrary point $p=(x,y,z)^T$ to the ellipsoid is calculated using the `calc_distance_one_point` method (or, for an array of points, `calc_distances`).

The algebraic distance normalized by the Euclidean norm of the coefficient vector is used.

# Formula
Let $c=(A,B,C,D,E,F,G,H,I,J)$ be the coefficients of the model. Then the algebraic distance is:

$$
d(\mathbf{p}) = \frac{|A x^2 + B y^2 + C z^2 + D xy + E xz + F yz + G x + H y + I z + J|}{\sqrt{A^2 + B^2 + C^2 + D^2 + E^2 + F^2 + G^2 + H^2 + I^2 + J^2}}
$$

If the coefficient norm is zero, `inf` is returned.

This distance is an algebraic measure of the proximity of a point to the surface of an ellipsoid and is used in RANSAC to classify points as `inliers/outliers` by threshold value.