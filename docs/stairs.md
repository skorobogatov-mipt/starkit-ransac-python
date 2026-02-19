# StepPlane (Stairs)
This class represents a staircase surface in 3D space.

# Model parameters
## stair_height
Total height of the staircase, computed as `max(z) - min(z)` from the sample.
## step_width
Horizontal step size along the staircase direction.
## step_height
Vertical rise per step.
## rotation_deg
Rotation of the staircase around the Z axis, in degrees.

# Model fitting
Let $X$ be the matrix passed to `fit_model` with dimensions $N \times 3$, where
$N = \text{num\_samples}$.

The algorithm:
1. Split points into low-Z and high-Z groups.
2. Compute the mean $(x, y)$ of each group and define a direction vector
   from low to high.
3. Compute the angle of that direction:
$$
\theta = \arctan2(d_y, d_x)
$$
4. Rotate all $(x, y)$ into the local staircase frame and estimate:
   - `step_width` from the period of $x_{local}$
   - `step_height` from the period of $z$
5. Compute:
$$
stair\_height = \max(z) - \min(z)
$$

# Distance evaluation
Distances are computed to the nearest horizontal tread.
For a point $(x, y, z)$:
1. Rotate into local frame to get $x_{local}$.
2. Determine the tread index:
$$
k = \left\lfloor \frac{x_{local}}{step\_width} \right\rfloor
$$
3. Tread height:
$$
z_{tread} = k \cdot step\_height
$$
4. Distance:
$$
d = |z - z_{tread}|
$$

Note: The current implementation returns only the tread distance
(`dist_tread`).

# Synthetic data generation
The static method `generate_ladder_points(...)` builds a synthetic staircase
point cloud by:
1. Sampling horizontal treads for each step.
2. Sampling vertical risers between steps.
3. Rotating the cloud by `rotation_deg`.
4. Adding Gaussian noise if `noise_sigma > 0`.
