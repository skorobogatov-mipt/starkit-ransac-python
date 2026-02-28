# CylindricalRing
This class represents a cylindrical ring (hollow cylinder) in 3D space. A cylindrical ring is defined by two concentric cylinders with different radii, sharing the same axis and height.

# Model parameters
## center_x
This parameter represents the x coordinate of the ring's center point.

## center_y
This parameter represents the y coordinate of the ring's center point.

## center_z
This parameter represents the z coordinate of the ring's center point.

## axis_x
This parameter represents the x component of the cylinder axis direction vector.

## axis_y
This parameter represents the y component of the cylinder axis direction vector.

## axis_z
This parameter represents the z component of the cylinder axis direction vector.

## inner_radius
This parameter represents the inner radius of the ring (distance from axis to inner cylindrical surface).

## outer_radius
This parameter represents the outer radius of the ring (distance from axis to outer cylindrical surface).

## height
This parameter represents the height of the ring along the cylinder axis.

## central_radius
This parameter represents the mean radius between inner and outer radii: $(inner\_radius + outer\_radius)/2$.

## thickness
This parameter represents the wall thickness of the ring: $outer\_radius - inner\_radius$.

# Model fitting
Let $X$ be the matrix that is passed to `fit_model`. This matrix has dimensions $5 \times 3$ (minimum 5 points are required for fitting).

The fitting process uses the first 5 points $(p_1, p_2, p_3, p_4, p_5)$:

1. **Axis direction** is determined from the first two points:
   $$
   \vec{a} = \frac{p_2 - p_1}{\|p_2 - p_1\|}
   $$
   If the points coincide, axis defaults to $(0, 0, 1)$.

2. **Center point** is the midpoint between $p_1$ and $p_2$:
   $$
   \vec{c} = \frac{p_1 + p_2}{2}
   $$

3. **Radii** are estimated from the remaining points $p_3, p_4, p_5$:
   For each point $p$, calculate distance to axis:
   $$
   d_i = \|(p - \vec{c}) - ((p - \vec{c}) \cdot \vec{a})\vec{a}\|
   $$
   Then:
   $$
   \begin{cases}
   inner\_radius = \min(d_3, d_4, d_5) \times 0.9\\
   outer\_radius = \max(d_3, d_4, d_5) \times 1.1
   \end{cases}
   $$

4. **Height** is calculated as the range of projections onto the axis:
   $$
   h_i = (p_i - \vec{c}) \cdot \vec{a}
   $$
   $$
   height = \max(h_1, h_2, h_3, h_4, h_5) - \min(h_1, h_2, h_3, h_4, h_5)
   $$
   If height is too small (less than 0.01), it defaults to 1.0.

5. **Derived parameters** are calculated:
   $$
   central\_radius = \frac{inner\_radius + outer\_radius}{2}
   $$
   $$
   thickness = outer\_radius - inner\_radius
   $$

# Distance evaluation

To evaluate distance from the fitted model, the following approach is used:

Let $\vec{c} = (c_x, c_y, c_z)^T$ be the center point, $\vec{a} = (a_x, a_y, a_z)^T$ be the normalized axis vector, $r_{in}$ be inner radius, $r_{out}$ be outer radius, and $h$ be height of the ring.

For a point $\vec{p} = (x, y, z)^T$, the distance is calculated as:

1. **Vector from center to point**:
   $$
   \vec{v} = \vec{p} - \vec{c}
   $$

2. **Projection onto axis**:
   $$
   t = \vec{v} \cdot \vec{a}
   $$

3. **Perpendicular component** (distance to axis):
   $$
   \vec{v}_{\perp} = \vec{v} - t\vec{a}
   $$
   $$
   d_{axis} = \|\vec{v}_{\perp}\|
   $$

4. **Distance calculation**:

   If $d_{axis} < r_{in}$ (point inside inner cylinder):
   $$
   d = r_{in} - d_{axis}
   $$

   If $d_{axis} > r_{out}$ (point outside outer cylinder):
   $$
   d = d_{axis} - r_{out}
   $$

   If $r_{in} \leq d_{axis} \leq r_{out}$ (point between cylinders):
   $$
   \text{Let } h_{half} = \frac{h}{2}
   $$
   $$
   d = \begin{cases}
   |t| - h_{half}, & \text{if } |t| > h_{half}\\
   0, & \text{otherwise}
   \end{cases}
   $$

The resulting $d$ is the signed distance to the cylindrical ring surface:
- $d = 0$: point lies exactly on the surface (either on cylindrical walls or on top/bottom rims)
- $d < 0$: point is inside the ring volume
- $d > 0$: point is outside the ring

For multiple points, `calc_distances` returns an array of these distances computed for each point using `calc_distance_one_point`.