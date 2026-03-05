# Mobius strip
This class represents the Mobius strip model: a surface of constant width formed by uniformly rotating a segment (section) along a circular guide (central circle). Parameters: circle radius, normal, width, twist orientation, and initial section vector.

# Model parameters
## center
This parameter represents center of the central circle of the Mobius strip.
## radius
This parameter represents radius of the central circle of the Mobius strip.
## normal
This parameter represents normal to the plane of the central circle of the Mobius strip. The projection of the normal onto the z-axis is co-directed with the z-axis.
## width
This parameter represents the thickness of the Mobius strip.
## orientation
This parameter represents the direction of the strip's twist as it traverses the guide circle. A value of 1 corresponds to a counterclockwise (right-handed) twist, while a value of -1 corresponds to a clockwise (left-handed) twist, when viewed in the same direction as the velocity vector along the curve.
## start_vector
This parameter represents unit radius vector from the center of the central circle, which specifies the initial direction of the section of the strip, which lies in the plane of the circle.

# Model fitting
Let $X$ be the matrix that is passed to `fit_model`.
This matrix' dimensions are $4 \times 3$.

The first three points are used to construct a guide circle, determining its center, radius, and normal. The closest point on this circle is found for the fourth point. The vector from the closest point to the fourth defines the section direction, and twice the distance between them is taken as the width of the strip. The angle between this vector and the circle's normal is called the twist_angle. Then, the unit vector from the center to the closest point is calculated; rotating it in the plane of the circle by an angle of 2 (π/2 − twist_angle) orientation (for orientation = 1) yields the start_vector, which defines the section's orientation at the starting point.

# Distance evaluation
To calculate the distance from an arbitrary point to a Mobius strip, the point's coordinates are converted to the local coordinate system of the strip: the origin is placed at the center of the directrix circle, the X-axis is directed along the start_vector, the Z-axis is directed along the normal to the plane of the circle, and the Y-axis complements these to form a right-hand orthogonal triple. The strip is then approximated by a set of 100 plane sections—segments obtained by intersecting the strip with radial planes passing through the Z-axis and uniformly distributed over an angle from 0 to 2π. For each such segment, the distance to the point is calculated, and the final result is the minimum of these distances.