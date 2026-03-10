import numpy as np

def generate_mobius(
        mobius_strip, 
        noise_sigma=0.1, 
        n_points=10, 
        n_uniform_ponts=0, 
        box_center=np.array([0, 0, 0]), 
        box_size=np.array([10, 10, 10])
    ):
    half_width = mobius_strip.width/2
    center = mobius_strip.center
    normal = mobius_strip.normal
    radius = mobius_strip.radius
    zero_angle_vector = mobius_strip.start_vector
    orientation = mobius_strip.orientation

    v1 = np.cross(normal, zero_angle_vector)
    v1 /= np.linalg.norm(v1)
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    values = np.random.uniform(0, 1, size=n_points)
    mobius_points = np.zeros((n_points, 3))
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=mobius_points.shape)
    for i in range(n_points):
        vector_in_circle_plane = np.cos(angles[i]) * zero_angle_vector + np.sin(angles[i]) * v1
        p = radius * vector_in_circle_plane
        half_width_vector = half_width*(vector_in_circle_plane * 
                                        np.cos(angles[i]*orientation/2) + normal*np.sin(angles[i]*orientation/2))
        one_end = p+half_width_vector
        another_end = p-half_width_vector
        mobius_points[i, :] = another_end * values[i] + one_end * (1-values[i]) + center

    if n_uniform_ponts >0:
        points_uniform = box_center + np.random.uniform(
        low=-box_size / 2,
        high= box_size / 2,
        size=(n_uniform_ponts, 3)
        )
        return np.vstack([mobius_points+noise, points_uniform])
    return mobius_points+noise
