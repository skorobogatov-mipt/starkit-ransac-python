import open3d as o3d
import numpy as np

def generate_mobius_mesh(
        mobius_strip, 
        color=np.array([1, 0, 0]), 
        n_theta=1000, 
        n_width=100,
    ):
    half_width = mobius_strip.model['width']/2
    center = mobius_strip.model['center']
    normal = mobius_strip.model['normal']
    radius = mobius_strip.model['radius']
    orientation = mobius_strip.model['orientation']
    zero_angle_vector = mobius_strip.model['start_vector']
    v1 = np.cross(normal, zero_angle_vector)
    v1 /= np.linalg.norm(v1)
    angles = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    i=0
    mobius_points = np.zeros((n_theta, n_width+1, 3))
    for theta in angles:
        vector_in_circle_plane = np.cos(theta) * zero_angle_vector + np.sin(theta) * v1
        p = radius * vector_in_circle_plane
        half_width_vector = half_width*(vector_in_circle_plane * np.cos(theta*orientation/2) + normal*np.sin(theta*orientation/2))
        one_end = p+half_width_vector
        another_end = p-half_width_vector
        for j in range(n_width+1):
            mobius_points[i, j, :] = another_end * j/n_width + one_end * (n_width - j)/n_width + center
        i+=1
    
    rows, cols, _ = mobius_points.shape
    points = mobius_points.reshape(-1, 3)
    lines = []
    def idx(i, j):
        return i * cols + j
    # 2. Соединения внутри строк
    for i in range(rows):
        for j in range(cols - 1):
            lines.append([idx(i, j), idx(i, j + 1)])
    # 3. Соединения между строками
    for i in range(rows - 1):
        for j in range(cols):
            lines.append([idx(i, j), idx(i + 1, j)])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)
    return line_set
