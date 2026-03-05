import open3d as o3d

import numpy as np
def create_ring_mesh(inner_r, outer_r, center, height, resolution=30):
    points = []
    lines = []
    center_x, center_y, center_z = center

    angles = np.linspace(0, 2 * np.pi, resolution)

    # Внешний цилиндр
    for i, angle in enumerate(angles):
        x = center_x + outer_r * np.cos(angle)
        y = center_y + outer_r * np.sin(angle)

        # Верхняя точка
        points.append([x, y, center_z + height / 2])
        # Нижняя точка
        points.append([x, y, center_z - height / 2])

    # Внутренний цилиндр
    offset = len(points)
    for i, angle in enumerate(angles):
        x = center_x + inner_r * np.cos(angle)
        y = center_y + inner_r * np.sin(angle)

        points.append([x, y, center_z + height / 2])
        points.append([x, y, center_z - height / 2])

    # Вертикальные линии внешнего цилиндра
    for i in range(resolution):
        lines.append([i * 2, i * 2 + 1])

    # Вертикальные линии внутреннего цилиндра
    for i in range(resolution):
        lines.append([offset + i * 2, offset + i * 2 + 1])

    # Окружности
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Внешний цилиндр
        lines.append([i * 2, next_i * 2])  # верх
        lines.append([i * 2 + 1, next_i * 2 + 1])  # низ
        # Внутренний цилиндр
        lines.append([offset + i * 2, offset + next_i * 2])  # верх
        lines.append([offset + i * 2 + 1, offset + next_i * 2 + 1])  # низ

    # Торцы
    for i in range(resolution):
        lines.append([i * 2, offset + i * 2])  # верхний торец
        lines.append([i * 2 + 1, offset + i * 2 + 1])  # нижний торец

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Красный

    return line_set

def visualize_ring(points, model, title="Cylindrical Ring"):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Серый

    result = model.get_model()
    center_x = result['center_x']
    center_y = result['center_y']
    center_z = result['center_z']
    inner_r = result['inner_radius']
    outer_r = result['outer_radius']
    height = result['height']

    ring_mesh = create_ring_mesh(inner_r, outer_r, (center_x, center_y, center_z), height)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    print(f"  center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"  inner radius: {inner_r:.3f}")
    print(f"  outer radius: {outer_r:.3f}")
    print(f"  height: {height:.3f}")
    print(f"  thickness: {result['thickness']:.3f}")
    print(f"  central radius: {result['central_radius']:.3f}")

    # Визуализация
    o3d.visualization.draw_geometries(
        [pcd, ring_mesh, coord_frame],
        window_name=title,
        width=1024,
        height=768
    )
