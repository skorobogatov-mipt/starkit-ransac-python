import pytest
import numpy as np
import open3d as o3d
from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.cylindrical_ring import CylindricalRing
import math

N_POINTS = 1000
SEED = 42


def create_ring_mesh(center_x, center_y, center_z, inner_r, outer_r, height, resolution=30):
    points = []
    lines = []

    angles = np.linspace(0, 2 * math.pi, resolution)

    # Внешний цилиндр
    for i, angle in enumerate(angles):
        x = center_x + outer_r * math.cos(angle)
        y = center_y + outer_r * math.sin(angle)

        # Верхняя точка
        points.append([x, y, center_z + height / 2])
        # Нижняя точка
        points.append([x, y, center_z - height / 2])

    # Внутренний цилиндр
    offset = len(points)
    for i, angle in enumerate(angles):
        x = center_x + inner_r * math.cos(angle)
        y = center_y + inner_r * math.sin(angle)

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


@pytest.fixture
def cylindrical_ring_data():
    np.random.seed(SEED)

    inner_r = 2.0
    outer_r = 3.0
    height = 4.0
    center = (0, 0, 0)

    points = []
    for _ in range(N_POINTS):
        radius = np.random.uniform(inner_r, outer_r)
        angle = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height / 2, height / 2)

        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        points.append([x, y, z])

    return np.array(points)


def visualize_ring(points, model, title="Цилиндрическое кольцо"):

    # Облако точек
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

    # Создаем сетку кольца
    ring_mesh = create_ring_mesh(center_x, center_y, center_z, inner_r, outer_r, height)

    # Система координат
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    print(f"  Центр: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"  Внутренний радиус: {inner_r:.3f}")
    print(f"  Внешний радиус: {outer_r:.3f}")
    print(f"  Высота: {height:.3f}")
    print(f"  Толщина стенки: {result['thickness']:.3f}")
    print(f"  Центральный радиус: {result['central_radius']:.3f}")

    # Визуализация
    o3d.visualization.draw_geometries(
        [pcd, ring_mesh, coord_frame],
        window_name=title,
        width=1024,
        height=768
    )


def test_cylindrical_ring_center(cylindrical_ring_data):


    ransac = RANSAC3D()
    ransac.add_points(cylindrical_ring_data)

    model = ransac.fit(
        CylindricalRing,
        1000,
        0.5
    )

    visualize_ring(cylindrical_ring_data, model, "Кольцо в центре координат")

    result = model.get_model()
    assert abs(result['center_x']) < 0.3
    assert abs(result['center_y']) < 0.3
    assert abs(result['center_z']) < 0.3
    assert abs(result['inner_radius'] - 2.0) < 0.3
    assert abs(result['outer_radius'] - 3.0) < 0.3
    assert abs(result['height'] - 4.0) < 0.99


def test_cylindrical_ring_fit_model():
    ring = CylindricalRing()

    # Две точки для вертикального кольца
    points = np.array([
        [2.5, 0, -2],
        [2.5, 0, 2]
    ])

    ring.fit_model(points)
    result = ring.get_model()

    assert result['center_x'] == 2.5
    assert result['center_y'] == 0.0
    assert result['center_z'] == 0.0
    assert result['central_radius'] == pytest.approx(2.5, rel=0.1)
    assert result['height'] == pytest.approx(4.8, rel=0.1)  # 4.0 * 1.2


def test_cylindrical_ring_distances():
    ring = CylindricalRing()

    ring.model = {
        'center_x': 0.0,
        'center_y': 0.0,
        'center_z': 0.0,
        'axis_x': 0.0,
        'axis_y': 0.0,
        'axis_z': 1.0,
        'inner_radius': 2.0,
        'outer_radius': 3.0,
        'height': 4.0,
        'central_radius': 2.5,
        'thickness': 1.0
    }

    # Точка на внешней поверхности
    dist = ring.calc_distance_one_point(np.array([3.0, 0, 0]))
    assert dist == pytest.approx(0, abs=0.01)

    # Точка внутри стенки
    dist = ring.calc_distance_one_point(np.array([2.5, 0, 0]))
    assert dist == 0

    # Точка снаружи
    dist = ring.calc_distance_one_point(np.array([3.5, 0, 0]))
    assert dist > 0

    # Точка выше кольца
    dist = ring.calc_distance_one_point(np.array([2.5, 0, 2.5]))
    assert dist > 0