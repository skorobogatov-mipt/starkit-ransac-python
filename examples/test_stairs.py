import os
import pytest
from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.stairs import StepPlane
import numpy as np
import open3d as o3d
import math

# Parameters of stairs
N_POINTS = 650
N_STEPS = 3
STEP_HEIGHT = 0.5
STEP_WIDTH = 0.6
STAIR_SPAN = 0.4
ROTATION_DEG = 20
NOISE_SIGMA = 0
SEED = 42


@pytest.fixture
def stairs_data():
    # The generated cloud indicates stairs with noise
    np.random.seed(SEED)
    return StepPlane.generate_ladder_points(
        n_steps=N_STEPS,
        step_height=STEP_HEIGHT,
        step_width=STEP_WIDTH,
        stair_span=STAIR_SPAN,
        n_points=N_POINTS,
        noise_sigma=NOISE_SIGMA,
        rotation_deg=ROTATION_DEG,
    )


@pytest.fixture
def acceptable_rmse():
    return 0.08


def test_stairs(
        stairs_data,
        acceptable_rmse,
        ):
    # Fitting model stairs RANSAC
    runsuck = RANSAC3D()
    runsuck.add_points(stairs_data)

    np.random.seed(SEED)
    model = runsuck.fit(
            StepPlane,
            1200,
            0.06
    )
    result = model.get_model()
    print("RESULT : ", result)
    distances = model.calc_distances(stairs_data)
    rmse = np.sqrt(np.mean(distances ** 2))

    assert np.isfinite(result['step_width']) and result['step_width'] > 0
    assert np.isfinite(result['step_height']) and result['step_height'] > 0
    assert np.isfinite(result['rotation_deg'])
    assert rmse < acceptable_rmse


@pytest.mark.skipif(
    os.environ.get("RUN_VISUAL_TESTS") != "1",
    reason="Set RUN_VISUAL_TESTS=1 to run interactive Open3D visualization.",
)
def test_stairs_visualization(stairs_data):
    runsuck = RANSAC3D()
    runsuck.add_points(stairs_data)

    np.random.seed(SEED)
    model = runsuck.fit(
            StepPlane,
            1200,
            0.01
    )

    distances = model.calc_distances(stairs_data)
    inlier_mask = distances <= 0.15

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(stairs_data)
    pcd_all.paint_uniform_color([0.6, 0.6, 0.6])


    result = model.get_model()
    print("RESULT : ", result)
    planes = []
    thickness = 0.01

    num_steps = math.ceil(result['stair_height'] / result['step_height'])
    for i in range(num_steps):
    # --- 1. ГОРИЗОНТАЛЬНАЯ СТУПЕНЬ ---
    # Размеры: x=глубина шага, y=ширина марша, z=толщина
        step = o3d.geometry.TriangleMesh.create_box(
            width=result['step_width'], 
            height=STAIR_SPAN, 
            depth=thickness
        )
        # Сдвигаем: x вперед, z вверх (на i-й уровень)
        step.translate([i * result['step_width'], 0, i * result['step_height']])
        step.paint_uniform_color([0.0, 1.0, 0.0]) # Светло-серый
        R = step.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(result['rotation_deg'])))
        step.rotate(R, center=(0, 0, 0))
        planes.append(step)

        # --- 2. ВЕРТИКАЛЬНЫЙ ПОДСТУПЕНОК ---
        # Создаем его в плоскости XY (как стенку), а потом двинем
        # Размеры: x=толщина, y=ширина марша, z=высота шага
        riser = o3d.geometry.TriangleMesh.create_box(
            width=thickness, 
            height=STAIR_SPAN, 
            depth=result['step_height']
        )
        
        if(i == num_steps - 1): 
            continue

        # Сдвигаем: ставим его ПЕРЕД ступенью
        # x - в начало текущей ступени, z - под текущую ступень
        riser.translate([(i + 1) * result['step_width'], 0, (i) * result['step_height']])
        
        # Чтобы закрыть самый первый шаг (от пола), добавим условие или просто рисуем всегда
        riser.paint_uniform_color([0.0, 1.0, 0.0]) # Темно-серый
        R = riser.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(result['rotation_deg'])))
        riser.rotate(R, center=(0, 0, 0))
        planes.append(riser)

    # Добавляем оси координат
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    planes.append(mesh_frame)

    # n_steps = int(round(result["stair_height"] / result["step_height"])) + 1
    # n_steps = max(2, n_steps)

    # stairs_from_model = StepPlane.generate_ladder_points(
    #     n_steps=n_steps,
    #     step_height=result["step_height"],
    #     step_width=result["step_width"],
    #     stair_span=STAIR_SPAN,          # ширину марша берёшь из своих входных
    #     n_points=N_POINTS,
    #     noise_sigma=0.0,                # без шума, чтобы видеть форму
    #     rotation_deg=result["rotation_deg"],
    # )

    # pcd_inliers = o3d.geometry.PointCloud()
    # pcd_inliers.points = o3d.utility.Vector3dVector(stairs_from_model)
    # pcd_inliers.paint_uniform_color([0.2, 0.8, 0.3])

    planes.append(pcd_all)
    o3d.visualization.draw_geometries(planes)
        
