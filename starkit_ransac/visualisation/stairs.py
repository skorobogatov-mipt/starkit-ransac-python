import open3d as o3d
import numpy as np
from starkit_ransac.surfaces.stairs import StepPlane

def generate_stairs_mesh(
        stairs:StepPlane,
        color=(0,1,0),
        resolution=None
    ):
    planes = []
    thickness = 0.01

    num_steps = int(np.ceil(stairs.stair_height / stairs.step_height))
    for i in range(num_steps):
    # --- 1. ГОРИЗОНТАЛЬНАЯ СТУПЕНЬ ---
    # Размеры: x=глубина шага, y=ширина марша, z=толщина
        step = o3d.geometry.TriangleMesh.create_box(
            width=stairs.step_width, 
            height=stairs.stair_height, 
            depth=thickness
        )
        # Сдвигаем: x вперед, z вверх (на i-й уровень)
        step.translate([i * stairs.step_width, 0, i * stairs.step_height])
        step.paint_uniform_color([0.0, 1.0, 0.0]) # Светло-серый
        R = step.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(stairs.rotation_deg))
        )
        step.rotate(R, center=(0, 0, 0))
        planes.append(step)

        # --- 2. ВЕРТИКАЛЬНЫЙ ПОДСТУПЕНОК ---
        # Создаем его в плоскости XY (как стенку), а потом двинем
        # Размеры: x=толщина, y=ширина марша, z=высота шага
        riser = o3d.geometry.TriangleMesh.create_box(
            width=thickness, 
            height=stairs.stair_height, 
            depth=stairs.step_height
        )
        
        if(i == num_steps - 1): 
            continue

        # Сдвигаем: ставим его ПЕРЕД ступенью
        # x - в начало текущей ступени, z - под текущую ступень
        riser.translate([
            (i + 1) * stairs.step_width, 
            0, 
            (i) * stairs.step_height
        ])
        
        # Чтобы закрыть самый первый шаг (от пола), добавим условие или просто рисуем всегда
        riser.paint_uniform_color([0.0, 1.0, 0.0]) # Темно-серый
        R = riser.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(stairs.rotation_deg))
        )
        riser.rotate(R, center=(0, 0, 0))
        planes.append(riser)
    for plane in planes[1:]:
        planes[0] += plane


    # Добавляем оси координат
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # planes.append(mesh_frame)

    return planes[0]

def visualize_stairs(points, stairs:StepPlane):
    stairs_meshes = generate_stairs_mesh(stairs)

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points)
    pcd_all.paint_uniform_color([0.6, 0.6, 0.6])

    stairs_meshes.append(pcd_all)
    o3d.visualization.draw_geometries(stairs_meshes)

