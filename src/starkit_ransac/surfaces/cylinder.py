import numpy as np
from ransac3d.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray

class Cylinder(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = {
            "point_on_axis": np.nan,
            "axis_direction": np.nan,
            "radius": np.nan,
            "h_min": -np.inf,
            "h_max": np.inf
        }
        self.num_samples = 2

    def fit_model(
            self,
            points: NDArray
            ):
        assert len(points) == self.num_samples
        
        p1, p2 = points[0], points[1]
        
        # Estimate normals (using neighboring points or passed separately)
        # This requires normals to be provided - in this implementation we assume
        # normals are passed as an additional parameter or precomputed
        # For now, we'll raise an error if normals aren't available
        if not hasattr(self, 'normals') or self.normals is None:
            raise ValueError("Normals must be provided for cylinder fitting")
        
        n1, n2 = self.normals[0], self.normals[1]
        
        # Axis direction from cross product of normals
        axis = np.cross(n1, n2)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-6:
            raise ValueError("Normals are parallel, cannot determine cylinder axis")
        axis = axis / norm_axis
        
        # Project points and normals to plane perpendicular to axis
        def project_to_plane(point, normal, axis):
            proj_axis = np.dot(point, axis) * axis
            point_plane = point - proj_axis
            
            normal_plane = normal - np.dot(normal, axis) * axis
            norm_np = np.linalg.norm(normal_plane)
            if norm_np < 1e-6:
                return None, None
            normal_plane = normal_plane / norm_np
            return point_plane, normal_plane
        
        p1p, n1p = project_to_plane(p1, n1, axis)
        p2p, n2p = project_to_plane(p2, n2, axis)
        
        if p1p is None or p2p is None:
            raise ValueError("Failed to project points to plane")
        
        # Create orthonormal basis in the plane
        u = np.array([1, 0, 0])
        if abs(np.dot(u, axis)) > 0.99:
            u = np.array([0, 1, 0])
        u = u - np.dot(u, axis) * axis
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        
        # Convert to 2D coordinates
        p1_2d = np.array([np.dot(p1p, u), np.dot(p1p, v)])
        n1_2d = np.array([np.dot(n1p, u), np.dot(n1p, v)])
        p2_2d = np.array([np.dot(p2p, u), np.dot(p2p, v)])
        n2_2d = np.array([np.dot(n2p, u), np.dot(n2p, v)])
        
        # Solve for circle center in 2D
        A = np.array([n1_2d, -n2_2d]).T
        b = p2_2d - p1_2d
        
        try:
            t = np.linalg.lstsq(A, b, rcond=None)[0]
            t1, t2 = t[0], t[1]
        except:
            raise ValueError("Failed to solve for circle center")
        
        center1_2d = p1_2d + t1 * n1_2d
        center2_2d = p2_2d + t2 * n2_2d
        
        if np.linalg.norm(center1_2d - center2_2d) > 1e-3:
            raise ValueError("Inconsistent circle centers")
        
        center_2d = center1_2d
        radius = np.linalg.norm(p1_2d - center_2d)
        
        if radius < 1e-3:
            raise ValueError("Radius too small")
        
        # Convert back to 3D
        t_axis = (np.dot(p1, axis) + np.dot(p2, axis)) / 2.0
        point_on_axis = center_2d[0] * u + center_2d[1] * v + t_axis * axis
        
        self.model['point_on_axis'] = point_on_axis
        self.model['axis_direction'] = axis
        self.model['radius'] = radius
        self.model['h_min'] = -np.inf
        self.model['h_max'] = np.inf

    def set_normals(self, normals: NDArray):
        """Set normals for the points used in fitting"""
        self.normals = normals

    def set_height_from_points(
            self,
            points: NDArray,
            margin: float = 0.0
            ):
        """Set cylinder height limits based on point projections"""
        v = points - self.model['point_on_axis']
        h = np.dot(v, self.model['axis_direction'])
        self.model['h_min'] = h.min() - margin
        self.model['h_max'] = h.max() + margin

    def calc_distances(
            self,
            points: NDArray
            ) -> NDArray:
        """Calculate distances from points to cylinder surface"""
        points = np.asarray(points)
        v = points - self.model['point_on_axis']
        
        # Project onto axis
        proj = np.dot(v, self.model['axis_direction'])[:, np.newaxis] * self.model['axis_direction']
        perp = v - proj
        dist_to_axis = np.linalg.norm(perp, axis=1)
        h = np.dot(v, self.model['axis_direction'])
        
        # Distance to side surface
        dist_side = np.abs(dist_to_axis - self.model['radius'])
        
        # Initialize distances array
        dist = dist_side.copy()
        
        # Handle points below bottom
        below = h < self.model['h_min']
        if np.any(below):
            dist_bottom = np.sqrt(
                (self.model['h_min'] - h[below])**2 + 
                dist_to_axis[below]**2
            )
            dist[below] = np.minimum(dist[below], dist_bottom)
        
        # Handle points above top
        above = h > self.model['h_max']
        if np.any(above):
            dist_top = np.sqrt(
                (h[above] - self.model['h_max'])**2 + 
                dist_to_axis[above]**2
            )
            dist[above] = np.minimum(dist[above], dist_top)
        
        return dist

    def calc_distance_one_point(
            self,
            point: NDArray
            ) -> float:
        """Calculate distance from a single point to cylinder surface"""
        return self.calc_distances(point.reshape(1, -1))[0]

    def get_cylinder_params(self) -> dict:
        """Return cylinder parameters"""
        return {
            'point_on_axis': self.model['point_on_axis'].copy(),
            'axis_direction': self.model['axis_direction'].copy(),
            'radius': self.model['radius'],
            'h_min': self.model['h_min'],
            'h_max': self.model['h_max']
        }