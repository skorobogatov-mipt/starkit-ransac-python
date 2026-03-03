import numpy as np
from numpy.typing import NDArray

from starkit_ransac.abstract_surface import AbstractSurfaceModel


def _estimate_period(values: NDArray) -> float:
    # Estimate repeating step from histogram autocorrelation.
    values = np.asarray(values, dtype=float)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    v_range = v_max - v_min

    if v_range <= 0:
        return 1.0
      
    n_bins = 256
    hist, edges = np.histogram(values, bins=n_bins, range=(v_min, v_max))
    signal = hist.astype(float) - np.mean(hist)
    ac = np.correlate(signal, signal, mode="full")[n_bins - 1 :]

    bin_w = edges[1] - edges[0]
    lags = np.arange(1, n_bins // 2)
    periods = lags * bin_w

    valid = (periods > (v_range / 40.0)) & (periods < (v_range / 2.0))
    
    if not np.any(valid):
        return max(v_range / 6.0, 1e-3)

    valid_lags = lags[valid]
    best_lag = int(valid_lags[np.argmax(ac[valid_lags])])
    return float(best_lag * bin_w)



class StepPlane(AbstractSurfaceModel):
    """Staircase model for RANSAC."""

    def __init__(self) -> None:
        super().__init__()
        self.model = {
            "stair_height": np.nan,
            "step_width": np.nan,
            "step_height": np.nan,
            "rotation_deg": np.nan,
        }
        self.num_samples = 90

    @staticmethod
    def generate_ladder_points(
        n_steps: int,
        step_height: float,
        step_width: float,
        stair_span: float,
        n_points: int,
        noise_sigma: float,
        rotation_deg: float,
    ) -> NDArray:
        # Generate treads + risers, rotate, add noise.
        theta = np.deg2rad(rotation_deg)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        points = []

        for i in range(n_steps):
            x = np.random.uniform(i * step_width, (i + 1) * step_width, n_points)
            y = np.random.uniform(0.0, stair_span, n_points)
            z = np.ones_like(x) * i * step_height
            points.append(np.column_stack((x, y, z)))

        for i in range(1, n_steps):
            x = np.ones(n_points) * i * step_width
            y = np.random.uniform(0.0, stair_span, n_points)
            z = np.random.uniform((i - 1) * step_height, i * step_height, n_points)
            points.append(np.column_stack((x, y, z)))

        cloud = np.vstack(points)
        cloud = cloud @ rot.T

        if noise_sigma > 0:
            cloud = cloud + np.random.normal(0.0, noise_sigma, cloud.shape)

        return cloud

    def fit_model(self, points: NDArray):
        # Estimate staircase params from one RANSAC sample.
        assert len(points) == self.num_samples

        points = np.asarray(points, dtype=float)
        xy = points[:, :2]
        z_local = points[:, 2]

        # Angle from centroid of low-Z points to centroid of high-Z points.
        z_min = float(np.min(z_local))
        z_max = float(np.max(z_local))
        z_range = max(z_max - z_min, 1e-6)
        band = 0.15 * z_range

        low_mask = z_local <= (z_min + band)
        high_mask = z_local >= (z_max - band)

        low_center = np.mean(xy[low_mask], axis=0)
        high_center = np.mean(xy[high_mask], axis=0)
        direction = high_center - low_center

        rotation_rad = float(np.arctan2(direction[1], direction[0]))
        rotation_deg = float(np.rad2deg(rotation_rad) % 180.0)

        theta = -rotation_rad
        rot2 = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        xy_local = xy @ rot2.T
        x_local = xy_local[:, 0]

        step_width = _estimate_period(x_local)
        step_height = _estimate_period(z_local)

        stair_height = float(np.max(z_local) - np.min(z_local))

        self.model["stair_height"] = stair_height
        self.model["step_width"] = float(step_width)
        self.model["step_height"] = float(step_height)
        self.model["rotation_deg"] = rotation_deg

    def calc_distances(self, points: NDArray) -> NDArray:
        # Distance to nearest tread or riser surface.
        points = np.asarray(points, dtype=float)

        step_width = self.model["step_width"]
        step_height = self.model["step_height"]
        rotation_deg = self.model["rotation_deg"]

        theta = -np.deg2rad(rotation_deg)
        rot2 = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        xy_local = points[:, :2] @ rot2.T
        x_local = xy_local[:, 0]
        z_local = points[:, 2]

        k_floor = np.floor((x_local) / step_width)
        z_tread = k_floor * step_height
        dist_tread = np.abs(z_local - z_tread)
        k_round = np.round((x_local) / step_width)
        x_riser =  k_round * step_width
        dist_riser = np.abs(x_local - x_riser)

        return dist_tread

    def calc_distance_one_point(self, point: NDArray) -> float:
        point = np.asarray(point, dtype=float).reshape(1, 3)
        return float(self.calc_distances(point)[0])
