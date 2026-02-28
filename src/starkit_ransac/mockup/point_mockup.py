import numpy as np
from numpy.typing import NDArray
from starkit_ransac.mockup.mockup import MockupModel

class PointMockup(MockupModel):

    def generate_data(self, n_points) -> NDArray:
        np.random.seed(self.SEED)
        return np.random.random((n_points, 3))

