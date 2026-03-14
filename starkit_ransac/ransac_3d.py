import open3d as o3d
import pdb
from starkit_ransac.generators.ellipsoid import generate_ellipsoid
from time import sleep
import numpy as np
from numpy.typing import NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from copy import deepcopy

from starkit_ransac.visualisation.visualize import generate_mesh

class RANSAC:
    """RANSAC algorithm implementation for 3D surface fitting.
    
    This class implements the Random Sample Consensus (RANSAC) algorithm
    for robust fitting of geometric surface models to 3D point cloud data.
    """
    
    def __init__(
            self,
            data:NDArray|None=None
        ) -> None:
        """Initialize the RANSAC3D object with an empty point cloud.
        
        Returns
        -------
        None
        """
        if data is None:
            self.__data: NDArray = np.zeros((0, 3), dtype=float)
        else:
            self.__data = np.copy(data)

    def load_data_from_file(
            self,
            path_to_file: str
            ):
        """Load point cloud data from a file.
        
        Parameters
        ----------
        path_to_file : str
            Path to the file containing point cloud data.
        """
        raise NotImplementedError

    def add_points(
            self,
            points: NDArray):
        """Add points to the internal point cloud data.
        
        Parameters
        ----------
        points : NDArray
            Array of 3D points to add, shape (N, 3) where N is the number of
            points.
            
        Returns
        -------
        None
        """
        if len(self.__data) == 0:
            self.__data = np.copy(points)
        else:
            self.__data = np.concatenate((self.__data, points))

    def fit(
            self, 
            object_type: type,
            iter_num: int,
            distance_threshold: float
        ):
        """Fit a surface model to the point cloud using RANSAC.
        
        Performs iterative random sampling and model fitting to find
        the best surface model that maximizes the number of inliers.
        
        Parameters
        ----------
        object_type : type
            The class type of the surface model to fit. Must be a subclass
            of AbstractSurfaceModel.
        iter_num : int
            Number of RANSAC iterations to perform.
        distance_threshold : float
            Maximum distance for a point to be considered an inlier.
            
        Returns
        -------
        best_model : AbstractSurfaceModel
            The fitted surface model with the highest inlier count.
        """
        self.__distance_threshold = distance_threshold
        self.model: AbstractSurfaceModel = object_type()
        
        best_model: AbstractSurfaceModel = None
        best_model_score = -1

        # best_model_mesh = generate_mesh(self.model, color=(1,0,0))
        #
        # viz = o3d.visualization.Visualizer()
        # viz.create_window()
        #
        # ctr = viz.get_view_control()
        #
        # opt = viz.get_render_option()
        # opt.background_color = np.array([0.2, 0.2, 0.2])
        # viz.get_render_option().line_width = 10.
        # viz.get_render_option().point_size = 1
        # # print(getattr(opt, 'line_width'))
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.__data)
        # pcd.paint_uniform_color([0.9, 0.9, 0.9])
        # viz.add_geometry(pcd)
        # viz.add_geometry(best_model_mesh)
        # hypothesis_mesh = generate_mesh(self.model)
        # viz.add_geometry(hypothesis_mesh)
        #
        # viz.poll_events()
        # viz.update_renderer()
        # sleep(5)

        i = 0
        for _ in range(iter_num):
            # i += 1
            # for j in range(10):
            #     ctr.rotate(.5, 0.0)
            #     sleep(0.001)
            # viz.get_render_option().line_width = 10.
            # viz.get_render_option().point_size = 1

            sample = self.__sample()
            success = self.model.fit_model(sample)
            if not success:
                continue

            # if i%5 == 0:
            #     viz.remove_geometry(hypothesis_mesh, False)
            #     hypothesis_mesh = generate_mesh(self.model, color=(1,0,0))
            #     viz.add_geometry(hypothesis_mesh, False)

            distances = self.model.calc_distances(self.__data)
            score = self.__score_from_distances(distances)
            
            # viz.poll_events()
            if score > best_model_score:
                # viz.remove_geometry(best_model_mesh, False)
                best_model = deepcopy(self.model)
                best_model_score = score

            #     best_model_mesh = generate_mesh(best_model)
            #     viz.add_geometry(best_model_mesh, False)
            # viz.update_renderer()
        return best_model

    def __score_from_distances(
            self,
            distances: NDArray
            ) -> float:
        """
            Calculate the inlier score from point-to-surface distances.
            
            Parameters
            ----------
            distances : NDArray
                Array of distances from each point to the fitted surface.
                
            Returns
            -------
            score : float
                The number of points within the distance threshold (inlier count).
        """

        return np.sum(distances <= self.__distance_threshold)

    def __sample(
            self
        ) -> NDArray:
        """Randomly sample points from the point cloud for model fitting.
        
        Selects a random subset of points equal to the number required
        by the current model (model.num_samples) without replacement.
        
        Returns
        -------
        sample : NDArray
            Array of randomly sampled points, shape (num_samples, 3).
        """
        indices = np.random.choice(
                self.__data.shape[0], 
                size=self.model.num_samples,
                replace=False
        )
        return self.__data[indices]

