import numpy as np
from numpy.typing import NDArray
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from copy import deepcopy


class RANSAC3D:
    """RANSAC algorithm implementation for 3D surface fitting.
    
    This class implements the Random Sample Consensus (RANSAC) algorithm
    for robust fitting of geometric surface models to 3D point cloud data.
    """
    
    def __init__(self) -> None:
        """Initialize the RANSAC3D object with an empty point cloud.
        
        Returns
        -------
        None
        """
        self.__data: NDArray = np.zeros((0, 3), dtype=float)

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
            Array of 3D points to add, shape (N, 3) where N is the number of points.
            
        Returns
        -------
        None
        """
        self.__data = np.concatenate(
                (self.__data, points),
                axis=0
        )

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

        for _ in range(iter_num):
            sample = self.__sample()
            self.model.fit_model(sample)
            distances = self.model.calc_distances(self.__data)
            score = self.__score_from_distances(distances)
            
            if score > best_model_score:
                best_model = deepcopy(self.model)
                best_model_score = score
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

        scores = (distances <= self.__distance_threshold)
        return np.sum(scores)

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
