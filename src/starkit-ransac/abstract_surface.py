import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from copy import deepcopy


class AbstractSurfaceModel(ABC):
    """
        Abstract base class for surface models.
        
        This class defines the interface for fitting geometric surface models
        to point data and calculating distances from points to the fitted surface.
        
        Subclasses must implement the abstract methods to provide specific
        surface fitting and distance calculation logic.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._model_data:dict = {}
        self._num_samples:int = -1

    @property
    def num_samples(self) -> int:
        """
            Get the number of samples required to fit the model.
            
            Returns
            -------
            int
                The number of points required to fit the surface model.
                Must be overridden by subclasses.
        """
        return self._num_samples

    @num_samples.setter
    def num_samples(self, n:int):
        self._num_samples = n

    @property
    def model(self) -> dict:
        """
            Get the internal model parameters.
            
            Returns
            -------
            dict
                A dictionary containing the model parameters.
                Is an empty dict by default, must be overridden by subclasses.
        """
        return self._model_data

    @model.setter
    def model(self, model:dict):
        self._model_data = deepcopy(model)

    @abstractmethod
    def fit_model(
            self,
            points: NDArray
            ):
        """
            Fit the surface model to a set of points.
            
            Parameters
            ----------
            points : NDArray
                Array of points to fit the model to. The length must equal `num_samples`.
                
            Notes
            -----
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError
        assert len(points) == self.num_samples

    @abstractmethod
    def calc_distances(
            self,
            points: NDArray
            ) -> NDArray:
        """
            Calculate distances from multiple points to the fitted surface.
            
            Parameters
            ----------
            points : NDArray
                Array of points for which to calculate distances.
                
            Returns
            -------
            NDArray
                Array of distances from each point to the surface.
        """
        raise NotImplementedError

    @abstractmethod
    def calc_distance_one_point(
            self, 
            point: NDArray
            ) -> float:
        """Calculate the distance from a single point to the fitted surface.
        
        Parameters
        ----------
        point : NDArray
            A single point for which to calculate the distance.
            
        Returns
        -------
        float
            The distance from the point to the surface.
            
        """
        raise NotImplementedError

    def __repr__(self):
        result = ''
        for key, value in self.model.items():
            result += key + ' : ' + str(value) + '\n'
        return result

    def get_model(self):
        """
            Return model dict.
            
            Returns
            -------
            model:dict
        """
        return deepcopy(self._model_data)
