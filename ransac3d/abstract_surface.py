import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from copy import deepcopy

class AbstractSurfaceModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.k = None
        self._model:dict = {}

    @abstractmethod
    def fit_model(
            self,
            points:NDArray
            ):
        assert len(points) == self.k

    @abstractmethod
    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        pass

    @abstractmethod
    def calc_distance_one_point(
            self, 
            point:NDArray
            ):
        pass

    def __repr__(self):
        result = ''
        for key, value in self._model.items():
            result += key + ' : ' + value + '\n'
        return result

    def get_model(self):
        return deepcopy(self._model)
