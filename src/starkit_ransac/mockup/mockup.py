from abc import ABC, abstractmethod
import abc
import numpy as np
from numpy.typing import NDArray

class MockupModel(ABC):

    def __init__(
            self,
            model_parameters:dict,
            seed=42,
            max_noise:float=0.5
        ) -> None:
        super().__init__()
        self.SEED = seed
        self.max_noise = max_noise
        self._model_parameters:dict = model_parameters

    @abstractmethod
    def generate_data(self, n_points:int) -> NDArray:
        raise NotImplementedError

