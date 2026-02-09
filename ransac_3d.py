import numpy as np
from numpy.typing import NDArray
from abstract_surface import SurfaceModel
from copy import deepcopy


class RANSAC3D:
    def __init__(self) -> None:
        self.__data:NDArray = np.zeros((0, 3), dtype=float)

    def load_data_from_file(
            self,
            path_to_file:str
            ):
        raise NotImplementedError

    def add_points(
            self,
            points:NDArray):
        self.__data = np.concatenate(
                (self.__data, points),
                axis=0
        )

    def fit(
            self, 
            object_type:type,
            iter_num:int,
            score_threshold:int
        ):
        self.__score_threshold = score_threshold
        self.model:SurfaceModel = object_type()
        
        best_model:SurfaceModel|None = None
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
            distances:NDArray
            ) -> float:
        scores = (distances <= self.__score_threshold)
        return np.sum(scores)

    def __sample(
            self
        ) -> NDArray:
        indices = np.random.choice(
                self.__data.shape[0], 
                size=self.model.k,
                replace=False
        )
        return self.__data[indices]

