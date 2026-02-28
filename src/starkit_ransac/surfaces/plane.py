import numpy as np
from starkit_ransac.abstract_surface import AbstractSurfaceModel
from numpy.typing import NDArray
from copy import deepcopy

class Plane3D(AbstractSurfaceModel):
    def __init__(self) -> None:
        super().__init__()
        self.num_samples = 3
        self.model = {
            'a' : np.nan,
            'b' : np.nan,
            'c' : np.nan,
            'd' : np.nan
        }

    def fit_model(
            self,
            points:NDArray
            ):
        assert len(points) == self.num_samples
        x1,y1,z1 = points[0][0],points[0][1],points[0][2]
        x2,y2,z2 = points[1][0],points[1][1],points[1][2]
        x3,y3,z3 = points[2][0],points[2][1],points[2][2]
        self.model['a'] = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
        self.model['b'] = (x3-x1)*(z2-z1)-(x2-x1)*(z3-z1)
        self.model['c'] = (y3-y1)*(x2-x1)-(y2-y1)*(x3-x1)
        norm = np.sqrt(self.model['a']**2+self.model['b']**2+self.model['c']**2)
        if norm==0:
            raise ValueError
        self.model['a'] /= norm
        self.model['b'] /= norm
        self.model['c'] /= norm
        self.model['d'] = -self.model['a']*x1-self.model['b']*y1-self.model['c']*z1
        

    def calc_distances(
            self,
            points:NDArray
            ) -> NDArray:
        a = self.model['a']
        b = self.model['b']
        c = self.model['c']
        d = self.model['d']
        distances = 0
        for i in points:
            if (a*a+b*b+c*c)==0:
                print(a,b,c,d)
                break
            distances+=abs(a*i[0]+b*i[1]+c*i[2]+d)/(a*a+b*b+c*c)**0.5
        return distances

    def calc_distance_one_point(self, point: NDArray):
        pass
