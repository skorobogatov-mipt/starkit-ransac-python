import numpy as np
from numpy.typing import NDArray

def midpoint(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return (p1 + p2) / 2

def line_from_2_points(p1, p2):
    """
        Creates a line from 2 points in the form of:
        y = kx + b
        Returns:
            - k - line slope 
            - b - line offset
    """
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - p2[0] * k
    return k, b

def normal_to_2d_line(k, b, p):
    """
        Creates a normal to a line at a given 2d point.
        Returns:
            - k - line slope 
            - b - line offset
    """
    k = -1/k
    b = p[1] - k * p[0]

    return k, b

def lines_intersection(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x,y

def rotate_rodrigues(points, axis, theta):
    # using rodrigue's formula
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    c_theta = np.cos(theta)
    points = np.asarray(points)
    axis = axis / np.linalg.norm(axis)
    rotated_points = points * c_theta + \
            (np.cross(axis, points) * np.sin(theta)) + \
            np.outer(np.dot(points, axis), axis) * (1 - c_theta)
    return rotated_points

def rotate_from_axis_to_axis(points, ax0, ax1):
    ax0 = ax0/np.linalg.norm(ax0)
    ax1 = ax1/np.linalg.norm(ax1)
    rotation_axis = np.cross(ax0, ax1)
    theta = np.arccos(np.dot(ax0, ax1))
    return rotate_rodrigues(points, rotation_axis, theta)

def normalize(array, axis=None, keepdims=True) -> NDArray:
    return array / np.linalg.norm(array, axis=axis, keepdims=keepdims)
