import math
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def rotate_vector_around_axis(vector: NDArray, axis: NDArray, angle: float):
    """
    Rotate a vector around another vector using Rodrigues' rotation formula.

    Parameters:
        vector (numpy.ndarray): The vector to be rotated.
        axis (numpy.ndarray): The axis around which the rotation will be performed.
        angle (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotated vector.
    """
    # see https://medium.com/@sim30217/rodrigues-rotation-formula-47489db49050
    # Ensure input vectors are numpy arrays
    vector = vector.astype(np.float32)
    axis = axis.astype(np.float32)
    # Normalize axis vector
    axis /= np.linalg.norm(axis)
    # Rodrigues' rotation formula
    rotated_vector = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) \
                     + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    return rotated_vector / np.linalg.norm(rotated_vector)


def create_perpendicular_vector(vec):
    """

    :param vec:
    :return:
    """
    # first we want to check that vec is not a zero vector of size (3,)
    assert (np.linalg.norm(vec) > 1e-8)
    assert (vec.shape == (3,))
    # we then normalise it so that it is a unit vector
    vec = np.copy(vec)
    vec /= np.linalg.norm(vec)
    # the cross product of two vectors is perpeendicular to both.
    # we can take the cross product of vec with one of the unit vectors to
    # create a vector perpendicular to vec
    possible_axis_choices = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    # choose the axis that is furthest away from parallel from vec
    dot_products = [abs(np.dot(vec, axis)) for axis in possible_axis_choices]
    axis = possible_axis_choices[dot_products.index(min(dot_products))]
    # compute the cross product
    return np.cross(vec, axis)


def calc_closest_y_direction(z_dirn: NDArray, preferred_y_direction: NDArray) -> NDArray:
    """

    :param z_dirn:
    :param preferred_y_direction:
    :return:
    """
    init_y_dirn = create_perpendicular_vector(z_dirn)

    def minimisation_fn(theta):
        y_vec = rotate_vector_around_axis(np.copy(init_y_dirn), np.copy(z_dirn), theta)
        l = abs(1.0 - np.dot(y_vec, preferred_y_direction))
        return l

    x0 = np.array([0.0])
    bounds = [(-2 * np.pi, 2 * np.pi)]
    result = minimize(minimisation_fn, x0, bounds=bounds, method="Nelder-Mead")  # method='L-BFGS-B')
    theta = result.x
    calc_y_dirn = rotate_vector_around_axis(init_y_dirn, z_dirn, theta)
    return calc_y_dirn


def euler_angles_to_rotation_matrix(euler_angles: NDArray, degrees: bool = True):
    r = Rotation.from_euler('xyz', euler_angles, degrees=degrees)
    return r.as_matrix()


def rotation_matrix_to_euler_angles(r: NDArray, degrees: bool = True):
    r = Rotation.from_matrix(r)
    return r.as_euler('xyz', degrees=degrees)


