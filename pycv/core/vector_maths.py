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


def rotation_matrix_to_axes(r: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Given a 3x3 rotation matrix that defines the transformation from the camera's coordinate frame (where it square_points
        in the direction (0, 0, 1)), to the world's coordinate frame. Returns a tuple of numpy arrays,
        corresponding to the x, y and z axes in the world coordinate frame.

    :param r: the 3x3 rotation matrix
    :type r: np.ndarray
    :return: (x_axis, up, z_axis) -
    :rtype: np.ndarray
    """
    # create a direction vector for each axis, then transform using the rotation matrix
    x_axis = np.matmul(r, np.array([1, 0, 0]))
    y_axis = np.matmul(r, np.array([0, 1, 0]))
    z_axis = np.matmul(r, np.array([0, 0, 1]))
    return x_axis, y_axis, z_axis


def euler_angles_to_rotation_matrix(euler_angles: NDArray, degrees: bool = True):
    r = Rotation.from_euler('xyz', euler_angles, degrees=degrees)
    return r.as_matrix()


def rotation_matrix_to_euler_angles(r: NDArray, degrees: bool = True):
    r = Rotation.from_matrix(r)
    return r.as_euler('xyz', degrees=degrees)


def lookpos_to_rotation_matrix(pos: NDArray, look_pos: NDArray, y_axis: NDArray):
    """
    Creates a 3x3 rotation matrix from lookpos

    :param pos: the position of the camera in world coordinates. Shape (3)
    :type pos: np.ndarray
    :param look_pos: a points in world coordinates the camera is looking at. Shape (3)
    :rtype lookpos: np.ndarray
    :param y_axis: the world direction vector corresponding to the y axis ('up') in the camera's frame of reference.
        Shape (3)
    :type y_axis: np.ndarray
    :return: the created 3x3 rotation matrix
    :rtype: NDArray
    """
    r = np.zeros((3, 3))
    # the z axis in the camera's local coordinates defines the plane going out from the optical centre of the
    # camera to a points where the camera is looking at
    z_prime = (look_pos - pos) / np.linalg.norm(look_pos - pos)
    # calculate the y axis closest to the one specified
    y_prime = calc_closest_y_direction(z_prime, y_axis)
    y_prime /= np.linalg.norm(y_prime)
    # the x axis is then calculated as the cross product between the y and z axes
    x_prime = np.cross(y_prime, z_prime)
    x_prime /= np.linalg.norm(x_prime)
    # the rotation matrix can then be constructed by the the axes
    r[:, 0] = x_prime
    r[:, 1] = y_prime
    r[:, 2] = z_prime
    return r


def rotation_matrix_to_lookpos(pos: NDArray, r: NDArray):
    """

    :param pos:
    :param r:
    :return:
    """
    look_dir = np.matmul(r, np.array([0, 0, 1]))
    lookpos = pos + look_dir
    return lookpos


