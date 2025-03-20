import numpy as np
import numbers
from typing import Tuple
from numpy.typing import NDArray
import scipy.spatial
from .vector_maths import calc_closest_y_direction, rotation_matrix_to_axes, rotation_matrix_to_axes


def focal_length_to_fov(fx: float, xres: float) -> float:
    """
    Calculates the horizontal field of view, in degrees, from the focal length in the x direction
        and the x resolution. (Alternatively, passing the focal length in the y direction, with the y
        resolution provides the VFOV).

        (see https://github.com/opencv/opencv/blob/82f8176b0634c5d744d1a45246244291d895b2d1/modules/calib3d/src/calibration.cpp#L1778
             https://github.com/opencv/opencv/blob/e0b7f04fd240b3ea23ae0cc2e3c071c2e018a7ec/modules/calib3d/src/calibration.cpp#L3879)


    :param fx: the focal length, in pixels
    :type fx: float
    :param xres: the x resolution of the camera, in pixels (i.e. the width of the image_safe_zone)
    :type xres: float
    :param cx: the optical center of the camera
    :type cx: float
    :return: the hfov, in degrees
    :rtype: float
    """
    # np.degrees(np.arctan2(cx, fx) + np.arctan2(xres - cx, fx)).item()
    return np.degrees(2.0 * np.arctan(xres / 2.0 / fx)).item()


def fov_to_focal_length(hfov: float, xres: int) -> float:
    """
    Calculates the x focal length, in pixels, from the hfov and the x resolution of the camera.
        (Alternatively, the y focal length can be calculated by passing the vfov and the yres)

        (see https://github.com/opencv/opencv/blob/82f8176b0634c5d744d1a45246244291d895b2d1/modules/calib3d/src/calibration.cpp#L1778
             https://github.com/opencv/opencv/blob/e0b7f04fd240b3ea23ae0cc2e3c071c2e018a7ec/modules/calib3d/src/calibration.cpp#L3879)

    :param hfov: the horizontal field of view, in degrees
    :type hfov: float
    :param xres: the x resolution of the camera, in pixels (i.e. width of the image_safe_zone)
    :type xres: int
    :return: the focal length, in pixels
    :rtype: float
    """
    hfov = np.radians(hfov)
    return xres / 2.0 / np.tan(hfov / 2)


def hfov_to_vfov(hfov: float, xres: int, yres: int) -> float:
    """
    Calculates the vfov, from the corresponding hfov, the xres and yres

    :param hfov: the vertical field of view, in degrees
    :type hfov: float
    :param xres: the x resolution of the camera, in pixels (i.e. the width of the image_safe_zone)
    :type xres: int
    :param yres: the y resolution of the camera, in pixels (i.e. the height of the image_safe_zone)
    :type yres: int
    :return: the vfov, in degrees
    :rtype: float
    """
    hfov_radians = np.radians(hfov)
    vfov_radians = 2 * np.arctan(np.tan(hfov_radians / 2) * float(yres) / xres)
    return np.degrees(vfov_radians).item()


def create_camera_matrix(cx: float, cy: float, fx: float, fy: float = None):
    """

    :param cx:
    :param cy:
    :param fx:
    :param fy:
    :return:
    """

    if fy is None:
        fy = fx

    camera_matrix = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
    return camera_matrix


def get_pixel_point_lies_in(points: NDArray, camera_pos, r, res, fov, centre) -> NDArray:
    """
    Deproject a point in 3D space on to the 2D image_safe_zone plane, and calculate the coordinates of it

    :param points: a point in 3D space. Shape: (3)
    :type points: np.ndarray
    :return: an array of shape (2) containing the x and y coordinates of the 3D point deprojected on to the
        image_safe_zone plane
    :rtype: np.ndarray
    """
    x_axis, y_axis, z_axis = rotation_matrix_to_axes(r)
    xres, yres = res
    hfov, vfov = fov
    hfov = np.radians(hfov)
    vfov = np.radians(vfov)
    cx, cy = centre

    init_shape = points.shape
    points = points.reshape(-1, 3)

    # calculate the direction vector from the camera to the defined points
    direction_vector = (points - camera_pos)
    # convert this vector to local coordinate space by doing dot product of
    # direction vector and each axis
    x_prime = np.sum(direction_vector*x_axis, axis=-1)
    y_prime = np.sum(direction_vector*y_axis, axis=-1)
    z_prime = np.sum(direction_vector*z_axis, axis=-1)
    # deproject on to image plane
    k_x = 2 * z_prime * np.tan(hfov / 2.0)
    k_y = 2 * z_prime * np.tan(vfov / 2.0)
    u = (x_prime / k_x * xres + cx)
    v = (y_prime / k_y * yres + cy)
    #
    result = np.zeros((x_prime.shape[0], 2))
    result[:, 0] = u
    result[:, 1] = v
    # returned shape is the same as initial shape, but final dimension is 2 instead of 3
    result = result.reshape((*init_shape[:-1], 2))

    return result

def get_pixel_direction(p: NDArray, r: NDArray, res, fov, centre) -> NDArray:
    """
    Get the direction vector corresponding to the given pixel coordinates

    :param p: the pixel coordinates
    :return: an array of length 3, which corresponds to the direction vector in world space for the given
             pixel coordinates
    :rtype: np.ndarray
    """

    cx, cy = centre
    hfov, vfov = fov
    xres, yres = res

    init_shape = p.shape
    p = p.reshape(-1, 2)
    n = p.shape[0]

    u = p[:, 0]
    v = p[:, 1]

    # calculate the direction vector of the rays in local coordinates
    vz = 1

    vec = np.zeros((n, 3))
    vec[:, 0] = 2.0 * vz * (u - cx) / xres * np.tan(np.radians(hfov / 2.0))
    vec[:, 1] = 2.0 * vz * (v - cy) / yres * np.tan(np.radians(vfov / 2.0))
    vec[:, 2] = vz

    # calculate the direction vector in world coordinates
    r = scipy.spatial.transform.Rotation.from_matrix(r)
    vec = r.apply(vec)
    vec /= np.linalg.norm(vec, axis=-1).reshape(-1, 1)

    # reshape to match input size
    vec = vec.reshape((*init_shape[:-1], 3))
    return vec
