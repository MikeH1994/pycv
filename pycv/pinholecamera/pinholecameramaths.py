import numpy as np
import numbers
from typing import Tuple
from numpy.typing import NDArray
import scipy.spatial
from pycv.core.vector_maths import calc_closest_y_direction
import cv2
from pycv.core import stack
from scipy.spatial.transform import Rotation

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


def create_camera_matrix(fx: float, fy: float, cx: float, cy: float):
    """

    :param cx:
    :param cy:
    :param fx:
    :param fy:
    :return:
    """

    camera_matrix = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
    return camera_matrix


def create_distortion_coeffs(k1, k2, k3, p1, p2, k4=0.0, k5=0.0, k6=0.0, mode="standard"):
    """

    :param k1:
    :param k2:
    :param k3:
    :param p1:
    :param p2:
    :param k4:
    :param k5:
    :param k6:
    :param mode:
    :return:
    """
    if k4 != 0.0 or k5 != 0.0 or k6 != 0.0:
        return np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float32)
    else:
        return np.array([k1, k2, p1, p2, k3], dtype=np.float32)

def project_points_to_2d(points: NDArray, camera_pos, camera_rotation, camera_matrix) -> NDArray:
    """
    Deproject a point in 3D space on to the 2D image_safe_zone plane, and calculate the coordinates of it


    :param points: the points in 3D space. Any size can be supplied, as long as the last dimension has 3 channels
    :type points: np.ndarray
    :param camera_rotation:
    :param camera_matrix
    :param camera_pos:
    :return: an array of shape (2) containing the x and y coordinates of the 3D point deprojected on to the
        image_safe_zone plane
    :rtype: np.ndarray
    """
    x_axis, y_axis, z_axis = rotation_matrix_to_axes(camera_rotation)
    fx, fy, cx, cy = unpack_camera_matrix(camera_matrix)


    init_shape = points.shape
    points = points.reshape((-1, 3))

    # calculate the direction vector from the camera to the defined points
    direction_vector = (points - camera_pos)

    # convert this vector to local coordinate space by doing dot product of
    # direction vector and each axis
    x_prime = np.sum(direction_vector*x_axis, axis=-1)
    y_prime = np.sum(direction_vector*y_axis, axis=-1)
    z_prime = np.sum(direction_vector*z_axis, axis=-1)

    # deproject on to image plane
    u = x_prime / z_prime * fx + cx
    v = y_prime / z_prime * fy + cy

    # stack into numpy array
    result = np.zeros((x_prime.shape[0], 2))
    result[:, 0] = u
    result[:, 1] = v

    # returned shape is the same as initial shape, but final dimension is 2 instead of 3
    result = result.reshape((*init_shape[:-1], 2))

    return result

def deproject_to_3d_vector(points, r: NDArray, camera_matrix, normalise=True) -> NDArray:
    """

    :param points:
    :param r:
    :param camera_matrix:
    :param normalise:
    :return:
    """

    if isinstance(points, tuple):
        points = stack(points)

    fx, fy, cx, cy = unpack_camera_matrix(camera_matrix)

    init_shape = points.shape
    points = points.reshape(-1, 2)
    n = points.shape[0]
    u = points[:, 0]
    v = points[:, 1]

    # calculate the direction vector of the rays in local coordinates

    vec = np.zeros((n, 3))
    vec[:, 0] = (u - cx) / fx
    vec[:, 1] = (v - cy) / fy
    vec[:, 2] = 1.0

    # calculate the direction vector in world coordinates
    r = scipy.spatial.transform.Rotation.from_matrix(r)
    vec = r.apply(vec)
    if normalise:
        vec /= np.linalg.norm(vec, axis=-1).reshape(-1, 1)

    # reshape to match input size
    vec = vec.reshape((*init_shape[:-1], 3))
    return vec


def find_camera_pose_from_pnp(camera_matrix: NDArray, object_points: NDArray, image_points: NDArray,
                              distortion_coeffs: NDArray = np.zeros(5)):
    retval, rvec, tvec = cv2.solvePnP(object_points.astype(np.float64), image_points.astype(np.float64),
                                      camera_matrix, distortion_coeffs)
    # to go from rvec and tvec to camera rotation matrix and position in world coordinates, see description in
    # https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation_matrix = np.linalg.inv(rotation_matrix)
    pos = -np.matmul(rotation_matrix, tvec.reshape(3))
    return pos, rotation_matrix

def find_object_pose_from_pnp(camera_matrix: NDArray, object_points: NDArray, image_points: NDArray,
                              distortion_coeffs: NDArray = np.zeros(5)):
    pos_cam, rotation_matrix_cam = find_camera_pose_from_pnp(camera_matrix, object_points, image_points, distortion_coeffs)
     # Invert to get object wrt camera:
    rotation_matrix = np.linalg.inv(rotation_matrix_cam)
    pos = -rotation_matrix @ pos_cam
    return pos, rotation_matrix

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


def unpack_camera_matrix(camera_matrix: NDArray):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return fx, fy, cx, cy

def distortion_coefficients_to_dict(distortion_coefficients):
    distortion_coefficients = distortion_coefficients.reshape(-1)
    if np.size(distortion_coefficients) == 5:
        k1, k2, p1, p2, k3 = distortion_coefficients
        k4 = k5 = k6 = 0.0
    elif np.size(distortion_coefficients) == 8:
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion_coefficients
    else:
        raise Exception(f"Unknown distortion coefficient type- shape {distortion_coefficients.shape}")
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "k6": k6,
        "p1": p1,
        "p2": p2
    }

def distort_points(points, camera_matrix, distortion_coeffs):
    assert(len(distortion_coeffs) == 5 or len(distortion_coeffs) == 8)
    init_shape = points.shape
    points = points.reshape((-1, 2))
    points_distorted = np.zeros(points.shape, dtype=np.float64)

    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]

    if len(distortion_coeffs) == 5:
        k1, k2, p1, p2, k3 = distortion_coeffs.reshape(-1)
        k4 = k5 = k6 = 0
    else:
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion_coeffs.reshape(-1)

    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3

    # Compute radial distortion
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)

    # Compute tangential distortion
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    points_distorted[:, 0] = x_distorted * fx + cx
    points_distorted[:, 1] = y_distorted * fy + cy
    points_distorted = points_distorted.reshape(init_shape)

    return points_distorted


def undistort_points(distorted_points, camera_matrix, distortion_coeffs,
                           max_iter=20, eps=1e-12):
    """
    Exact inverse of distort_points() using iterative Newton-style refinement.
    Supports 5 or 8 distortion coefficients (Brown–Conrady model).
    Returns pixel coordinates (same convention as distort_points).
    """
    init_shape = distorted_points.shape
    distorted_points = distorted_points.reshape(-1, 2).astype(np.float64)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # unpack distortion coefficients
    if len(distortion_coeffs) == 5:
        k1, k2, p1, p2, k3 = distortion_coeffs.reshape(-1)
        k4 = k5 = k6 = 0.0
    else:
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion_coeffs.reshape(-1)

    # Convert distorted pixels → normalized distorted coords
    xd = (distorted_points[:, 0] - cx) / fx
    yd = (distorted_points[:, 1] - cy) / fy

    # Initial guess: assume no distortion
    x = xd.copy()
    y = yd.copy()

    for _ in range(max_iter):

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)

        x_tan = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tan = p1*(r2 + 2*y*y) + 2*p2*x*y

        # forward-distorted estimate of (x,y)
        x_est = x * radial + x_tan
        y_est = y * radial + y_tan

        # update using difference between predicted and actual distorted
        dx = xd - x_est
        dy = yd - y_est

        x += dx
        y += dy

        if np.max(np.abs(dx)) < eps and np.max(np.abs(dy)) < eps:
            break

    # convert normalized → pixel units
    undistorted = np.zeros_like(distorted_points)
    undistorted[:, 0] = x * fx + cx
    undistorted[:, 1] = y * fy + cy
    return undistorted.reshape(init_shape)



def invert_distortion_maps(map_u, map_v, n_iterations=100, decay=0.9):
    """

    :param map_u: 2D array. The value at each pixel coordinate defines corresponding u coordinate in distorted space
    :param map_v: 2D array. The value at each pixel coordinate defines corresponding v coordinate in distorted space
    :return:
    """
    assert(map_u.shape == map_v.shape)
    F = np.zeros((map_u.shape[0], map_u.shape[1], 2), dtype=np.float32)
    F[:, :, 0] = map_u
    F[:, :, 1] = map_v

    (h, w) = F.shape[:2]  # (h, w, 2), "xymap"
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
    P = np.copy(I)
    k = 1
    for i in range(n_iterations):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * k
        k *= decay
    inverted_map_u, inverted_map_v = P[:, :, 0], P[:, :, 1]
    return inverted_map_u, inverted_map_v