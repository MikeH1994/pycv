import numpy as np
import open3d as o3d
import pycv
from pycv.pinholecamera import PinholeCamera

from numpy.typing import NDArray
from typing import List


def overlay_images_on_mesh(mesh: o3d.geometry.TriangleMesh, images: List[NDArray], cameras: List[PinholeCamera],
                           mode="average"):
    """

    :param mesh:
    :param images:
    :param cameras:
    :return:
    """
    n_channels = pycv.n_channels(images[0])

    assert(all([pycv.n_channels(images[i]) == n_channels for i in range(len(images))]))

    average_intensity_shape = (mesh.vertices.shape[0]) if n_channels == 1 else (mesh.vertices.shape[0], n_channels)
    average_intensity = np.zeros(average_intensity_shape, dtype=np.float32)
    average_intensity_n = np.zeros(mesh.vertices.shape[0], dtype=np.int32)

    for i, image in enumerate(images):
        height, width = image.shape[:2]
        interpolated_image = pycv.imageutils.InterpolatedImage(image)
        camera = cameras[i]
        px_coords = camera.project_points_to_image_plane(mesh.vertices)
        x, y = px_coords[:, 0], px_coords[:, 1]
        valid_indices = (x >= 0 & x <= width-1) & (y >= 0 & y <= height-1)
        valid_vertex_vals = interpolated_image.f(x[valid_indices], y[valid_indices])

        if mode == "average":
            average_intensity[valid_indices] += valid_vertex_vals
            average_intensity_n[valid_indices] += 1
        else:
            average_intensity[valid_indices] = valid_vertex_vals
            average_intensity_n[valid_indices] = 1


def points_are_visible(points: NDArray, mesh: o3d.geometry.TriangleMesh, camera: PinholeCamera):
    pass