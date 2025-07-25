import open3d as o3d
import numpy as np
import pycv
from pycv.pinholecamera import PinholeCamera
from pycv.core import stack_coords, unstack_coords
from numpy.typing import NDArray
from typing import List

class OverlayedMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.mesh = mesh
        self.vertex_intensity = np.copy(self.mesh.vertex_colors).astype(np.float32)

    def n_vertices(self):
        return np.asarray(self.mesh.vertices).shape[0]

    def n_triangles(self):
        return np.asarray(self.mesh.triangles).shape[0]

    def overlay_images(self, images: List[NDArray], cameras: List[PinholeCamera], mode="average"):
        """

        :param images:
        :param cameras:
        :return:
        """
        assert(mode=="average" or mode=="last")
        n_channels = pycv.n_channels(images[0])
        n_vertices = self.n_vertices()
        vertices = np.asarray(self.mesh.vertices)
        assert (all([pycv.n_channels(images[i]) == n_channels for i in range(len(images))]))
        assert(n_channels == 1 or n_channels == 3)
        average_intensity_shape = n_vertices if n_channels == 1 else (n_vertices, n_channels)
        average_intensity = np.zeros(average_intensity_shape, dtype=np.float32)
        average_intensity_n = np.zeros(n_vertices, dtype=np.int32)

        for i, image in enumerate(images):
            camera = cameras[i]
            height, width = image.shape[:2]
            vertices_visible = self.points_are_unoccluded(vertices, camera)

            interpolated_image = pycv.imageutils.InterpolatedImage(image)
            px_coords = camera.project_points_to_2d(vertices)
            x, y = px_coords[:, 0], px_coords[:, 1]
            valid_indices = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1) & vertices_visible
            valid_vertex_vals = interpolated_image.f(x[valid_indices], y[valid_indices])

            if mode == "average":
                average_intensity[valid_indices] += valid_vertex_vals
                average_intensity_n[valid_indices] += 1
            else:
                average_intensity[valid_indices] = valid_vertex_vals
                average_intensity_n[valid_indices] = 1
        average_intensity[average_intensity_n > 0] /= average_intensity_n[average_intensity_n > 0]
        min_val = np.min(average_intensity[average_intensity_n > 0])
        max_val = np.max(average_intensity[average_intensity_n > 0])
        print(min_val, max_val)
        vertex_colors = average_intensity
        vertex_colors = (vertex_colors - min_val)/(max_val-min_val)
        if n_channels == 1:
            vertex_colors_ = np.zeros((n_vertices, 3), dtype=np.float32)
            vertex_colors_[:] = vertex_colors.reshape(-1, 1)
            vertex_colors = vertex_colors_
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        self.vertex_intensity = average_intensity

    def faces_are_visible(self, camera: PinholeCamera):
        faces = self.mesh.triangles.numpy()
        vertices_visible = self.points_are_unoccluded(np.asarray(self.mesh.vertices), camera)
        faces_visible_sum = vertices_visible[faces[:, 0]] + vertices_visible[faces[:, 1]] + vertices_visible[
            faces[:, 2]]
        faces_visible_sum = np.sum(faces_visible_sum, axis=1)
        faces_visible = faces_visible_sum == 3
        return faces_visible

    def reset_colour(self):
        vertex_colors = np.zeros((self.n_vertices(), 3), dtype=np.float32)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


    def colour_by_visibility(self, camera, visible_colour=(0,255,0), occluded_colour=(0,0,0)):
        occluded_colour = np.array([occluded_colour])/255.0
        visible_colour = np.array([visible_colour])/255.0
        vertex_colors = np.full((self.n_vertices(), 3), fill_value=occluded_colour, dtype=np.float32)
        points_are_unoccluded = self.points_are_unoccluded(np.asarray(self.mesh.vertices), camera)
        vertex_colors[points_are_unoccluded] = visible_colour
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        return vertex_colors

    def create_mask(self, camera: PinholeCamera):
        mask = np.zeros((camera.yres, camera.xres), dtype=np.uint8)
        coords = camera.project_points_to_2d(np.asarray(self.mesh.vertices), return_as_int=True)
        u, v = unstack_coords(coords)
        valid_indices = (v >= 0) & (v < camera.yres - 1) & (u >= 0) & (u < camera.xres - 1)
        mask[v[valid_indices]][u[valid_indices]] = 1
        return mask

    def points_are_unoccluded(self, points: NDArray, camera: PinholeCamera) -> NDArray:
        assert (len(points.shape) == 2 and points.shape[1] == 3)
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))

        rays = np.zeros((points.shape[0], 6), dtype=np.float32)
        dirn = camera.p - points
        p = points + 0.005 * dirn
        rays[:, :3] = p
        rays[:, 3:] = dirn
        ans = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

        # if the rays have not hit anything, this means that the point is not occluded,
        # and we can therefore see it
        is_visible = ans['primitive_ids'].numpy() == raycasting_scene.INVALID_ID
        return is_visible

    @staticmethod
    def read_mesh(fpath: str):
        mesh = o3d.io.read_triangle_mesh(fpath)
        return OverlayedMesh(mesh)