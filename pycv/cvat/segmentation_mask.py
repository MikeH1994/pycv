from typing import List, Dict, Tuple, Callable
import subprocess
import os
import shutil
import itertools
import cv2
import numpy as np
import glob
from tqdm.auto import tqdm
import mytorch.core as core
import mytorch.segmentation as segmentation

def generate_dst_fpath(src_fpath, dst_root, name_replacer: Callable = None) -> str:
    """

    :param src_fpath:
    :param dst_root:
    :param name_replacer:
    :return:
    """
    fname = os.path.basename(src_fpath)
    if name_replacer is not None:
        fname = name_replacer(fname)
    return os.path.join(dst_root, fname)


class CVATSegmentationMaskWriter:
    object_colors = [ele for ele in itertools.product([0, 64, 128, 192], repeat=3)]
    object_colors.remove((0, 0, 0))
    class_colors = [core.get_colour(i) for i in range(64)]
    class_folder_path = ""
    object_folder_path = ""
    imagesets_folder_fpath = ""
    jpeg_folder_path = ""
    label_map_fpath = ""

    def __init__(self):
        pass

    def run(self, dst_root: str, image_fpaths: List[str], mask_fpaths: List[str],
            classes: Dict[int, str], image_name_replacer: Callable = None,
            mask_name_replacer: Callable = None, open_image_fn: Callable = None):
        """

        :param dst_root:
        :param image_fpaths:
        :param mask_fpaths:
        :param classes:
        :param image_name_replacer: a 2-tuple indicating a replacement to make on the image filenames when copying
        :param mask_name_replacer: a 2-tuple indicating a replacement to make on the mask filenames when copying
        :param open_image_fn: a function that defines how to convert an image in to the correct form to write- for
            example, if the src images are stored as tiffs or numpy arrays. The function should take a single argument,
            the filepath, and return a uint8 array that can be written as a .jpg or .png. Note that the
            image_name_replacer should be set to change the extension to .jpg
        """
        if not os.path.isabs(dst_root):
            dst_root = os.path.abspath(dst_root)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        core.verify_classes(classes)

        for val in classes.values():
            if list(classes.values()).count(val)>1:
                raise Exception("Multiple instances of {} found in ")

        self.class_folder_path = os.path.join(dst_root, "SegmentationClass")
        self.object_folder_path =  os.path.join(dst_root, "SegmentationObject")
        self.imagesets_folder_fpath = os.path.join(dst_root, "ImageSets", "Segmentation")
        self.jpeg_folder_path = os.path.join(dst_root, "JPEGImages")
        self.label_map_fpath = os.path.join(dst_root, "labelmap.txt")

        image_fpaths = sorted(image_fpaths)
        mask_fpaths = sorted(mask_fpaths)

        self.write_labelmap(classes)
        self.write_image_sets(image_fpaths, image_name_replacer)
        self.write_jpeg_images(image_fpaths, open_image_fn, image_name_replacer)
        self.write_segmentation_class(mask_fpaths, classes, mask_name_replacer)
        self.write_segmentation_object(mask_fpaths, classes, mask_name_replacer)

        class_folder_path = os.path.relpath(self.class_folder_path, dst_root)
        object_folder_path = os.path.relpath(self.object_folder_path, dst_root)
        imagesets_folder_fpath = os.path.relpath(self.imagesets_folder_fpath, dst_root)
        jpeg_folder_path = os.path.relpath(self.jpeg_folder_path, dst_root)
        label_map_fpath = os.path.relpath(self.label_map_fpath, dst_root)

        zip_filename = "output.zip"
        current_dir = os.path.abspath(os.getcwd())

        os.chdir(dst_root)
        subprocess.run(["zip", "-r", zip_filename, class_folder_path, object_folder_path,
                        imagesets_folder_fpath, jpeg_folder_path, label_map_fpath])
        os.chdir(current_dir)

    def write_labelmap(self, classes: Dict[int, str]):
        """

        :param fpath:
        :param classes:
        :return:
        """
        with open(self.label_map_fpath, 'w') as f:
            f.write("# label:color_rgb:parts:actions")
            for class_id in classes:
                class_name = classes[class_id]
                class_colour = self.class_colors[class_id]
                f.write("\n{}:{},{},{}::".format(class_name, class_colour[0], class_colour[1], class_colour[2]))
            # file should finish with a blank line
            f.write("\n")

    def write_segmentation_class(self, mask_fpaths: List[str], classes: Dict[int, str], name_replacer: Callable = None):
        """

        :param mask_fpaths:
        :param name_replacer:
        :return:
        """
        if not os.path.exists(self.class_folder_path):
            os.makedirs(self.class_folder_path)

        classes = {cls_idx: self.class_colors[cls_idx] for cls_idx in classes}

        progress_bar = tqdm(mask_fpaths, disable=False, dynamic_ncols=True)
        progress_bar.set_description("segmentation class")

        for src_mask_fpath in progress_bar:
            mask_class = cv2.imread(src_mask_fpath, -1)
            mask_colour = segmentation.convert_class_mask_to_color_mask(mask_class, classes, flip_channels=True)
            dst_mask_fpath = generate_dst_fpath(src_mask_fpath, self.class_folder_path, name_replacer)
            cv2.imwrite(dst_mask_fpath, mask_colour)

    def write_segmentation_object(self, mask_fpaths: List[str],  classes: Dict[int, str],
                                  name_replacer: Callable = None):
        """

        :param mask_fpaths:
        :param classes:
        :param name_replacer:
        :return:
        """

        if not os.path.exists(self.object_folder_path):
            os.makedirs(self.object_folder_path)

        progress_bar = tqdm(mask_fpaths, disable=False, dynamic_ncols=True)
        progress_bar.set_description("segmentation object")

        for src_mask_fpath in progress_bar:
            mask_class = cv2.imread(src_mask_fpath, -1)
            mask_class = cv2.cvtColor(mask_class, cv2.COLOR_BGR2RGB)
            mask_object = np.zeros(mask_class.shape, dtype=np.uint8)
            for class_id in classes:
                if class_id == 0:
                    # if color is background
                    continue
                mask_object[np.where(np.all(mask_class == class_id, axis=-1))] = CVATSegmentationMaskWriter.object_colors[class_id]

            dst_mask_fpath = generate_dst_fpath(src_mask_fpath, self.object_folder_path, name_replacer)
            cv2.imwrite(dst_mask_fpath, mask_object)

    def write_jpeg_images(self, image_fpaths: List[str], open_image_fn: Callable = None,
                          name_replacer: Callable = None):
        """

        :param image_fpaths:
        :param name_replacer:
        :param open_image_fn:
        :return:
        """
        if not os.path.exists(self.jpeg_folder_path):
            os.makedirs(self.jpeg_folder_path)

        progress_bar = tqdm(image_fpaths, disable=False, dynamic_ncols=True)
        progress_bar.set_description("jpeg images")

        for src_image_fpath in progress_bar:
            dst_image_fpath = generate_dst_fpath(src_image_fpath, self.jpeg_folder_path, name_replacer)
            assert (dst_image_fpath.endswith(".png") or dst_image_fpath.endswith(".jpg")), "dst image fpath should" \
                "end with either .jpg or .png extension. Set the name_replacer variable"
            if open_image_fn is not None:
                img = open_image_fn(src_image_fpath)
                cv2.imwrite(dst_image_fpath, img)
            else:
                shutil.copy(src_image_fpath, dst_image_fpath)

    def write_image_sets(self, image_fpaths: List[str], name_replacer: Callable = None):
        """

        :param image_fpaths:
        :param name_replacer:
        :return:
        """
        if not os.path.exists(self.imagesets_folder_fpath):
            os.makedirs(self.imagesets_folder_fpath)

        with open(os.path.join(self.imagesets_folder_fpath, "default.txt"), 'w') as f:
            for src_image_fpath in image_fpaths:
                image_fname = os.path.basename(src_image_fpath)
                if name_replacer is not None:
                    image_fname = name_replacer(image_fname)
                image_fname = os.path.splitext(image_fname)[0]
                f.write("{}\n".format(image_fname))

    @staticmethod
    def copy_annotated_masks(src_root, dst_root, image_extension=".CSV", src_mask_extension=".png",
                             dst_mask_extension="_mask_roi.png"):
        """
        :param src_root: the location of the new masks, in the raw form outputted by CVAT
        :param dst_root: the location of the unprocessed dataset (v1)
        :param image_extension: the file extension of the raw image file (".CSV")
        :param src_mask_extension: the file extension of the mask (".png")
        :param dst_mask_extension:

        :return:
        """
        src_masks_fpath = sorted(glob.glob(os.path.join(src_root, '**/*{}'.format(src_mask_extension)), recursive=True))
        src_masks_fname = [os.path.basename(f) for f in src_masks_fpath]
        dst_masks_fpath = sorted(glob.glob(os.path.join(dst_root, '**/*{}'.format(image_extension)), recursive=True))
        dst_masks_fpath = [f.replace(image_extension, dst_mask_extension) for f in dst_masks_fpath]
        dst_masks_fname = [os.path.basename(f) for f in dst_masks_fpath]

        for i, src_mask_fpath in enumerate(src_masks_fpath):
            src_mask_fname = src_masks_fname[i].replace(src_mask_extension, dst_mask_extension)
            if src_mask_fname not in dst_masks_fname:
                continue
            dst_mask_fpath = dst_masks_fpath[dst_masks_fname.index(src_mask_fname)]
            if not os.path.exists(os.path.dirname(dst_mask_fpath)):
                os.makedirs(os.path.dirname(dst_mask_fpath))
            shutil.copy(src_mask_fpath, dst_mask_fpath)
            print("Copied ", dst_mask_fpath)


class CVATSegmentationCopier:
    def __init__(self, src_root, dst_root, classes: Dict[int, str]):
        self.src_root = src_root
        self.classes = classes
        self.dst_root = dst_root
        self.class_colours = self.load_class_colours(os.path.join(src_root, "labelmap.txt"))
        self.jpeg_images = sorted(glob.glob(os.path.join(src_root, "JPEGImages", "*")))
        self.class_masks = sorted(glob.glob(os.path.join(src_root, "SegmentationClass", "*")))
        self.object_masks = sorted(glob.glob(os.path.join(src_root, "SegmentationObject", "*")))
        self.initialise()

    def run(self, copy_images=False, copy_class_masks=False, copy_object_masks=False):
        if copy_images:
            self.copy_images(self.jpeg_images)
        if copy_class_masks:
            self.copy_class_masks(self.class_masks)
        if copy_object_masks:
            self.copy_object_masks(self.object_masks)

    def load_class_colours(self, label_fpath) -> Dict[str, Tuple]:
        if os.path.exists(label_fpath) is False:
            raise Exception("{} not found".format(label_fpath))
        class_colours = {}
        class_names = core.flip_classes(self.classes)
        with open(label_fpath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                class_name = line.split(":")[0]
                if class_name not in class_names:
                    raise Exception("Class {} found in labelmap was not supplied classes".format(class_name))
                rgb= tuple([int (v) for v in line.split(":")[1].split(",")])
                class_colours[class_name] = rgb
        return class_colours

    def copy_images(self, src_images: List[str]):
        progress_bar = tqdm(src_images, disable=False, dynamic_ncols=True)
        progress_bar.set_description("copying jpegs...")
        for src_fpath in progress_bar:
            dst_fpath = self.get_image_dst_fpath(src_fpath)
            shutil.copy(src_fpath, dst_fpath)

    def copy_class_masks(self, src_images: List[str]):
        class_index_to_color: Dict[int, Tuple[int, int, int]] = {}
        for class_idx in self.classes:
            class_name = self.classes[class_idx]
            color = self.class_colours[class_name]
            class_index_to_color[class_idx] = color

        progress_bar = tqdm(src_images, disable=False, dynamic_ncols=True)
        progress_bar.set_description("copying class masks...")

        for src_fpath in progress_bar:
            dst_fpath = self.get_class_mask_fpath(src_fpath)
            mask_color = cv2.imread(src_fpath, -1)
            mask_class = segmentation.convert_color_mask_to_class_mask(mask_color, class_index_to_color)
            cv2.imwrite(dst_fpath, mask_class)

    def copy_object_masks(self, src_images: List[str]):
        progress_bar = tqdm(src_images, disable=False, dynamic_ncols=True)
        progress_bar.set_description("copying object masks...")

        for src_fpath in progress_bar:
            dst_fpath = self.get_object_mask_fpath(src_fpath)
            raise Exception("copy_object_masks not yet implemented")

    def initialise(self):
        raise Exception("initialise() not implemented")

    def get_image_dst_fpath(self, src_fpath):
        raise Exception("get_image_dst_fpath() not implemented")

    def get_class_mask_fpath(self, src_fpath):
        raise Exception("get_class_mask_fpath() not implemented")

    def get_object_mask_fpath(self, src_fpath):
        raise Exception("get_object_mask_fpath() not implemented")


class OrderedCVATSegmentationCopier(CVATSegmentationCopier):
    raw_dst_images = []
    raw_dst_image_fnames = []
    raw_image_extension = ".jpg"
    mask_extension = ".png"

    def __init__(self, src_root, dst_root, class_names: Dict[int, str], raw_image_extension =".jpg", mask_extension=".png"):
        self.mask_extension = mask_extension
        self.raw_image_extension = raw_image_extension
        super().__init__(src_root, dst_root, class_names)


    def initialise(self):
        self.raw_dst_images = sorted(glob.glob(os.path.join(self.dst_root, "**/*{}".format(self.raw_image_extension)), recursive=True))
        self.raw_dst_image_fnames = [os.path.basename(f).replace(self.raw_image_extension, "") for f in self.raw_dst_images]

    def get_class_mask_fpath(self, src_fpath):
        src_fname = os.path.basename(src_fpath).replace(".png", "")
        if src_fname not in self.raw_dst_image_fnames:
            raise Exception("Could not find {} in dst_fname list".format(src_fname))
        index = self.raw_dst_image_fnames.index(src_fname)
        return self.raw_dst_images[index].replace(self.raw_image_extension, self.mask_extension)