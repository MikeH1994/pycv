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
import xml.etree.ElementTree as ET
from ..keypoint_rcnn.utils import find_skeletons_corresponding_to_bbox

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

class CVATWriter:
    def __init__(self):
        pass

    def run(self, image_fpaths: List[str], dst_root: str, name_replacer: Callable = None,
            open_image_fn: Callable = None):
        """

        :param dst_root:
        :param image_fpaths:
        :param image_name_replacer: a 2-tuple indicating a replacement to make on the image filenames when copying
        :param open_image_fn: a function that defines how to convert an image in to the correct form to write- for
            example, if the src images are stored as tiffs or numpy arrays. The function should take a single argument,
            the filepath, and return a uint8 array that can be written as a .jpg or .png. Note that the
            image_name_replacer should be set to change the extension to .jpg
        """
        if not os.path.isabs(dst_root):
            dst_root = os.path.abspath(dst_root)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        self.write_images(image_fpaths, dst_root, name_replacer, open_image_fn)

    def write_images(self, image_fpaths: List[str], dst_root, name_replacer: Callable = None,
                     open_image_fn: Callable = None):
        progress_bar = tqdm(image_fpaths, disable=False, dynamic_ncols=True)
        progress_bar.set_description("images")

        for src_fpath in progress_bar:
            dst_fpath = generate_dst_fpath(src_fpath, dst_root, name_replacer)
            if open_image_fn is not None:
                img = open_image_fn(src_fpath)
                cv2.imwrite(dst_fpath, img)
            else:
                shutil.copy(src_fpath, dst_fpath)


class CVATCopier:
    def __init__(self, xml_filepath: str, dst_root: str):
        self.xml_filepath = xml_filepath
        self.dst_root = dst_root
        self.initialise()

    def initialise(self):
        raise Exception("initialise() not implemented")

    def get_annotations_fpath(self, fname):
        raise Exception("get_annotations_fpath() not implemented")

    def write_annotation(self, fname, image_data, label_ids: Dict[str, int] = None):
        pass

    def get_skeleton_from_xml(self, xml_elem) -> Tuple[List, str]:
        """
        Given an xml element corresponding to the skeleton in an image
        Returns a dictionary containing:
            points -  list of [x, y, visibility]
            label - the label

        A skeleton element is of the form
        <skeleton label="right_foot" source="manual" z_order="0">
          <points label="1" source="manual" outside="0" occluded="0" points="156.61,74.73">
          </points>
          <points label="2" source="manual" outside="0" occluded="0" points="134.77,70.36">
          </points>
          <points label="3" source="manual" outside="0" occluded="0" points="117.07,79.41">
          </points>
        </skeleton>

        :param xml_elem:
        :return:
        """
        assert (xml_elem.tag == "skeleton")

        label = xml_elem.attrib["label"]
        skeleton = [None for point_elem in xml_elem if "points" in point_elem.attrib]
        for point_elem in xml_elem:
            # each point element will be of the form
            # <points label="6" source="manual" outside="0" occluded="0" points="157.72,94.43">
            assert ("points" in point_elem.attrib)
            # in keypointrcnn format, points are defined by [x, y, visibility]
            point = [float(f) for f in point_elem.attrib["points"].split(",")]
            point.append(1 - int(point_elem.attrib["occluded"]))
            point_id = int(point_elem.attrib["label"])
            skeleton[point_id - 1] = point
        assert (None not in skeleton)
        return skeleton, label

    def get_bbox_from_xml(self, xml_elem) -> Tuple[List, str]:
        """

        <box label="right_foot_bbox" source="manual" occluded="0" xtl="79.30" ytl="57.70" xbr="173.00" ybr="284.00" z_order="0">
        </box>

        :param xml_elem:
        :param label_ids:
        :return:
        """
        assert(xml_elem.tag == "box")
        x0 = float(xml_elem.attrib["xtl"])
        x1 = float(xml_elem.attrib["xbr"])
        y0 = float(xml_elem.attrib["ytl"])
        y1 = float(xml_elem.attrib["ybr"])
        bbox = [x0, y0, x1, y1]
        label = xml_elem.attrib["label"]

        return bbox, label

    def get_point_from_xml(self, xml_elem):
        raise Exception("get_point_from_xml not implemented yet")

    def generate_data_from_xml(self, xml_filepath) -> Dict:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        data = {}
        for child in root:
            if child.tag == "image":
                assert("width" in child.attrib and "height" in child.attrib)
                image_name = child.attrib["name"]
                skeletons = []
                skeleton_labels = []
                bboxes = []
                bbox_labels = []
                points = []
                point_labels = []
                for xml_elem in child:
                    if xml_elem.tag == "skeleton":
                        skeleton, label = self.get_skeleton_from_xml(xml_elem)
                        skeletons.append(skeleton)
                        skeleton_labels.append(label)
                    if xml_elem.tag == "point":
                        point, label = self.get_point_from_xml(xml_elem)
                        points.append(point)
                        point_labels.append(point)
                    if xml_elem.tag == "box":
                        bbox, label = self.get_bbox_from_xml(xml_elem)
                        bboxes.append(bbox)
                        bbox_labels.append(label)
                data[image_name] = {
                    "skeletons": skeletons,
                    "skeleton_labels": skeleton_labels,
                    "bboxes": bboxes,
                    "bbox_labels": bbox_labels,
                    "points": points,
                    "point_labels": point_labels
                }
        return data

    def run(self, label_ids: Dict[str, int] = None):
        data = self.generate_data_from_xml(self.xml_filepath)
        progress_bar = tqdm(data, disable=False, dynamic_ncols=True)
        progress_bar.set_description("copying annotations...")
        for fname in progress_bar:
            try:
                dst_fpath = self.get_annotations_fpath(fname)
                image_data = data[fname]
                self.write_annotation(dst_fpath, image_data, label_ids)
            except AssertionError as e:
                print("{} failed".format(fname))

class CVATKeypointRCNNCopier(CVATCopier):
    def __init__(self, xml_filepath: str, dst_root: str, raw_image_extension = ".jpg"):
        self.raw_image_extension = raw_image_extension
        self.raw_dst_images = []
        self.raw_dst_image_fnames = []
        super().__init__(xml_filepath, dst_root)

    def initialise(self):
        self.raw_dst_images = sorted(glob.glob(os.path.join(self.dst_root, "**/*{}".format(self.raw_image_extension)),
                                               recursive=True))
        self.raw_dst_image_fnames = [os.path.basename(f).replace(self.raw_image_extension, "")
                                     for f in self.raw_dst_images]

    def get_annotations_fpath(self, fname):
        src_fname = os.path.splitext(os.path.basename(fname))[0]
        if src_fname not in self.raw_dst_image_fnames:
            raise Exception("Could not find {} in dst_fname list".format(src_fname))
        index = self.raw_dst_image_fnames.index(src_fname)
        return self.raw_dst_images[index].replace(self.raw_image_extension, ".json")

    def write_annotation(self, fname, image_data, label_ids: Dict[str, int] = None):
        data = find_skeletons_corresponding_to_bbox(image_data["bboxes"], image_data["bbox_labels"],
                                                    image_data["skeletons"], image_data["skeleton_labels"],
                                                    label_ids)
        bboxes = data["bboxes"]
        skeletons = data["skeletons"]
        labels = "[" + ', '.join(['"{}"'.format(i) for i in data["labels"]]) + "]"

        if len(bboxes) == 0:
            return

        with open(fname, 'w') as f:
            str_data = '"bboxes": {}, "keypoints": {}, "labels": {}'.format(bboxes,skeletons, labels)
            f.write("{" + str_data + "}")

