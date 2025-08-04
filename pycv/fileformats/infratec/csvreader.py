from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pycv
from tqdm.auto import tqdm
import os

class IRBISMetadata:
    fields = {
        "ImageWidth": int,
        "ImageHeight": int,
        "ShotRange": str,
        "CalibRange": str,
        "TempUnit": str,
        "RecDate": str,
        "RecTime": str,
        "ms": float,
        "CalName": str,
        "CamTemp": float,
        "IntegTime": float
    }

    def __init__(self, metadata):
        self.folderpath = metadata.pop("Folderpath")
        self.width = metadata.pop("ImageWidth")
        self.height = metadata.pop("ImageHeight")
        self.calibration_name = metadata.pop("CalName")
        self.calibration_range = metadata.pop("CalibRange")
        self.integration_time = metadata.pop("IntegTime")
        self.temp_unit = metadata.pop("TempUnit")
        self.header_length = metadata.pop("HeaderLength")
        self.delimiter = metadata.pop("Delimiter")
        self.metatada = metadata

    def settings_to_dict(self):
        return {
            "Folderpath": self.folderpath,
            "ImageWidth": self.width,
            "ImageHeight": self.height,
            "CalName": self.calibration_name,
            "CalibRange": self.calibration_range,
            "IntegTime": self.integration_time,
            "TempUnit": self.temp_unit,
            "HeaderLength": self.header_length,
            "Delimiter": self.delimiter
        }

    def key(self):
        calibration_range = self.calibration_range if self.temp_unit != "DL" else None
        return self.folderpath, self.width, self.height, self.calibration_name, calibration_range, self.integration_time, self.temp_unit

    def name(self):
        return f"{os.path.basename(self.folderpath)} {self.calibration_name} {self.calibration_range} {self.integration_time} {self.temp_unit}"

    @staticmethod
    def load(fpath: str) -> IRBISMetadata:
        folderpath = os.path.dirname(fpath)
        metadata: Dict[str, Any] = {key: None for key in IRBISMetadata.fields}
        with open(fpath) as f:
            header_length = 0
            for line in f:
                if "[Data]" in line:
                    break
                header_length += 1
                if line == "":
                    continue
                if "=" not in line:
                    continue
                token, value = line.rstrip().split("=")
                if token == "ShotRange" or token == "CalibRange":
                    metadata["Delimiter"] = IRBISMetadata.determine_delimiter(value)
                if token in IRBISMetadata.fields:
                    metadata[token] = IRBISMetadata.fields[token](value)
                else:
                    metadata[token] = value
        metadata["TempUnit"] = metadata["TempUnit"].replace("°C", "degC")
        metadata["HeaderLength"] = header_length + 1
        metadata["Filepath"] = fpath
        metadata["Folderpath"] = folderpath
        return IRBISMetadata(metadata)

    @staticmethod
    def determine_delimiter(value, expected_length=2):
        for delimiter in [";", "\t", " ", ","]:
            if delimiter in value and len(value.split(delimiter)) == expected_length:
                return delimiter
        return None

    @staticmethod
    def collate(metadata: List[IRBISMetadata]):
        if len(metadata) == 0:
            return None
        assert(all([metadata[0].key() == m.key() for m in metadata]))
        settings = metadata[0].settings_to_dict()
        collated_metadata = {}

        # set initial lists for each element
        for m in metadata:
            for key in m.metatada.keys():
                collated_metadata[key] = []

        # collate variables
        for metadata_i in metadata:
            for key in collated_metadata:
                if key in metadata_i.metatada.keys():
                    collated_metadata[key].append(metadata_i.metatada[key])
                else:
                    collated_metadata[key].append(None)

        # if float or int, convert to numpy array
        for key in collated_metadata:
            dtype = IRBISMetadata.fields[key] if key in IRBISMetadata.fields else None
            if dtype == int:
                collated_metadata[key] = np.array(collated_metadata[key], dtype=np.int32)
            elif dtype == float:
                collated_metadata[key] = np.array(collated_metadata[key], dtype=np.float32)
        collated_metadata.update(settings)
        return IRBISMetadata(collated_metadata)

class IRBISCSVImage:
    def __init__(self, image: np.ndarray, metadata: IRBISMetadata):
        self.image = image
        self.metadata = metadata

    def is_temp(self):
        return self.metadata.temp_unit == "degC"

    def key(self):
        return self.metadata.key()

    def name(self):
        return self.metadata.name()

    @staticmethod
    def load(fpath) -> IRBISCSVImage:
        metadata = IRBISMetadata.load(fpath)
        img = np.genfromtxt(fpath, delimiter=metadata.delimiter, skip_header=metadata.header_length)
        img = img[:metadata.height, :metadata.width]
        return IRBISCSVImage(img, metadata)

    @staticmethod
    def collate(image_list: List[IRBISCSVImage]) -> IRBISCSVImage:
        metadata_list = [im.metadata for im in image_list]
        image_stack = np.array([im.image for im in image_list], dtype=np.float32)
        metadata = IRBISMetadata.collate(metadata_list)
        return IRBISCSVImage(image_stack, metadata)

class IRBISFolder:
    def __init__(self, datasets: List[IRBISCSVImage]):
        self.datasets = datasets

    def get(self, index=0):
        if index>=len(self.datasets):
            return None
        return self.datasets[index]

    @staticmethod
    def load(folderpath, extension=".csv", recursive=False, show_progress_bar=False, units=None, clip_dataset=None):
        results = {}
        fpaths = pycv.get_all_files_in_folder(folderpath, extension, recursive=recursive)
        if clip_dataset is not None:
            fpaths = fpaths[:clip_dataset]
        loader = tqdm(fpaths, disable = not show_progress_bar)
        for fpath in loader:
            img = IRBISCSVImage.load(fpath)
            if img.key() not in results:
                results[img.key()] = []
            results[img.key()].append(img)

        for key in results.keys():
            results[key] = IRBISCSVImage.collate(results[key])
        results = [f for f in results.values()]

        return IRBISFolder(results)
