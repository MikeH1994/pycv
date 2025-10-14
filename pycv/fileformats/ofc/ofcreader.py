from __future__ import annotations
import datetime
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class OFCSetpoint:
    setpoint_name: str | None = None
    channel_names: List[str] | None = None
    timestamps: List[datetime.datetime] | None = None
    data: np.ndarray = None
    mean_per_channel: np.ndarray = None
    std_per_channel: np.ndarray = None
    room_temperature: float | None = None
    relative_humidity: float | None = None
    source: str | None = None

    def mean(self, index: int | str):
        if isinstance(index, int):
            return self.mean_per_channel[index]
        elif isinstance(index, str):
            assert(index in self.channel_names)
            i = self.channel_names.index(index)
            return self.mean_per_channel[i]
        else:
            raise Exception(f"Invalid type passed ({type(index)})")

    def std(self, index: int | str):
        if isinstance(index, int):
            return self.std_per_channel[index]
        elif isinstance(index, str):
            assert(index in self.channel_names)
            i = self.channel_names.index(index)
            return self.std_per_channel[i]
        else:
            raise Exception(f"Invalid type passed ({type(index)})")

def load_data_from_section(lines, n_channels = None) -> Tuple[List, NDArray]:
    timestamps = []
    data = []
    for line in lines:
        if not is_data_line(line, n_channels):
            continue
        tokens = line.split("\t")
        timestamps.append(to_timestamp(tokens[0] + "\t" + tokens[1]))
        data.append([float(val) for val in tokens[2:]])
    return timestamps, np.array(data, dtype=np.float32)


def split_file_into_sections(lines: List[str]) -> List[List[str]]:
    section_start_points = find_start_points_of_sections(lines)
    sections = []
    for index in range(len(section_start_points)):
        i0 = section_start_points[index]
        if index != len(section_start_points) - 1:
            i1 = section_start_points[index + 1] - 1
            sections.append(lines[i0:i1])
        else:
            sections.append(lines[i0:])
    return sections


def find_start_points_of_sections(lines: List[str]) -> List[int]:
    start_points = []
    for i, line in enumerate(lines):
        if line.startswith("Data collected with Online Furnace Control"):
            start_points.append(i)
    return start_points


def find_chennel_names_in_section(lines: List[str]):
    substring = "Temperature\tHumidity"
    if substring not in lines:
        return []
    index = lines.index(substring)
    tokens = lines[index+2].strip().split("\t")
    names = []
    for token in tokens:
        if token not in names:
            names.append(token.replace("::INSTR", ""))
    return names


def find_temperature_and_humidity_in_section(lines: List[str]):
    substring = "Temperature\tHumidity"

    if substring not in lines:
        return None, None

    index = lines.index(substring)
    tokens = lines[index + 1].split("\t")

    if len(tokens) != 2:
        return None, None

    if tokens[0] == "21.0" and tokens[1] == "0.0":
        # these are the default values; if these are found then nothing has been entered
        return None, None

    return float(tokens[0]), float(tokens[1])


def find_start_and_end_of_data(lines: List[str], n_channels=None):
    indices = []
    for i, line in enumerate(lines):
        line = line.strip()
        if is_data_line(line, n_channels):
            indices.append(i)
    if len(indices) == 0:
        return None, None
    start_index = indices[0]
    end_index = indices[-1]
    return start_index, end_index


def find_data_in_section_from_tags(lines: List[str]):
    pass


def is_data_line(line: str, n_channels = None):
    line = line.strip()
    tokens = line.split("\t")
    if len(tokens) < 3:
        return False

    if not is_datetime_stamp(tokens[0] + "\t" + tokens[1]):
        return

    remaining_tokens = tokens[2:]
    if not n_channels is None:
        if len(remaining_tokens) != n_channels:
            return False

    for token in remaining_tokens:
        if not is_number(token):
            return False

    return True


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_datetime_stamp(date_string):
    return not to_timestamp(date_string) is None


def to_timestamp(date_string):
    fmt = '%d/%m/%Y\t%H:%M:%S'
    try:
        return datetime.datetime.strptime(date_string, fmt)
    except ValueError:
        return None


def process_section(lines: List[str]):
    channel_names = find_chennel_names_in_section(lines)
    n_channels = len(channel_names)
    start_index, end_index = find_start_and_end_of_data(lines, n_channels)
    if start_index is None or end_index is None:
        return None

    dataset_name = lines[start_index - 1] if start_index > 0 else ""
    dataset_lines = lines[start_index: end_index+1]
    timestamps, data = load_data_from_section(dataset_lines, n_channels)
    room_temp, humidity = find_temperature_and_humidity_in_section(lines)
    mean_per_channel = np.mean(data, axis=0)
    std_per_channel = np.std(data, axis=0)

    return OFCSetpoint(setpoint_name=dataset_name, channel_names=channel_names, timestamps = timestamps,
                       data=data, room_temperature=room_temp, relative_humidity=humidity,
                       mean_per_channel=mean_per_channel, std_per_channel=std_per_channel)


class OFCReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.setpoints = self.read()

    def read(self):
        with open(self.filepath) as f:
            lines = [l.strip() for l in f.readlines()]
        sections = split_file_into_sections(lines)
        return [process_section(section) for section in sections]
