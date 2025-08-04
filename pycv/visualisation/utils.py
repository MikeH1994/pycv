from dataclasses import dataclass
from typing import List, Dict
import matplotlib.axes
import matplotlib.image
import matplotlib.lines
import numpy as np

@dataclass
class DataSeries:
    x_label: str = None
    y_label: str = None
    label:str  = None


    def show(self, ax: matplotlib.axes.Axes):
        raise Exception("Function not implemented")

    def update(self):
        raise Exception("Function not implemented")

@dataclass
class ImageSeries(DataSeries):
    data: List[np.ndarray] = None
    img: matplotlib.image.AxesImage = None
    current_index: int = 0

    def show(self, ax: matplotlib.axes.Axes) -> matplotlib.image.AxesImage:
        self.img = ax.imshow(self.data[0])
        return self.img

    def update(self):
        if self.img is None:
            raise Exception("show() must be called first")

        if self.current_index >= len(self.data):
            self.current_index = 0
        self.img.set_data(self.data[self.current_index])
        self.current_index += 1

@dataclass
class LineplotSeries(DataSeries):
    x_data: List[np.ndarray] = None
    y_data: List[np.ndarray] = None
    line: matplotlib.lines.Line2D = None
    current_index: int = 0

    def show(self, ax: matplotlib.axes.Axes) -> matplotlib.lines.Line2D:
        self.line, = ax.plot(self.x_data[0], self.y_data[0])
        return self.line

    def update(self):
        if self.line is None:
            raise Exception("show() must be called first")

        if self.current_index >= len(self.x_data):
            self.current_index = 0

        self.line.set_xdata(self.x_data[self.current_index])
        self.line.set_ydata(self.y_data[self.current_index])
        self.current_index += 1
