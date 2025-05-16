from __future__ import annotations
import io
import subprocess
import pandas as pd
import os


class TensorboardReader:
    def __init__(self, log_dir: str):
        """

        :param log_dir:
        """
        self.data = self.get_data(log_dir)

    def get_data(self, log_dir):
        """

        :param log_dir:
        :return:
        """
        output = str(subprocess.check_output(['tensorboard', '--inspect', '--logdir', log_dir]))[2:-1]
        params = []
        current_param = None
        param_names = ["audio", "graph", "histograms", "images", "scalars", "tensor"]
        variable_names = ["first_step", "last_step", "max_step", "min_step", "num_steps"]
        for line in output.split("\\n"):
            tokens = line.split()
            if len(tokens) == 0:
                continue
            if tokens[0] in param_names:
                if current_param is not None:
                    params.append(current_param)
                current_param = {"name": tokens[0]}
            elif len(tokens) == 2:
                if tokens[0] in variable_names:
                    try:
                        current_param[tokens[0]] = int(tokens[1])
                    except ValueError:
                        pass
        params.append(current_param)
        return params

    def start_epoch(self) -> int:
        """

        :return:
        """
        start_epoch = 0
        for param in self.data:
            if "first_step" in param:
                start_epoch = min(start_epoch, param["first_step"])
        return start_epoch

    def end_epoch(self) -> int:
        """

        :return:
        """
        end_epoch = -1
        for param in self.data:
            if "last_step" in param:
                end_epoch = max(end_epoch, param["last_step"])
        return end_epoch
