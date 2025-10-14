import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycv.fileformats import IRBISFolder, IRBISCSVImageStack, IRBISMetadata
import pycv
from tqdm.auto import tqdm
import os
import tkinter as tk
from tkinter import filedialog

def run(root, extension):
    folders = [f for f in pycv.get_all_folders_containing_filetype(root, extension)[0] if "offset" not in f]
    pbar = tqdm(folders)
    image_number = 0
    for i, input_folderpath in enumerate(pbar):
        print(input_folderpath)
        folder = IRBISFolder.load(input_folderpath, extension=extension)

        for image_stack in folder.image_stacks:
            image = image_stack.image
            metadata = image_stack.metadata
            integration_time = int(metadata.integration_time)

            plt.imshow(image)
            plt.title(f"Integration time: {integration_time}us")

if __name__ == '__main__':
    run()