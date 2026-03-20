import glob
import os
import tifffile as tiff
import pycv
from pycv.calibration import CameraCalibration, CalibrationTarget
from pycv.constants import CALIB_CB_CHECKERBOARD


def calibrate_camara(image_fpaths, target, device_name="device_name", plot=False, verbose=False):
    calibration = CameraCalibration(device_name)
    for i, fpath in enumerate(image_fpaths):
        img32f = tiff.imread(fpath)
        img8 = pycv.convert_to_8_bit(img32f)
        calibration.add_calibration_point(img8, target, plot=plot, verbose=verbose)
    calibration.calibrate(verbose=True)
    return calibration

def run(root):
    image_fpaths = glob.glob(os.path.join(root, "*.tif"))
    target = CalibrationTarget((13,8), 0.02, CALIB_CB_CHECKERBOARD)
    cal = calibrate_camara(image_fpaths, target)
    cal.save(os.path.join(root, "calibration.json"))

if __name__ == "__main__":
    run("some_folderpath")
