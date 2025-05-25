import numpy as np

def load_sco_file(fpath, n_header_bytes=419):
    # .sco file has a 419 byte header
    # bytes 1,2 are ???
    # bytes 3,4 are the width as uint16
    # bytes 5,6 are the height as uint16
    # last (height*width*2) bytes are the offsets, stored as uint16
    data = np.fromfile(fpath, dtype=np.uint8)
    header = data[:n_header_bytes]
    width = header[2:4].view(dtype=np.int16).item()
    height = header[4:6].view(dtype=np.int16).item()

    n_image_bytes = width * height * 2
    image = data[-n_image_bytes:].view(dtype=np.int16)
    return image.reshape((height, width))


def load_scg_file(fpath, n_header_bytes=419):
    # .scg file has a 419 byte header

    # bytes 1,2 are ???
    # bytes 3,4 are the width as uint16
    # bytes 5,6 are the height as uint16
    # last (height*width*4) bytes are the gain, stored as float32
    data = np.fromfile(fpath, dtype=np.uint8)
    header = data[:n_header_bytes]
    width = header[2:4].view(dtype=np.int16).item()
    height = header[4:6].view(dtype=np.int16).item()

    n_image_bytes = width * height * 4
    image = data[-n_image_bytes:].view(dtype=np.float32)

    return image.reshape((height, width))

def load_sbp_file(fpath, n_header_bytes=419):
    # sbp file has 419 byte header, with width*height*9 data buffer
    return None