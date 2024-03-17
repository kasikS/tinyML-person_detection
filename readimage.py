import numpy as np
import struct
import serial
from enum import Enum

class Format(Enum):
    RGB565 = 1
    GRAYSCALE = 2
    GRAYSCALE_INT = 3

def get_image(format, width, height, serial_handle):
    """
    Retrieves image from serial port and returns it as numpy array
    """
    bytes_per_pixel = 1
    pixels = width * height
    bytes_to_read = pixels * bytes_per_pixel

    if format == Format.GRAYSCALE:
        data_type = np.uint8
        repres = 'B'
    elif format == Format.RGB565:
        data_type = np.int
        repres = 'H'
    else:
        data_type = np.uint8
        repres = 'b'

    data_format = '>' + str(pixels) + repres

    data = serial_handle.read(bytes_to_read)
    raw_bytes = struct.unpack(data_format, data)


    # data_type = np.uint8 if format == Format.GRAYSCALE else np.uint16
    image = np.zeros((width * height, bytes_per_pixel), dtype = data_type)

    if format == Format.RGB565:
        #convert rgb565 to rgb888
        for i in range(len(raw_bytes)):
            # Read 16-bit pixel
            pixel = raw_bytes[i]
            # Convert RGB565 to RGB 24-bit
            r = ((pixel >> 11) & 0x1f) << 3
            g = ((pixel >> 5) & 0x3f) << 2
            b = ((pixel >> 0) & 0x1f) << 3
            image[i] = [r, g, b]

    image_np = np.reshape(raw_bytes,(height, width, bytes_per_pixel))
    return image_np