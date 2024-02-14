#shows and saves image retrieved from a camera

# Import the needed libraries
from matplotlib import pyplot as plt
import numpy as np
import struct
import cv2
import serial
import datetime


# todo
# person_detection.py to try: normalize. use data augmentation, more data, without resizing. play with layers-size/accuracy etc. more metrics (confusion matrix)?
# prepare use_person_detection_model.py to  check representative dataset
# check the same dataset with edge impulse lib


image_size_grayscale = 176 * 144 * 1
image_size_rgb565 = 176 * 144 * 2

ser = serial.Serial('/dev/ttyACM0', baudrate=9600)
GRAYSCALE = True

while True:

    if GRAYSCALE:
        data = ser.read(image_size_grayscale) #full image
        raw_bytes = struct.unpack('>25344B', data) #176*144  uint8
        image = np.zeros((176 * 144, 1), dtype=np.uint8)
        image_np = np.reshape(raw_bytes,(144, 176,1))

    else:
        # for rgb 565 format
        data = ser.read(image_size_rgb565) #full image
        raw_bytes = struct.unpack('>25344H', data) #176*144 unsigned short 2 bytes
        image = np.zeros((176*144, 3), dtype=int)
        for i in range(len(raw_bytes)):
            #Read 16-bit pixel
            pixel = raw_bytes[i]
            #Convert RGB565 to RGB 24-bit
            r = ((pixel >> 11) & 0x1f) << 3
            g = ((pixel >> 5) & 0x3f) << 2
            b = ((pixel >> 0) & 0x1f) << 3
            image[i] = [r,g,b]

        image_np = np.reshape(image,(144, 176,3)) #QCIF resolution



    print(raw_bytes)
    print(image_np.dtype)
    print(image_np.shape)

    # Show the image
    plt.imshow(image_np, cmap='gray', vmin=0, vmax=255)
    plt.show()


    # Save the image as a JPEG file
    print("save?")
    s = input()
    if s=='t':
        x = datetime.datetime.now()
        cv2.imwrite("arduino_"+x.strftime("%Y%m%d%H%M%S") +'.jpg', image_np)