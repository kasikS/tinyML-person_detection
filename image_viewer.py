#shows and saves image retrieved from a camera

# Import the needed libraries
from matplotlib import pyplot as plt
import numpy as np
import struct
import cv2
import serial
import datetime
import readimage

WIDTH = 176
HEIGHT = 144
ser = serial.Serial('/dev/ttyACM0', baudrate=9600)

while True:

    #read image
    image_np = readimage.get_image(readimage.Format.GRAYSCALE, WIDTH, HEIGHT, ser)

    # Show the image
    plt.imshow(image_np, cmap='gray', vmin=0, vmax=255)
    plt.show()


    # Save the image as a JPEG file
    print("save?")
    s = input()
    if s=='t':
        x = datetime.datetime.now()
        cv2.imwrite("ard"
                    ""+x.strftime("%Y%m%d%H%M%S") +'.jpg', image_np)