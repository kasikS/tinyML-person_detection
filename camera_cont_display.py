from matplotlib import pyplot as plt
import numpy as np
import struct
import cv2
import serial
import datetime
import readimage

import time

WIDTH = 96
HEIGHT = 96

ser = serial.Serial('/dev/ttyACM0', baudrate=9600)

while True:

    #read first 2 bytes that represent label and probability
    data_header = ser.read(2)
    results = struct.unpack('>2B', data_header)
    print(results[0])
    print(results[1])

    image_np = readimage.get_image(readimage.Format.GRAYSCALE, WIDTH, HEIGHT, ser)
    image_np = cv2.resize(image_np.astype('uint8'), (WIDTH*4, HEIGHT*4), interpolation=cv2.INTER_AREA)


    if results[1] == 0:
        label = 'not a person'
    else:
        label = 'person'

    image_cv2 = cv2.UMat(np.array(image_np))
    # # Draw label on preview window
    cv2.putText(image_cv2,
                'label: '+ label,
                (0, 12),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255))

    # Draw probability on preview window
    cv2.putText(image_cv2,
                'score: ' + str(results[0]),
                (0, 24),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255))

    image_display = cv2.convertScaleAbs(image_cv2)
    # Show the frame
    cv2.imshow("Frame", image_display)

# Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()