import os
import cv2
import numpy as np


IMG_WIDTH = 96
IMG_HEIGHT = 96
def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    `data_dir` has one directory named after each category, numbered
    0 and 1.

    """
    subfolders = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]

    images = []
    labels = []
    for subfolder in subfolders:
        for file in os.listdir(os.path.join(data_dir,subfolder)):
            labels.append(subfolder)
            image = cv2.imread(os.path.join(data_dir, subfolder, file), 0)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = np.array(image)
            images.append(image)
    return images, labels