import os
import cv2
import numpy as np


IMG_WIDTH = 96
IMG_HEIGHT = 96
def load_data(data_dir, normalize = False, toint = False):
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

            # normalize for the range [-1, 1] as we'll use int8 on microcontroller
            # in that case it cannot be used as a layer thus part of the model since we'll need to quantize inputs. Which means model will expect int8 values for pixels
            if normalize:
                image = np.array(image)/127.5 -1
            elif toint:
                image = np.int8(image - 128)
            else:
                image = np.array(image)

            images.append(image)
    return images, labels