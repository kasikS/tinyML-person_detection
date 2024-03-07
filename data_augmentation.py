import numpy as np
import matplotlib.pyplot as plt
import random
import os
import PIL

import skimage.transform
import skimage.util

# Location of dataset and output folder
DATASET_PATH = "dataset"
OUT_PATH = "output"
OUT_ZIP = "augmented_dataset.zip"

# File format to use for new dataset
IMG_EXT = ".jpg"
SEED = 42
random.seed(SEED)

### Create output directory
try:
  os.makedirs(OUT_PATH)
except FileExistsError:
  print("WARNING: Output directory already exists. Check to make sure it is empty.")

### Function to create 3 new flipped images of the input
def create_flipped(img):
  # Create a list of flipped images
  flipped = []
  flipped.append(np.fliplr(img))
  flipped.append(np.flipud(img))
  flipped.append(np.flipud(np.fliplr(img)))

  return flipped

### Function to create new rotated images of the input
def create_rotated(img, rotations):
  # Create list of rotated images (keep 8-bit values)
  rotated = []
  for rot in rotations:
    img_rot = skimage.transform.rotate(img, angle=rot, mode='edge', preserve_range=True)
    img_rot = img_rot.astype(np.uint8)
    rotated.append(img_rot)

  return rotated

### Function to add random noise to images
def create_noisy(img, types, seed=None):
  # Add noise of different types
  noisy_imgs = []
  for t in types:
    noise = skimage.util.random_noise(img, mode=t, seed=seed)
    noise = (noise * 255).astype(np.uint8)
    noisy_imgs.append(noise)

  return noisy_imgs


### Function to open image and create a list of new transforms
def create_transforms(file_path):
  # Open the image
  img = PIL.Image.open(file_path)

  # Convert the image to a Numpy array (keep all color channels)
  img_array = np.asarray(img)

  # Add original image to front of list
  img_tfs = []
  img_tfs.append([img_array])

  # Perform transforms (call your functions)
  img_tfs.append(create_flipped(img_array))
  img_tfs.append(create_rotated(img_array, [45, 90, 135]))
  img_tfs.append(create_noisy(img_array, ['gaussian', 's&p'], SEED))

  # Flatten list of lists (to create one long list of images)
  img_tfs = [img for img_list in img_tfs for img in img_list]

  return img_tfs


### Load all images, create transforms, and save in output directory

# Find the directories in the dataset folder
for label in os.listdir(DATASET_PATH):
  class_dir = os.path.join(DATASET_PATH, label)
  # Create output directory
  out_path = os.path.join(OUT_PATH, label)
  os.makedirs(out_path, exist_ok=True)

  # Go through each image in the subfolder
  for i, filename in enumerate(os.listdir(class_dir)):
    # Get the root of the filename before the extension
    file_root = os.path.splitext(filename)[0]

    # Do all transforms for that one image
    file_path = os.path.join(DATASET_PATH, label, filename)
    img_tfs = create_transforms(file_path)

    # Save images to new files in output directory
    for i, img in enumerate(img_tfs):
      # Create a Pillow image from the Numpy array
      img_pil = PIL.Image.fromarray(img)

      # Construct filename (_.)
      out_file_path = os.path.join(out_path, file_root + "_" + str(i) + IMG_EXT)

      # Convert Numpy array to image and save as a file
      img_pil = PIL.Image.fromarray(img)
      img_pil.save(out_file_path)

