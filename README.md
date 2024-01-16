# tinyML-person_detection
Experimenting with tinyML - person detection

### image_viewer.py
script to save the values retrieved from a camera module OV7675 as .jpg

### person_detection.py
creates a neural network using tensorflow to detect a person on the image. 
It expects the dataset in a folder `dataset` with subfolders:

`0` a dataset of 1000 images out of lanscapes, cats and dogs

`1` a dataset of total 1000 images of human faces

I downloaded images from [kaggle](https://www.kaggle.com/datasets) and added some images from OV7675 camera saved with `image_viewer.py` script

The script saves a model to a file.

TODO
experiment:

- further with layers
- with data augmentation
- with depthwise separable convolution
- with quantization
