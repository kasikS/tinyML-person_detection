# tinyML-person_detection
Experimenting with tinyML - person detection

### image_viewer.py
script to save the values retrieved from a camera module OV7675 as .jpg


### camera_cont_display.py
script to continuously display an image from a camera module OV7675
It also displays label and prediction.

### person_detection.py
creates a neural network using tensorflow to detect a person on the image. 
It expects the dataset in a folder `dataset` with subfolders:

`0` a dataset 60 images of various background

`1` a dataset of 60 images of human faces

It expects test images in a folder `test` (used for inference) with the same subfolder setup.

Images retrieved from OV7675 camera saved with `image_viewer.py` script

The script saves a model to a file.


### use_person_detection_model.py
Checks how the model performs when loaded from a file and when converted to tflite.
It expects test images in a folder `test` (used for inference)
Saves a model as c array.


### Next steps:
- experiment with model and layers and hyperparameters.
- create a model in Edge Impulse

