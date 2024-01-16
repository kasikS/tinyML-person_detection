#use a saved model and run inference, convert to TFlite and run inference again

import cv2
import numpy as np
import tensorflow as tf
import keras
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


IMG_WIDTH = 100
IMG_HEIGHT = 100

saved_dir = 'detect_person'
tflite_file_path = 'detect_person_tflite'

reconstructed_model = keras.models.load_model(saved_dir)

probability_model = tf.keras.Sequential([reconstructed_model, tf.keras.layers.Softmax()])

##
test_image = cv2.imread(os.path.join('test', 'dog.201.jpg')) #person_2_in_dataset.jpg

test_image = cv2.resize(test_image, (IMG_WIDTH, IMG_HEIGHT))
plt.imshow(test_image)
plt.colorbar()
plt.grid(False)
plt.show()

test_image = np.array(test_image)

test_image = np.array(test_image)/255
test_image = test_image.astype(np.float32)


image_dtype = test_image.dtype
print("Image data type:", image_dtype)

test_image = (np.expand_dims(test_image, 0))
print(test_image.shape)

predictions_single = probability_model.predict(test_image)

print(predictions_single)
sel_label = np.argmax(predictions_single[0])
print(sel_label)


# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)

tflite_model = converter.convert()
with open(tflite_file_path + '.tflite' , 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(tflite_file_path + '.tflite')

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input details
print(input_details)
# output details
print(output_details)

image_dtype = test_image.dtype
print("Image data type:", image_dtype)


print(test_image.shape)

# input_details[0]['index'] = the index which accepts the input
interpreter.set_tensor(input_details[0]['index'], test_image)

# run the inference
interpreter.invoke()

# output_details[0]['index'] = the index which provides the input
output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)