#use a saved model and run inference, convert to TFlite and run inference again

import cv2
import numpy as np
import tensorflow as tf
import keras
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


IMG_WIDTH = 96
IMG_HEIGHT = 96

saved_dir = 'detect_person'
tflite_file_path = 'detect_person_tflite.tflite'
c_array_model_path = 'array.c'

def convert_to_c_array():
  """ Converts the TFLite model to C array"""
  os.system('xxd -i  ' + tflite_file_path + ' > ' + c_array_model_path)
  print("C array model created " + c_array_model_path)


reconstructed_model = keras.models.load_model(saved_dir)
probability_model = tf.keras.Sequential([reconstructed_model, tf.keras.layers.Softmax()])

test_image = cv2.imread(os.path.join('test', '1', 'arduino_20240212135913.jpg'),0) #person_2_in_dataset.jpg
test_image = cv2.resize(test_image, (IMG_WIDTH, IMG_HEIGHT))
test_image = np.int8(test_image)
plt.imshow(test_image,  cmap='gray', vmin=-128, vmax=127)
plt.colorbar()
plt.grid(False)
plt.show()

#test_image = np.array(test_image)/255.0
image_dtype = test_image.dtype
print("Image data type:", image_dtype)

test_image = (np.expand_dims(test_image, 0))
test_image = (np.expand_dims(test_image, 3))
print(test_image.shape)

predictions_single = probability_model.predict(test_image)
print(predictions_single)
sel_label = np.argmax(predictions_single[0])
print(sel_label)


#check this and test image as we quantize input to int8
def representative_data_gen():
  for file in os.listdir('representative'):
    # open, resize, convert to numpy array
    image = cv2.imread(os.path.join('representative',file), 0)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image)
    image = (np.expand_dims(image, 0))
    image = (np.expand_dims(image, 3))
    image = np.float32(image)
    yield [image]
  # for input_value in tf.data.Dataset.from_tensor_slices(train_arr).batch(1).take(100):
  #   yield [input_value]


# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
# #quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()


with open(tflite_file_path, 'wb') as f:
  f.write(tflite_model)
convert_to_c_array()

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(tflite_file_path)  #+ '.tflite')

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

interpreter.set_tensor(input_details[0]['index'], test_image)

# run the inference
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)