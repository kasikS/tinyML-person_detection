#use a saved model and run inference, convert to TFlite and run inference again

import cv2
import numpy as np
import tensorflow as tf
import flatbuffers
# from tensorflow import keras
import keras
import os

from keras.models import Model

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import dataset

IMG_WIDTH = 96
IMG_HEIGHT = 96

test_dir = 'output' #'test'

saved_dir = 'model'
file = 'detect_person.keras'
tflite_file_path = 'detect_person_tflite.tflite'
c_array_model_fname = 'model.h'

def convert_to_c_array():
  """ Converts the TFLite model to C array"""
  os.system('xxd -i  ' + tflite_file_path + ' > ' + c_array_model_fname)
  print("C array model created " + c_array_model_fname)


def representative_data_gen():
  for file in os.listdir('representative'):
    # open, resize, convert to numpy array
    image = cv2.imread(os.path.join('representative',file), 0)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype(np.float32)
    # image -=128
    image = image/127.5 -1
    image = (np.expand_dims(image, 0))
    image = (np.expand_dims(image, 3))
    yield [image]

#load model
probability_model= keras.models.load_model(os.path.join(saved_dir, file ))

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

#save .tflite
with open(tflite_file_path, 'wb') as f:
  f.write(tflite_model)
convert_to_c_array()


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Get input quantization parameters.
input_quant = input_details[0]['quantization_parameters']
input_scale = input_quant['scales'][0]
input_zero_point = input_quant['zero_points'][0]


#read test image and plot it
test_image = cv2.imread(os.path.join('test', '1', 'arduino_k.jpg'),0)  #example test image, modify as needed
test_image= cv2.resize(test_image, (IMG_WIDTH, IMG_HEIGHT))
test_image= np.array(test_image)/127.5 -1

plt.imshow(test_image,  cmap='gray', vmin=-1, vmax=1)
plt.colorbar()
plt.grid(False)
plt.show()


test_image = (np.expand_dims(test_image, 0))
test_image = (np.expand_dims(test_image, 3))
print(test_image.shape)
#quantize input image
input_value = (test_image/ input_scale) + input_zero_point
input_value = tf.cast(input_value, dtype=tf.int8)

interpreter.set_tensor(input_details[0]['index'], input_value)

# run the inference
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)




#check accuracy of saved model and converted to tflite - not quantized one, for comparison
test_images, test_labels = dataset.load_data(test_dir, normalize= True, toint= False)
test_labels = tf.keras.utils.to_categorical(test_labels)
test_images_np = np.array(test_images)
test_labels_np = np.array(test_labels)
score = probability_model.evaluate(test_images_np, test_labels_np, verbose=2)

# print("Test loss:", score[0])
print("Test accuracy model:", score[1])



correct = 0
for img_idx in range(len(test_images)):

  tst = np.int8( (test_images[img_idx] / input_scale) + input_zero_point)

  # tst = np.int8(test_images[img_idx])

  tst = (np.expand_dims(tst, 0))
  tst = (np.expand_dims(tst, 3))

  interpreter.set_tensor(input_details[0]['index'], tst)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details[0]['index'])
  prediction = np.argmax(output[0])
  lbl=np.argmax(test_labels[img_idx])
  if prediction == lbl:
    correct += 1
accuracy_quant = correct/len(test_images)
print("Test accuracy quant:", accuracy_quant)


basic_model_size = os.path.getsize(os.path.join(saved_dir, file))
quantized_model_size = os.path.getsize(tflite_file_path)
print("Basic model is %d bytes" % basic_model_size)
print("Quantized model is %d bytes" % quantized_model_size)



