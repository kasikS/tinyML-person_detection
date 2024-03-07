#use a saved model and run inference, convert to TFlite and run inference again

import cv2
import numpy as np
import tensorflow as tf
import keras
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import dataset

IMG_WIDTH = 96
IMG_HEIGHT = 96

test_dir = 'test'

saved_dir = 'detect_person'
tflite_file_path = 'detect_person_tflite.tflite'
c_array_model_fname = 'array.c'

def convert_to_c_array():
  """ Converts the TFLite model to C array"""
  os.system('xxd -i  ' + tflite_file_path + ' > ' + c_array_model_fname)
  print("C array model created " + c_array_model_fname)


### Based on: https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/saliency.py
def get_saliency_map(img_array, model, class_idx):
  model.layers[-1].activation = None
  # Gradient calculation requires input to be a tensor
  imagef = np.float32(img_array)
  img_tensor = tf.convert_to_tensor(imagef)

  # Do a forward pass of model with image and track the computations on the "tape"
  with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
    # Compute (non-softmax) outputs of model with given image
    tape.watch(img_tensor)
    outputs = model(img_tensor, training=False)

    # Get score (predicted value) of actual class
    score = outputs[:, class_idx]

  # Compute gradients of the loss with respect to the input image
  grads = tape.gradient(score, img_tensor)

  # Finds max value in each color channel of the gradient (should be grayscale for this demo)
  grads_disp = [np.max(g, axis=-1) for g in grads]

  # There should be only one gradient heatmap for this demo
  grad_disp = grads_disp[0]

  # The absolute value of the gradient shows the effect of change at each pixel
  # Source: https://christophm.github.io/interpretable-ml-book/pixel-attribution.html
  grad_disp = tf.abs(grad_disp)

  # Normalize to between 0 and 1 (use epsilon, a very small float, to prevent divide-by-zero error)
  heatmap_min = np.min(grad_disp)
  heatmap_max = np.max(grad_disp)
  heatmap = (grad_disp - heatmap_min) / (heatmap_max - heatmap_min + tf.keras.backend.epsilon())

  return heatmap.numpy()

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


#load model
# reconstructed_model = keras.models.load_model(saved_dir)
# probability_model = tf.keras.Sequential([reconstructed_model, tf.keras.layers.Softmax()])
probability_model= keras.models.load_model(saved_dir)

#read test image and plot it
test_image = cv2.imread(os.path.join('test', '1', 'arduino_13.jpg'),0)  #this image is incorrectly classified 'bcg_20240207160425.jpg'
true_idx = 1
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

#make prediction
predictions_single = probability_model.predict(test_image)
print(predictions_single)
sel_label = np.argmax(predictions_single[0])
print(sel_label)



### Generate saliency map for the given input image
saliency_map = get_saliency_map(test_image, probability_model, true_idx)
### Draw map
plt.imshow(saliency_map, cmap='jet', vmin=0.0, vmax=1.0)

### Overlay the saliency map on top of the original input image
idx = 0
ax = plt.subplot()
ax.imshow(test_image[idx,:,:,0], cmap='gray', vmin=0.0, vmax=255)
ax.imshow(saliency_map, cmap='jet', alpha=0.25)
plt.show()


# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
# #quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open(tflite_file_path, 'wb') as f:
  f.write(tflite_model)
convert_to_c_array()

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(tflite_file_path)

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

#check accuracy of saved model and converted to tflite and optimized, perhaps check accuracy without optimization as well?
test_images, test_labels = dataset.load_data(test_dir)
test_labels = tf.keras.utils.to_categorical(test_labels)
test_images_np = np.array(test_images)
test_labels_np = np.array(test_labels)
score = probability_model.evaluate(test_images_np, test_labels_np, verbose=2)
# print("Test loss:", score[0])
print("Test accuracy model:", score[1])

correct = 0
for img_idx in range(len(test_images)):
  tst = np.int8(test_images[img_idx])
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


