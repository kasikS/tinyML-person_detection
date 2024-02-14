#train and save a model to recognize a human face on the picture

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import random
import PIL

from skimage.transform import resize
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split

#consider adding those constants to another file
EPOCHS = 200
NUM_CATEGORIES = 2
TEST_SIZE = 0.4
IMG_WIDTH = 96
IMG_HEIGHT = 96


def main():


    # Get image arrays and labels for all image files
    data_dir = 'dataset'
    test_dir = 'test'

    images, labels = load_data(data_dir)
    plt.imshow(images[1], cmap='gray', vmin= -128, vmax=127)
    plt.show()

    test_images, test_labels = load_data(test_dir)

    # images = np.array(images)/255.0
    labels = tf.keras.utils.to_categorical(labels)

    X_train, x_test, Y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size= 0.2)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.25)


    # Get a compiled neural network
    model = get_model()
    print(model.summary())

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val))

    ### Plot training and validation accuracy and loss over time
    # Extract accuracy and loss values (in list form) from the history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create a list of epoch numbers
    epochs = range(1, len(acc) + 1)

    # Plot training and validation loss values over time
    plt.figure()
    plt.plot(epochs, loss, color='blue', marker='.', label='Training loss')
    plt.plot(epochs, val_loss, color='orange', marker='.', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # Plot training and validation accuracies over time
    plt.figure()
    plt.plot(epochs, acc, color='blue', marker='.', label='Training acc')
    plt.plot(epochs, val_acc, color='orange', marker='.', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()


    # Evaluate neural network performance
    score = model.evaluate(x_test,  y_test, verbose=2)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save model to file
    model.save('detect_person')

    #use model
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    plt.imshow(x_test[0], cmap='gray', vmin=-128, vmax=127)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    test_image = (np.expand_dims(x_test[0], 0))
    print(test_image.shape)

    image_dtype = test_image[0].dtype
    print("Image data type:", image_dtype)

    predictions_single = probability_model.predict(test_image)

    print(predictions_single)
    sel_label = np.argmax(predictions_single[0])
    print(sel_label)



    ### Create confusion matrix from validation set
    # Find predictions from all validation samples
    # y_pred = model.predict(x_val)
    # print("Validation output shape:", predictions_single.shape)
    #
    # # Convert actual and predicted validation one-hot encoding to numerical labels
    # y_val = np.argmax(y_val, axis=1)
    # y_pred = np.argmax(y_pred, axis=1)
    #
    # # Print some values from actual and predicted validation sets (first 50 samples)
    # print("Actual validation labels:\t", y_val[:50])
    # print("Predicted validation labels:\t", y_pred[:50])
    #
    # # Compute confusion matrix (note: we need to transpose SKLearn matrix to make it match Edge Impulse)
    # cm = confusion_matrix(y_val, y_pred)
    # cm = np.transpose(cm)
    #
    # # Print confusion matrix
    # print()
    # print(" ---> Predicted labels")
    # print("|")
    # print("v Actual labels")
    # print("\t\t\t" + ' '.join("{!s:6}".format('(' + str(i) + ')') for i in range(2)))
    # for row in range(2):
    #     print("{:>12} ({}):  [{}]".format(labels[row], row, ' '.join("{:6}".format(i) for i in cm[row])))





def resize_images(images, width, height, anti_aliasing=True):
  """
  Prove a list of Numpy arrays (in images parameter) to have them all resized to desired height and
  width. Returns the list of newly resized image arrays.

  NOTE: skimage resize returns *normalized* image arrays (values between 0..1)
  """
  x_out = []
  for i, img in enumerate(images):
    x_out.append(resize(img, (height, width), anti_aliasing=anti_aliasing))
  return x_out

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
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

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layer.
        tf.keras.layers.Conv2D(
            8, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            4, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # NEW Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),


        # Add a hidden layer with dropout
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all NUM_CATEGORIES signs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
