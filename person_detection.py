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
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split

EPOCHS = 100
NUM_CATEGORIES = 2
TEST_SIZE = 0.4
IMG_WIDTH = 96
IMG_HEIGHT = 96


def main():


    # Get image arrays and labels for all image files
    data_dir ='output' #'dataset'

    images, labels = load_data(data_dir, normalize=False, toint=True)
    plt.imshow(images[1], cmap='gray', vmin= -128, vmax=127)
    plt.show()

    labels = tf.keras.utils.to_categorical(labels)

    #split images to train, validation and test
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
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])  #do we need to add this softmax here as again?
    probability_model = model

    plt.imshow(x_test[0], cmap='gray', vmin=-128, vmax= 127)
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
    y_pred = model.predict(x_val)
    print("Validation output shape:", predictions_single.shape)

    # Convert actual and predicted validation one-hot encoding to numerical labels
    y_val = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Print some values from actual and predicted validation sets
    print("Actual validation labels:\t", y_val[:20])
    print("Predicted validation labels:\t", y_pred[:20])

    # Compute confusion matrix (note: we need to transpose SKLearn matrix to make it match Edge Impulse)
    cm = confusion_matrix(y_val, y_pred)
    # cm = np.transpose(cm)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['person', 'Not person'],
                yticklabels=['person', 'Not person'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()


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

#moved to module, replace
def load_data(data_dir, normalize = False, toint = False):
    """
    Loads image data from directory `data_dir`.

    `data_dir` has one directory named after each category, numbered
    0 and 1
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

def get_model():
    """
    Returns a compiled convolutional neural network model. .
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

    # lr = 0.0005
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
