#include "image.h"
#include <Arduino_OV767X.h>
#include <process_image.h>


#define IMAGE_SIZE  (176*144*1) // QCIF: 176x144 x 1 byte per pixel
#define IMAGE_SIZE_SCALE (117*96*1)
#define IMAGE_SIZE_CROP (96*96*1)

// Image buffer;
byte image[IMAGE_SIZE];
byte image_scale[IMAGE_SIZE_SCALE];
byte image_crop[IMAGE_SIZE_CROP];

int bytesPerFrame;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!Camera.begin(QCIF, GRAYSCALE, 1)){
    Serial.println("Failed to initialize camera");
    while (1);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  int bytes=0;
  // bytes = Camera.bytesPerPixel();
  // Serial.print("bytes per frame: \n");
  // Serial.print(bytes);

}

void loop() {
  int i = 0;
  String command;
  static Status status;


  Camera.readFrame(image);

    // // Scale image
    status = scale(image, Camera.width(), Camera.height(), image_scale, 117, 96, 1);
    if (status != PI_OK) {
      Serial.println("scaling error");
      return;
    }

    // Crop image to square
    status = crop_center(image_scale, 117, 96, image_crop, 96, 96, 1, false);
    if (status != PI_OK) {
      Serial.println("cropping error");
      return;
    }

  Serial.write(image_crop, IMAGE_SIZE_CROP); // send read image from camera

}