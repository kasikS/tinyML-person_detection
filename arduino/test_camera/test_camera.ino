/*
  camera tests

*/

// #include <TinyMLShield.h>
#include "image.h"
#include "shield.h"

#include <process_image.h>
#include <Arduino_OV767X.h>


#define GRAY 1
#define IMAGE_SIZE  (176*144*1) 
#define IMAGE_SIZE_TRANSMIT  (176*144*1) // valid only for grayscale, to be changed for flexibility with rgb
#define IMAGE_SIZE_SINGLE  (176*144*1)
#define IMAGE_SIZE_SCALE (160*120*1)
#define IMAGE_SIZE_CROP (96*96*1)
#define IMAGE_SIZE_CROP_TRANSMIT (96*96*1)


#if GRAY == 1
  int image_format = GRAYSCALE;
#else
  int image_format = RGB565;
#endif


bool commandRecv = false; // flag used for indicating receipt of commands from serial port
bool liveFlag = true; // flag as true to live stream raw camera bytes, set as false to take single images on command
bool captureFlag = false;

// Image buffer;
byte image[IMAGE_SIZE];
// byte image_single[IMAGE_SIZE_SINGLE];
byte image_scale[IMAGE_SIZE_SCALE];
byte image_crop[IMAGE_SIZE_CROP];

int bytesPerFrame;
uint8_t score = 197;
uint8_t label = 0;


void setup() {
  Serial.begin(9600);
  while (!Serial);

  initializeShield();

  // Initialize the OV7675 camera
  if (!Camera.begin(QCIF, image_format, 1)) { //QCIF
    Serial.println("Failed to initialize camera");
    while (1);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  // for(int i=0; i< (96*96); i++){
  //   image_crop[i] = 122;
  // }
}

void loop() {
  int i = 0;

  bool clicked = readShieldButton();
  if (clicked) {
    if (!liveFlag) {
      if (!captureFlag) {
        captureFlag = true;
      }
    }
  }

  
  static Status status;


  if (liveFlag) {
 
    Camera.readFrame(image);

    // status = remove_byte(image, Camera.width(), Camera.height(), image_single);
    // if (status != OK) {
    //   Serial.println("removing byte error");
    //   return;
    // }

    // // Scale image
    status = scale(image, Camera.width(), Camera.height(), image_scale, 160, 120, 1);
    if (status != OK) {
      Serial.println("scaling error");
      return;
    }

    // Crop image to square
    status = crop_center(image_scale, 160, 120, image_crop, 96, 96, 1, false);
    if (status != OK) {
      Serial.println("cropping error");
      return;
    }

    Serial.write(&score, 1);
    Serial.write(&label,1);
    Serial.write(image_crop, IMAGE_SIZE_CROP_TRANSMIT); //send read image from camera
    // Serial.write(image, IMAGE_SIZE);
    // Serial.write(image, IMAGE_SIZE_TRANSMIT); //send read image from camera

    // send a known image
    // Serial.write(GRAY_IMAGE, bytesPerFrame);
  }
  else {
    if (captureFlag) {
      captureFlag = false;
      Camera.readFrame(image);
      Serial.write(image, bytesPerFrame); // send read image from camera
    }
  }
}