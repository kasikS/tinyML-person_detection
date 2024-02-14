/*
  Based on Active Learning Labs
  Harvard University 
  tinyMLx - OV7675 Camera Test

*/

#include <TinyMLShield.h>
#include "image.h"

#define GRAY 1

#if GRAY == 1
  #define IMAGE_SIZE  (176*144*1)
  int image_format = GRAYSCALE;
#else
  #define IMAGE_SIZE  (176*144*2)
  int image_format = RGB565;
#endif


bool commandRecv = false; // flag used for indicating receipt of commands from serial port
bool liveFlag = false; // flag as true to live stream raw camera bytes, set as false to take single images on command
bool captureFlag = false;

// Image buffer;
byte image[IMAGE_SIZE]; // QCIF: 176x144 x 2 bytes per pixel (RGB565)
int bytesPerFrame;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  initializeShield();

  // Initialize the OV7675 camera
  if (!Camera.begin(QCIF, image_format, 1, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void loop() {
  int i = 0;
  String command;

  bool clicked = readShieldButton();
  if (clicked) {
    if (!liveFlag) {
      if (!captureFlag) {
        captureFlag = true;
      }
    }
  }

  // Read incoming commands from serial monitor
  while (Serial.available()) {
    char c = Serial.read();
    if ((c != '\n') && (c != '\r')) {
      command.concat(c);
    } 
    else if (c == '\r') {
      commandRecv = true;
      command.toLowerCase();
    }
  }

  // Command interpretation
  if (commandRecv) {
    commandRecv = false;
    if (command == "live") {
      Serial.println("\nRaw image data will begin streaming in 5 seconds...");
      liveFlag = true;
      delay(5000);
    }
    else if (command == "single") {
      Serial.println("\nCamera in single mode, type \"capture\" to initiate an image capture");
      liveFlag = false;
      delay(200);
    }
    else if (command == "capture") {
      if (!liveFlag) {
        if (!captureFlag) {
          captureFlag = true;
        }
        delay(200);
      }
      else {
        Serial.println("\nCamera is not in single mode, type \"single\" first");
        delay(1000);
      }
    }
  }
  
  if (liveFlag) {
    Camera.readFrame(image);
    Serial.write(image, bytesPerFrame); //send read image from camera

    // send a known image
    // Serial.write(GRAY_IMAGE, bytesPerFrame);
  }
  else {
    if (captureFlag) {
      captureFlag = false;
      Camera.readFrame(image);

      // test of grayscale calculation
      // for (int i = 0; i < 144; i++) {
      //   for(int j = 0; j < 176; j++){
      //     uint16_t pixel = image[176*i +j];

      //     int red   = ((pixel >> 11) & 0x1f) << 3;
      //     int green = ((pixel >> 5) & 0x3f) << 2; 
      //     int blue  = ((pixel >> 0) & 0x1f) << 3; 
          
      //     uint8_t grayscale4 = 0.2126 * red +0.7152 * green +0.0722 * blue;
      //     image[176*i +j] = grayscale4;
      //   }
      // }

      Serial.write(image, bytesPerFrame); // send read image from camera
    }
  }
}