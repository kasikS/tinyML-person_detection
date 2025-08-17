// #include <TinyMLShield.h>
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// arduino nano ble sense - camera not great, but in general works nice

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "Arduino.h"
#include "image.h"
#include "model.h"
#include <process_image.h>
#include <Arduino_OV767X.h>


#define IMAGE_SIZE  (176*144*1)
#define IMAGE_CROPPED_SIZE (96*96)
#define IMAGE_SIZE_SCALE (160*120*1)

byte test = 150;

byte image[IMAGE_SIZE];
byte image_cropped[IMAGE_CROPPED_SIZE];
byte image_scale[IMAGE_SIZE_SCALE];
int bytesPerFrame;

const tflite::Model*  tflModel = nullptr; 
// tflite::ErrorReporter*  tflErrorReporter = nullptr; 
TfLiteTensor* tflInputTensor = nullptr;  
TfLiteTensor* tflOutputTensor = nullptr; 
tflite::MicroInterpreter* tflInterpreter = nullptr; 

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.


constexpr int tensorArenaSize = 140 * 1024; 
// Keep aligned to 16 bytes for CMSIS
// alignas(16) uint8_t tensor_arena[tensorArenaSize];
uint8_t tensorArena[tensorArenaSize];
static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;

long int t1;
long int t2;
uint8_t score;
uint8_t label;
static Status status;


void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!Camera.begin(QCIF, GRAYSCALE, 5)) { 
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  // static tflite::MicroErrorReporter micro_error_reporter; 
  // tflErrorReporter = &micro_error_reporter;

   tflModel = tflite::GetModel(model);
   if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
  //  TF_LITE_REPORT_ERROR(tflErrorReporter,
  //       "Model provided is schema version %d not equal "
  //       "to supported version %d.",
      MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        tflModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // static tflite::MicroInterpreter static_interpreter(tflModel, micro_op_resolver, tensorArena, tensorArenaSize, tflErrorReporter);
  static tflite::MicroInterpreter static_interpreter(tflModel, micro_op_resolver, tensorArena, tensorArenaSize);
  tflInterpreter = &static_interpreter;

  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    // TF_LITE_REPORT_ERROR(tflErrorReporter, "AllocateTensors() failed");
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflInterpreter->input(0)->quantization.params);

  // Get the quantization parameters (per-tensor quantization)
  tflu_scale = i_quantization->scale->data[0];
  tflu_zeropoint = i_quantization->zero_point->data[0];
}

// void test_func(){
//   for(int i = 0; i < 96; i++){
//     for(int j =0; j < 96; j++){
//       tflInterpreter->input(0)->data.f[96*i+j] = bcg[96*i+j];
//     } 
//   }

//    for(int i = 0; i < 96; i++){
//     for(int j =0; j < 96; j++){
//       Serial.print(tflInterpreter->input(0)->data.f[96*i+j]);
//       Serial.print(", ");
//     } 
//       Serial.println("");
//   }
//  Serial.println("");
//   for(int i = 0; i < 96; i++){
//     for(int j =0; j < 96; j++){
//       Serial.print(image[96*i+j]);
//       Serial.print(", ");
//     } 
//       Serial.println("");
//   }
//   Serial.println("");
//   for(int i = 0; i < 96; i++){
//     for(int j =0; j < 96; j++){
//       Serial.print(tflInterpreter->input(0)->data.f[96*i+j]-image[96*i+j]);
//       Serial.print(", ");
//     } 
//       Serial.println("");
//   }
// }



void loop() {
  // Read camera data
  Camera.readFrame(image);
  t1 = millis();
  int index = 0;
  // int pixel=0;

  status = scale(image, Camera.width(), Camera.height(), image_scale, 160, 120, 1);
  if (status != PI_OK) {
    Serial.println("scaling error");
    return;
  }

  // Crop image to square
  status = crop_center(image_scale, 160, 120, image_cropped, 96, 96, 1, false);
  if (status != PI_OK) {
    Serial.println("cropping error");
    return;
  }

  float pixel_f;
  uint8_t pixel_g;
  int8_t pixel =0;
  int8_t pixel_q;

  // set input tensor 
  for (int y = 0; y < 96; y++) {
    for (int x = 0; x < 96; x++) {
        // pixel = static_cast<int8_t>(image_cropped[(y * 96) + x]-128);
      pixel_g = image_cropped[(y * 96) + x];
      pixel_f = ((float)pixel_g/127.5f) - 1.f;
        // pixel_f = normalize((float)pixel, 1.f/127.5f, -1.f);
      pixel_q = quantize(pixel_f, tflu_scale, tflu_zeropoint);
      tflInterpreter->input(0)->data.int8[index++] = static_cast<int8_t>(pixel_q);
    }
  }

  t2 = millis();
  // Serial.print("Time taken by readout and data curation: "); Serial.print(t2-t1); Serial.println(" ms");


  t1 = millis();
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (kTfLiteOk != invokeStatus) {
    // TF_LITE_REPORT_ERROR(tflErrorReporter, "Invoke failed.");
      MicroPrintf("Invoke failed.");
  }
  t2 = millis();
  // Serial.print("Time taken by inference: "); Serial.print(t2-t1); Serial.println(" ms");

 // Process the inference results.
  int8_t person_score = tflInterpreter->output(0)->data.int8[1];
  int8_t no_person_score = tflInterpreter->output(0)->data.int8[0];

  int16_t res=0;

 if (person_score > no_person_score) {
    label = 1;
    res = ((person_score+128) * 100)/255;
    score = static_cast<uint8_t>(res);
    // score = static_cast<uint8_t>(person_score+128);
  }else{
    label = 0;
    res = ((no_person_score+128) * 100)/255;
    score = static_cast<uint8_t>(res);
    // score = static_cast<uint8_t>(no_person_score+128);
  }
  


  //send prediction-score and label, send the image
  Serial.write(&score, 1);
  Serial.write(&label,1);
  Serial.write(image_cropped, IMAGE_CROPPED_SIZE); //send read image from camera
  // Serial.write(image, IMAGE_SIZE); //send read image from camera

 

static bool is_initialized = false; //is this here needed every time??
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.
  // Switch the person/not person LEDs off
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDB, HIGH); //blue off

  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (label) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }

  // TF_LITE_REPORT_ERROR(tflErrorReporter, "Person score: %d No person score: %d",
  //                      person_score, no_person_score);

  // delay(3000);

}
