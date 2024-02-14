#include <TinyMLShield.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "Arduino.h"
#include "image.h"
#include "model.h"

#define IMAGE_SIZE  (176*144*1)

byte image[IMAGE_SIZE];
int bytesPerFrame;

const tflite::Model*  tflModel = nullptr; 
tflite::ErrorReporter*  tflErrorReporter = nullptr; 
TfLiteTensor* tflInputTensor = nullptr;  
TfLiteTensor* tflOutputTensor = nullptr; 
tflite::MicroInterpreter* tflInterpreter = nullptr; 

constexpr int tensorArenaSize = 140 * 1024; 
uint8_t tensorArena[tensorArenaSize];

long int t1;
long int t2;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println();

  if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) { 
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  static tflite::MicroErrorReporter micro_error_reporter; 
  tflErrorReporter = &micro_error_reporter;

   tflModel = tflite::GetModel(model);
   if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
   TF_LITE_REPORT_ERROR(tflErrorReporter,
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

  static tflite::MicroInterpreter static_interpreter(tflModel, micro_op_resolver, tensorArena, tensorArenaSize, tflErrorReporter);
  tflInterpreter = &static_interpreter;

  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(tflErrorReporter, "AllocateTensors() failed");
    return;
  }
  tflInputTensor = tflInterpreter->input(0);
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
  int min_x = (176 - 96) / 2;
  int min_y = (144 - 96) / 2;
  t1 = millis();
  int index = 0;

  // Crop 96x96 image. This lowers FOV, ideally we would downsample but this is simpler. 
  for (int y = min_y; y < min_y + 96; y++) {
    for (int x = min_x; x < min_x + 96; x++) {
      // tflInterpreter->input(0)->data.int8[index++] = static_cast<int8_t>(image[(y * 176) + x]); // - 128);
      tflInterpreter->input(0)->data.int8[index++] = static_cast<int8_t>(image[(y * 176) + x]); // - 128);

    }
  }
  t2 = millis();
  Serial.print("Time taken by readout and data curation: "); Serial.print(t2-t1); Serial.println(" milliseconds");


  t1 = millis();
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (kTfLiteOk != invokeStatus) {
    TF_LITE_REPORT_ERROR(tflErrorReporter, "Invoke failed.");}
  t2 = millis();
   Serial.print("Time taken by inference: "); Serial.print(t2-t1); Serial.println(" milliseconds");

 // Process the inference results.
  int8_t person_score = tflInterpreter->output(0)->data.int8[1];
  int8_t no_person_score = tflInterpreter->output(0)->data.int8[0];


static bool is_initialized = false;
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
  if (person_score > no_person_score) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }

  TF_LITE_REPORT_ERROR(tflErrorReporter, "Person score: %d No person score: %d",
                       person_score, no_person_score);

  // delay(3000);

}
