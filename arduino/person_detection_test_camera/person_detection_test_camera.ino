#include <Arduino_OV767X.h>
#include "Arduino.h"
#include <process_image.h> // in libraries
// #include "process_image.h" // local, next to sketch

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr; 
tflite::MicroInterpreter* interpreter = nullptr; 
TfLiteTensor* input = nullptr;  

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 140 * 1024; 
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;
uint8_t score;
uint8_t label;


#define IMAGE_SIZE  (176*144*1)
#define IMAGE_CROPPED_SIZE (96*96)
#define IMAGE_SIZE_SCALE (160*120*1)


byte image[IMAGE_SIZE];
byte image_crop[IMAGE_CROPPED_SIZE];
byte image_scale[IMAGE_SIZE_SCALE];

int bytesPerFrame;

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

   model = tflite::GetModel(g_person_detect_model_data);
   if (model->version() != TFLITE_SCHEMA_VERSION) {
      MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

// Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

//needed on top of the example!
  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(interpreter->input(0)->quantization.params);

  // Get the quantization parameters (per-tensor quantization)
  tflu_scale = i_quantization->scale->data[0];
  tflu_zeropoint = i_quantization->zero_point->data[0];
}




void loop() {
  static Status status;


  // Read camera data
  Camera.readFrame(image);
  int index = 0;

  status = scale(image, Camera.width(), Camera.height(), image_scale, 160, 120, 1);
  if (status != PI_OK) {
    Serial.println("scaling error");
    return;
  }

  // Crop image to square
  status = crop_center(image_scale, 160, 120, image_crop, 96, 96, 1, false);
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
      pixel_g = image_crop[(y * 96) + x];
      pixel_f = ((float)pixel_g/127.5f) - 1.f;
      pixel_q = quantize(pixel_f, tflu_scale, tflu_zeropoint);
      interpreter->input(0)->data.int8[index++] = static_cast<int8_t>(pixel_q);
    }
  }


  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }


  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[1];
  int8_t no_person_score = output->data.uint8[0];
  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float no_person_score_f =
      (no_person_score - output->params.zero_point) * output->params.scale;

 int16_t res=0;
 if (person_score > no_person_score) {
    label = 1;
    res = person_score_f * 100;
    score = static_cast<uint8_t>(res);
  }else{
    label = 0;
    res = no_person_score_f *100;
    score = static_cast<uint8_t>(res);
  }
  

  Serial.write(&score, 1);
  Serial.write(&label,1);
  Serial.write(image_crop, IMAGE_CROPPED_SIZE); //send read image from camera


}
