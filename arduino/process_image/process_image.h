#ifndef PROCESS_IMAGE_H
#define PROCESS_IMAGE_H
#include <Arduino.h>

typedef enum {
  PI_OK = 0,
  PI_ERROR = 1
} Status;

Status crop_center(byte *in_pixels, unsigned int in_width, unsigned int in_height, byte *out_pixels, unsigned int out_width, unsigned int out_height, unsigned int bytes_per_pixel, bool cast_int);

Status scale(byte *in_pixels, unsigned int in_width, unsigned int in_height, byte *out_pixels, unsigned int out_width, unsigned int out_height, unsigned int bytes_per_pixel);

Status remove_byte(byte *in_pixels, unsigned int width, unsigned int height, byte *out_pixels, unsigned int bytes_per_pixel);

float normalize(float x, float scale, float offset);

int8_t quantize(float x, float scale, float zero_point);

#endif
