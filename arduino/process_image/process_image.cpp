/*
 * A library to process images retrieved from a camera
 * based on edge impulse workshop for arduino
 */


#include "process_image.h"

// Function: crop an image, store in another buffer - consider in place?
Status crop_center(byte *in_pixels, unsigned int in_width, unsigned int in_height, byte *out_pixels, unsigned int out_width, unsigned int out_height, unsigned int bytes_per_pixel, bool cast_int) {
  unsigned int in_x_offset;
  unsigned int in_y_offset;
  unsigned int out_x_offset;
  unsigned int out_y_offset;
  int pixel;

  // Verify crop is smaller
  if ((in_width < out_width) || (in_height < out_height)) {
    return PI_ERROR;
  }

  // Calculate size of output image
  unsigned int out_buf_len = out_width * out_height;

  // Go through each row
  for (unsigned int y = 0; y < out_height; y++) {
    in_y_offset = bytes_per_pixel * in_width * \
                  ((in_height - out_height) / 2 + y);
    out_y_offset = bytes_per_pixel * out_width * y;

    // Go through each pixel in each row
    for (unsigned int x = 0; x < out_width; x++) {
      in_x_offset = bytes_per_pixel * ((in_width - out_width) / 2 + x);
      out_x_offset = bytes_per_pixel * x;

      // go through each byte in each pixel
      for (unsigned int b = 0; b < bytes_per_pixel; b++) {
      	pixel = in_pixels[in_y_offset + in_x_offset + b];
      	if (cast_int){ 
        	out_pixels[out_y_offset + out_x_offset + b] = static_cast<int8_t>(pixel-128);
        }else{
	        out_pixels[out_y_offset + out_x_offset + b] = pixel;
        }
      }
    }
  }

  return PI_OK;
}

// Function: scale image using nearest neighber - consider in place?
Status scale(byte *in_pixels, unsigned int in_width, unsigned int in_height, byte *out_pixels, unsigned int out_width, unsigned int out_height, unsigned int bytes_per_pixel) {
  unsigned int in_x_offset;
  unsigned int in_y_offset;
  unsigned int out_x_offset;
  unsigned int out_y_offset;
  unsigned int src_x;
  unsigned int src_y;

  // Compute ratio between input and output widths/heights (fixed point)
  unsigned long ratio_x = (in_width << 16) / out_width;
  unsigned long ratio_y = (in_height << 16) / out_height;

  // Loop through each row
  for (unsigned int y = 0; y < out_height; y++) {
    
    // Find which pixel to sample from original image in y direction
    src_y = (y * ratio_y) >> 16;
    src_y = (src_y < in_height) ? src_y : in_height - 1;

    // Calculate buffer offsets for y
    in_y_offset = bytes_per_pixel * in_width * src_y;
    out_y_offset = bytes_per_pixel * out_width * y;

    // Go through each pixel in each row
    for (unsigned int x = 0; x < out_width; x++) {
      // Find which pixel to sample from original image in x direction
      src_x = int(x * ratio_x) >> 16;
      src_x = (src_x < in_width) ? src_x : in_width -1;

      // Calculate buffer offsets for x
      in_x_offset = bytes_per_pixel * src_x;
      out_x_offset = bytes_per_pixel * x;

      // Copy pixels from source image to destination
      for (unsigned int b = 0; b < bytes_per_pixel; b++) {
        out_pixels[out_y_offset + out_x_offset + b] =
          in_pixels[in_y_offset + in_x_offset + b];
      }
    }
  }

  return PI_OK;
}

// OV767X library uses 2 bytes per pixel in grayscale, so just keep first pixel
 Status remove_byte(byte *in_pixels, unsigned int width, unsigned int height, byte *out_pixels, unsigned int bytes_per_pixel){
   for (unsigned int i = 0; i < (width * height); i++) {
     out_pixels[i] = in_pixels[bytes_per_pixel * i];
   }

   return PI_OK;
 }
 
float normalize(float x, float scale, float offset) {
  return (x * scale) - offset;
}

int8_t quantize(float x, float scale, float zero_point) {
  return (x / scale) + zero_point;
}
