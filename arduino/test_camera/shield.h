/*
 * Based on 
 * Active Learning Labs
 * Harvard University 
 *
 */

#ifndef SHIELD_H
#define SHIELD_H
#include <Arduino.h>

void nrf_gpio_cfg_out_with_input(uint32_t pin_number);
void initializeShield();
bool readShieldButton();


#endif