/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2019 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * hello_world.c - App layer application of a simple hello world debug print every
 *   2 seconds.
 */


#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"
#include "cpx_internal_router.h"
#include "cpx_external_router.h"
#include "cpx.h"

#define DEBUG_MODULE "HELLOWORLD"
#include "debug.h"

static CPXPacket_t rxPacket;

typedef struct
{
  unsigned char header;
  unsigned char TYPE;
  float vx;
  float vy;
  float vz;
  float yawrate;
} __attribute__((packed)) vel_message_t;

void onCpxAppMessage(const CPXPacket_t* cpxRx){
    vel_message_t * vel_message = (vel_message_t*) cpxRx->data;
    uint8_t datalen = cpxRx->dataLength;
      float d1 = vel_message->vx;
      float d2 = vel_message->vy;
      float d3 = vel_message->yawrate;
      //DEBUG_PRINT("message type: %c.\n", vel_message->TYPE);
      //DEBUG_PRINT("received data length: %d, data[0]: %f, data[1]: %f, yawrate: %f.\n", datalen, (double)d1,(double)d2, (double)d3);
      
      if (vel_message->TYPE == 0x01) {
      DEBUG_PRINT("received data length2: %d, data[0]: %f, data[1]: %f, yawrate: %f.\n", datalen, (double)d1,(double)d2, (double)d3);
      }
    

}

/*
void cpxGetRxPacket(CPXPacket_t* packet) {
  packet->route.destination=cpxRx.route.destination;
  packet->route.source=cpxRx.route.source;
  packet->route.lastPacket=cpxRx.route.lastPacket;
  packet->route.function=cpxRx.route.function;
  packet->route.version=cpxRx.route.version;
  packet->dataLength=cpxRx.dataLength;
  memcpy(packet->data, cpxRx.data, cpxRx.dataLength);
}
*/

void ReceiveTask()
{
      //cpxRegisterAppMessageHandler(cpxGetRxPacket);
      
      //cpxGetRxPacket(&rxPacket);
      vel_message_t * vel_message = (vel_message_t*) rxPacket.data;
      //cpxInternalRouterReceiveOthers(&rxPacket);
      uint8_t datalen = rxPacket.dataLength;
      float d1 = vel_message->vx;
      float d2 = vel_message->vy;
      float d3 = vel_message->yawrate;
      //DEBUG_PRINT("message type: %c.\n", vel_message->TYPE);
      DEBUG_PRINT("received data length: %d, data[0]: %f, data[1]: %f, yawrate: %f.\n", datalen, (double)d1,(double)d2, (double)d3);
      
      if (vel_message->TYPE == 0x01) {
      DEBUG_PRINT("received data length: %d, data[0]: %f, data[1]: %f, yawrate: %f.\n", datalen, (double)d1,(double)d2, (double)d3);
      }
      //CPXPacket_t rxPacket = cpxGetRxPacket();
      //cpxReceivePacketBlocking(CPX_F_APP, &packet);
      //uint8_t datalen = rxPacket.dataLength[0];
      //DEBUG_PRINT("received %d.\n", datalen);
      //cpxPrintToConsole(LOG_TO_CRTP, "received %d.\n", datalen);
      //pi_time_wait_us(2000 * 1000);
    
}


void appMain() 
{
  DEBUG_PRINT("Waiting for activation ...\n");
  vTaskDelay(M2T(20000));
  cpxInit();
  //cpxInternalRouterInit();
  //cpxExternalRouterInit();
  
  DEBUG_PRINT("APP layer is listening...\n");
  //cpxPrintToConsole(LOG_TO_CRTP, "APP layer is listening...\n");
    while(1)
    {
        cpxRegisterAppMessageHandler(onCpxAppMessage);
        //ReceiveTask();
        vTaskDelay(M2T(1000));  
    }
}//:
