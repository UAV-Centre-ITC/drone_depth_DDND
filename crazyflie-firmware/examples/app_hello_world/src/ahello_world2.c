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

//static CPXPacket_t rxPacket;

typedef struct
{
  unsigned char header;
  unsigned char TYPE;
  float vx;
  float vy;
  float vz;
  float yawrate;
} __attribute__((packed)) vel_message_t;

//void onCpxAppMessage(const CPXPacket_t* cpxRx){

//}

void ReceiveTask()
{
      //cpxInternalRouterReceiveCRTP(&rxPacket);
      //cpxGetRxPacket(&rxPacket);
      //vel_message_t * vel_message = (vel_message_t*) rxPacket.data;
      //cpxInternalRouterReceiveOthers(&rxPacket);
      //uint8_t datalen = rxPacket.dataLength;
      //float d1 = vel_message->vx;
      //float d2 = vel_message->vy;
      //float d3 = vel_message->yawrate;
      //if (vel_message->TYPE == 0x01) {
      //DEBUG_PRINT("received data length: %d, data[0]: %f, data[1]: %f, yawrate: %f.\n", datalen, (double)d1,(double)d2, (double)d3);
      //}


      DEBUG_PRINT("received data length: 1, data[0]: 2, data[1]: 3, yawrate: 4.\n");
      
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
  //cpxRegisterAppMessageHandler(onCpxAppMessage);
  DEBUG_PRINT("APP layer is listening...\n");
  //cpxPrintToConsole(LOG_TO_CRTP, "APP layer is listening...\n");
    while(1)
    {
        ReceiveTask();
        vTaskDelay(M2T(1000));  
    }
}//:
