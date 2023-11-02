cmd_/module/examples/app_hello_world2/src/hello_world.o := arm-none-eabi-gcc -Wp,-MD,/module/examples/app_hello_world2/src/.hello_world.o.d     -I/module/examples/app_hello_world2/src -D__firmware__ -fno-exceptions -Wall -Wmissing-braces -fno-strict-aliasing -ffunction-sections -fdata-sections -Wdouble-promotion -std=gnu11 -DCRAZYFLIE_FW   -I/module/vendor/CMSIS/CMSIS/Core/Include   -I/module/vendor/CMSIS/CMSIS/DSP/Include   -I/module/vendor/libdw1000/inc   -I/module/vendor/FreeRTOS/include   -I/module/vendor/FreeRTOS/portable/GCC/ARM_CM4F   -I/module/src/config   -I/module/src/platform/interface   -I/module/src/deck/interface   -I/module/src/deck/drivers/interface   -I/module/src/drivers/interface   -I/module/src/drivers/bosch/interface   -I/module/src/drivers/esp32/interface   -I/module/src/hal/interface   -I/module/src/modules/interface   -I/module/src/modules/interface/kalman_core   -I/module/src/modules/interface/lighthouse   -I/module/src/modules/interface/outlierfilter   -I/module/src/modules/interface/cpx   -I/module/src/modules/interface/p2pDTR   -I/module/src/modules/interface/controller   -I/module/src/modules/interface/estimator   -I/module/src/utils/interface   -I/module/src/utils/interface/kve   -I/module/src/utils/interface/lighthouse   -I/module/src/utils/interface/tdoa   -I/module/src/lib/FatFS   -I/module/src/lib/CMSIS/STM32F4xx/Include   -I/module/src/lib/STM32_USB_Device_Library/Core/inc   -I/module/src/lib/STM32_USB_OTG_Driver/inc   -I/module/src/lib/STM32F4xx_StdPeriph_Driver/inc   -I/module/src/lib/vl53l1   -I/module/src/lib/vl53l1/core/inc   -I/module/examples/app_hello_world2/build/include/generated -fno-delete-null-pointer-checks -Wno-unused-but-set-variable -Wno-unused-const-variable -fomit-frame-pointer -fno-var-tracking-assignments -Wno-pointer-sign -fno-strict-overflow -fconserve-stack -Werror=implicit-int -Werror=date-time -DCC_HAVE_ASM_GOTO -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -g3 -fno-math-errno -DARM_MATH_CM4 -D__FPU_PRESENT=1 -mfp16-format=ieee -Wno-array-bounds -Wno-stringop-overread -Wno-stringop-overflow -DSTM32F4XX -DSTM32F40_41xxx -DHSE_VALUE=8000000 -DUSE_STDPERIPH_DRIVER -Os -Werror   -c -o /module/examples/app_hello_world2/src/hello_world.o /module/examples/app_hello_world2/src/hello_world.c

source_/module/examples/app_hello_world2/src/hello_world.o := /module/examples/app_hello_world2/src/hello_world.c

deps_/module/examples/app_hello_world2/src/hello_world.o := \
  /usr/include/newlib/string.h \
  /usr/include/newlib/_ansi.h \
  /usr/include/newlib/newlib.h \
  /usr/include/newlib/_newlib_version.h \
  /usr/include/newlib/sys/config.h \
    $(wildcard include/config/h//.h) \
  /usr/include/newlib/machine/ieeefp.h \
  /usr/include/newlib/sys/features.h \
  /usr/include/newlib/sys/reent.h \
  /usr/include/newlib/_ansi.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stddef.h \
  /usr/include/newlib/sys/_types.h \
  /usr/include/newlib/machine/_types.h \
  /usr/include/newlib/machine/_default_types.h \
  /usr/include/newlib/sys/lock.h \
  /usr/include/newlib/sys/cdefs.h \
  /usr/include/newlib/sys/_locale.h \
  /usr/include/newlib/strings.h \
  /usr/include/newlib/sys/string.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdint.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdbool.h \
  /module/src/modules/interface/app.h \
  /module/vendor/FreeRTOS/include/FreeRTOS.h \
  /module/src/config/FreeRTOSConfig.h \
    $(wildcard include/config/h.h) \
    $(wildcard include/config/debug/queue/monitor.h) \
  /module/src/config/config.h \
    $(wildcard include/config/h/.h) \
    $(wildcard include/config/block/address.h) \
  /module/src/drivers/interface/nrf24l01.h \
  /module/src/drivers/interface/nRF24L01reg.h \
  /module/src/config/trace.h \
  /module/src/hal/interface/usec_time.h \
  /module/src/utils/interface/cfassert.h \
  /module/vendor/FreeRTOS/include/projdefs.h \
  /module/vendor/FreeRTOS/include/portable.h \
  /module/vendor/FreeRTOS/include/deprecated_definitions.h \
  /module/vendor/FreeRTOS/portable/GCC/ARM_CM4F/portmacro.h \
  /module/vendor/FreeRTOS/include/mpu_wrappers.h \
  /module/vendor/FreeRTOS/include/task.h \
  /module/vendor/FreeRTOS/include/list.h \
  /module/src/utils/interface/debug.h \
    $(wildcard include/config/debug/print/on/uart1.h) \
  /module/src/config/config.h \
  /module/src/modules/interface/console.h \
  /module/src/utils/interface/eprintf.h \
  /usr/lib/gcc/arm-none-eabi/10.3.1/include/stdarg.h \

/module/examples/app_hello_world2/src/hello_world.o: $(deps_/module/examples/app_hello_world2/src/hello_world.o)

$(deps_/module/examples/app_hello_world2/src/hello_world.o):
