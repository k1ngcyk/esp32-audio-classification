# ESP32 Audio Classification Example

## Requirements

- ESP-IDF

### Building the example

Set the chip target (For esp32s3 target, IDF version `release/v4.4` is needed):

```
idf.py set-target esp32s3
```

Then build with `idf.py`
```
idf.py build
```

### Adjust ESP NN and rebuild

Modify `managed_components/espressif__esp-tflite-micro/CMakeLists.txt` to adjust the ESP NN kernels.

```cmake
# remove sources which will be provided by esp_nn
list(REMOVE_ITEM srcs_kernels
          "${tfmicro_kernels_dir}/add.cc"
          "${tfmicro_kernels_dir}/conv.cc"
          "${tfmicro_kernels_dir}/depthwise_conv.cc"
          # "${tfmicro_kernels_dir}/fully_connected.cc"
          "${tfmicro_kernels_dir}/mul.cc"
          "${tfmicro_kernels_dir}/pooling.cc"
          "${tfmicro_kernels_dir}/softmax.cc")

FILE(GLOB esp_nn_kernels
          "${tfmicro_kernels_dir}/esp_nn/add.cc"
          "${tfmicro_kernels_dir}/esp_nn/conv.cc"
          "${tfmicro_kernels_dir}/esp_nn/depthwise_conv.cc"
          # "${tfmicro_kernels_dir}/esp_nn/fully_connected.cc"
          "${tfmicro_kernels_dir}/esp_nn/mul.cc"
          "${tfmicro_kernels_dir}/esp_nn/pooling.cc"
          "${tfmicro_kernels_dir}/esp_nn/softmax.cc")
```

### Partition

this partition table is used to load local audio data to the ESP32.

### Notes

audio data and arena is on psram.