
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc model.cc mel_spec.cc
    PRIV_REQUIRES spi_flash esp_psram esp_timer spiffs
    INCLUDE_DIRS "")

spiffs_create_partition_image(storage ../spiffs_image FLASH_IN_PROJECT)