
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc model.cc
    PRIV_REQUIRES spi_flash esp_partition esp_psram
    INCLUDE_DIRS "")