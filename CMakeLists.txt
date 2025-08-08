cmake_minimum_required(VERSION 3.5)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hello_world)

# ─── Add the extra image ─────────────────────────────────────────────
set(BLOB_OFFSET 0x610000)                     # must match partitions.csv
set(BLOB_FILE   ${CMAKE_SOURCE_DIR}/audio_data.raw)

# Keep a copy in the build dir so 'cmake --build' has a dependency it can track
configure_file(${BLOB_FILE} ${CMAKE_BINARY_DIR}/audio_data.raw COPYONLY)

set(FLASH_EXTRA_IMAGES
    ${BLOB_OFFSET} ${CMAKE_BINARY_DIR}/audio_data.raw
)
