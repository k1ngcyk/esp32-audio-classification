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

### Partition

this partition table is used to load local audio data to the ESP32.

### Notes

audio data and arena is on psram.