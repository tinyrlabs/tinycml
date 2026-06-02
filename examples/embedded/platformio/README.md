# tinycml-embed PlatformIO Example

ESP32 + Arduino framework + KNN comfort predictor.

## Build & Upload

```bash
# Install PlatformIO Core
pip install platformio

# Build
cd examples/embedded/platformio
pio run

# Upload to ESP32
pio run --target upload

# Monitor serial output (115200 baud)
pio device monitor
```

## Project Structure

```
platformio/
  platformio.ini      # ESP32 config, build flags
  src/main.cpp        # KNN comfort predictor sketch
  lib/tinycml/
    tinycml.h         # Single-header ML library
    embed/
      cml_config.h    # Compile-time config
      cml_pool.h      # Static memory pool
```

## Configuration

Edit `platformio.ini` build_flags to change algorithm:

```ini
build_flags = -DCML_ENABLE_NAIVE_BAYES -DCML_USE_POOL -DCML_POOL_SIZE=4096
```

## Memory

Pool: 2048 bytes default (`-DCML_POOL_SIZE=N`).
KNN + core modules: ~4 KB Flash, <1 KB RAM.
