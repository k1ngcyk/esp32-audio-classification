#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "classes.h"
#include "mel_spec.h"

#include "esp_heap_caps.h"
#include "esp_err.h"
#include <math.h>
#include <algorithm>
#include <memory.h>
#include "esp_attr.h"

#include "esp_timer.h"
#include "esp_spiffs.h"

static int16_t *audio_psram = nullptr;
static size_t audio_len = 0; // bytes

static void load_audio_to_psram()
{
  FILE* f = fopen("/spiffs/audio_data.raw", "r");
  if (f == NULL) {
    MicroPrintf("Failed to open audio.raw");
    return;
  }
  fseek(f, 0, SEEK_END);
  audio_len = ftell(f);
  fseek(f, 0, SEEK_SET);
  audio_psram = static_cast<int16_t *>(heap_caps_malloc(audio_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  fread(audio_psram, 1, audio_len, f);
  fclose(f);
  MicroPrintf("audio blob copied: %zu bytes at %p", audio_len, audio_psram);
}

constexpr int kAudioLength = 15600; // 0.975 seconds at 16kHz

// YAMNet model parameters
constexpr int kNumClasses = 521; // YAMNet has 521 AudioSet classes

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  int current_index = 0;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  const int kNumFrames = 96;
  const int kNumMelBins = 64;

  constexpr int kTensorArenaSize = 7168 * 1024;
  uint8_t *tensor_arena = nullptr;
}

static mel_ctx_t *g_mel_ctx = nullptr;

// The name of this function is important for Arduino compatibility.
void setup()
{
  MicroPrintf("Setup");
  esp_vfs_spiffs_conf_t conf = {
    .base_path = "/spiffs",
    .partition_label = "storage",
    .max_files = 1,
    .format_if_mount_failed = false
  };
  esp_err_t ret = esp_vfs_spiffs_register(&conf);

  if (ret != ESP_OK) {
      if (ret == ESP_FAIL) {
          MicroPrintf("Failed to mount or format filesystem");
      } else if (ret == ESP_ERR_NOT_FOUND) {
          MicroPrintf("Failed to find SPIFFS partition");
      } else {
          MicroPrintf("Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
      }
      return;
  }
  MicroPrintf("SPIFFS mounted");
  load_audio_to_psram();
  g_mel_ctx = mel_init(
    /*sample_rate*/ 16000,
    /*n_fft*/       512,
    /*win_length*/  400,
    /*hop_length*/  160,
    /*n_mels*/      64,
    /*fmin*/        125.0f,
    /*fmax*/        7500.0f
  );
  if (g_mel_ctx == nullptr)
  {
    MicroPrintf("Failed to initialize mel context\n");
    return;
  }
  MicroPrintf("Mel context initialized");

  tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr)
  {
    MicroPrintf("Failed to allocate memory for tensor arena\n");
    return;
  }
  MicroPrintf("Tensor arena allocated");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk)
  {
    return;
  }
  if (resolver.AddMean() != kTfLiteOk)
  {
    return;
  }
  if (resolver.AddLogistic() != kTfLiteOk)
  {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  MicroPrintf("Arena used: %zu / %zu bytes\n",
              interpreter->arena_used_bytes(), kTensorArenaSize);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  int16_t *temp_audio = nullptr;
  temp_audio = static_cast<int16_t *>(heap_caps_malloc(kAudioLength * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (current_index * kAudioLength + kAudioLength > audio_len / sizeof(int16_t))
  {
    current_index = 0;
    MicroPrintf("Out of audio data, resetting index to 0");
    return;
  }
  memcpy(temp_audio, &audio_psram[current_index * kAudioLength], kAudioLength * sizeof(int16_t));
  MicroPrintf("Starting calculation of mel spectrogram, index: %d", current_index);
  int64_t start = esp_timer_get_time();
  // Calculate mel spectrogram from audio data
  static int8_t* mel_spectrogram = static_cast<int8_t *>(heap_caps_malloc(kNumFrames * kNumMelBins, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (mel_spectrogram == nullptr)
  {
    MicroPrintf("Failed to allocate memory for mel spectrogram\n");
    return;
  }
  mel_compute_int8(g_mel_ctx, temp_audio, mel_spectrogram);
  int64_t end = esp_timer_get_time();
  MicroPrintf("Computed mel spectrogram: %dx%d\n", kNumFrames, kNumMelBins);
  for (int i = 0; i < 10; ++i)
  {
    printf("%d, ", static_cast<int>(mel_spectrogram[i]));
  }
  printf(" ... ");
  for (int i = 96*64 - 10; i < 96*64; ++i)
  {
    printf("%d, ", static_cast<int>(mel_spectrogram[i]));
  }
  printf("\n");
  MicroPrintf("Mel spectrogram time taken: %lld ms", (end - start) / 1000);

  // Verify input tensor dimensions match our mel spectrogram
  if (input->dims->data[1] != kNumFrames || input->dims->data[2] != kNumMelBins)
  {
    MicroPrintf("Warning: Input tensor shape mismatch. Expected %dx%d, got %dx%d\n",
                kNumFrames, kNumMelBins,
                input->dims->data[1], input->dims->data[2]);
  }

  // Copy mel spectrogram to model input tensor
  MicroPrintf("Copying mel spectrogram to model input tensor with size %d", input->bytes);
  memcpy(input->data.int8, mel_spectrogram, input->bytes);

  MicroPrintf("Running inference\n");
  start = esp_timer_get_time();
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    MicroPrintf("Invoke failed: %d\n", invoke_status);
    return;
  }
  end = esp_timer_get_time();
  MicroPrintf("Inference time taken: %lld ms", (end - start) / 1000);

  // Obtain the quantized output from model's output tensor
  int8_t *scores = output->data.int8;

  // Find the index of the highest score
  int max_index = 0;
  int8_t max_score = scores[0];

  for (int i = 1; i < kNumClasses; i++)
  {
    if (scores[i] > max_score)
    {
      max_score = scores[i];
      max_index = i;
    }
  }

  // Get the class name
  const char *predicted_class;
  if (max_index < kNumClasses)
  {
    predicted_class = kClassNames[max_index];
  }
  else
  {
    predicted_class = "Unknown class";
  }

  MicroPrintf("Predicted class: %s (index: %d, score: %d)\n",
              predicted_class, max_index, static_cast<int>(max_score));
  current_index++;
  if (current_index >= audio_len / kAudioLength)
  {
    current_index = 0;
  }
}
