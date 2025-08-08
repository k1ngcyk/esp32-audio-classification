#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "output_handler.h"

#include "esp_heap_caps.h"
#include "esp_dsp.h"
#include "esp_err.h"
#include <math.h>
#include <algorithm>
#include <memory.h>
#include "esp_attr.h"

#include "esp_partition.h"

static int16_t *audio_psram = nullptr;
static size_t audio_len = 0; // bytes

static void load_audio_to_psram()
{
  static constexpr esp_partition_subtype_t kSUBTYPE =
      static_cast<esp_partition_subtype_t>(0x40);
  const esp_partition_t *part =
      esp_partition_find_first(ESP_PARTITION_TYPE_DATA, kSUBTYPE, "audio");
  if (!part)
  {
    MicroPrintf("audio partition not found");
    abort();
  }

  audio_len = 31200;
  audio_psram = static_cast<int16_t *>(
      heap_caps_malloc(audio_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!audio_psram)
  {
    MicroPrintf("PSRAM alloc failed (%d B)", audio_len);
    abort();
  }

  ESP_ERROR_CHECK(esp_partition_read(part, 0, audio_psram, audio_len));

  MicroPrintf("audio blob copied: %zu bytes at %p", audio_len, audio_psram);
}

constexpr int kAudioLength = 15600; // 0.975 seconds at 16kHz

// YAMNet model parameters
constexpr int kNumClasses = 521; // YAMNet has 521 AudioSet classes

// Simplified class names for common audio events (first 10 most common classes)
// For the full 521 class names, you would need to include the complete AudioSet ontology
const char *kClassNames[10] = {
    "Speech",
    "Music",
    "Silence",
    "Human voice",
    "Singing",
    "Vehicle",
    "Animal",
    "Wind",
    "Water",
    "Other"};
constexpr int kDisplayedClasses = 10; // Number of class names we have defined

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;

  constexpr int kTensorArenaSize = 7168 * 1024;
  uint8_t *tensor_arena = nullptr;

  // ---------- YAMNet front-end parameters ----------
  constexpr int kSampleRate = 16'000;
  constexpr int kFrameLen = 400;  // 25 ms
  constexpr int kFrameStep = 160; // 10 ms
  constexpr int kFftLen = 512;
  constexpr int kNumFrames = 96; // (15600-400)/160 + 1
  constexpr int kNumMelBins = 64;
  constexpr float kFMin = 125.f;
  constexpr float kFMax = 7'500.f;

  // ---------- Helpers ----------
  inline float HertzToMel(float hz)
  {
    return 2595.f * std::log10f(1.f + hz / 700.f);
  }
  inline float MelToHertz(float mel)
  {
    return 700.f * (std::powf(10.f, mel / 2595.f) - 1.f);
  }

  // Pre-compute Hann window and Mel filter-bank ---------------------------
  struct FrontEndTables
  {
    std::vector<float> window;                   // 400 floats
    std::vector<float> w_fft;                    // sin/cos table
    std::vector<std::vector<float>> mel_filters; // 64 × 257

    FrontEndTables()
    {
      MicroPrintf("FrontEndTables constructor");
      // 1) Hann window (esp-dsp helper)
      window.resize(kFrameLen);
      dsps_wind_hann_f32(window.data(), kFrameLen);

      MicroPrintf("Hann window filled");
      // 2) FFT twiddle factors – allocate via esp-dsp
      w_fft.resize(kFftLen);
      dsps_fft2r_init_fc32(w_fft.data(), kFftLen);

      MicroPrintf("FFT twiddle factors allocated");

      // 3) Mel filter bank (triangular filters in linear freq domain)
      const int num_fft_bins = kFftLen / 2 + 1; // 257
      mel_filters.assign(kNumMelBins, std::vector<float>(num_fft_bins, 0.f));

      const float mel_min = HertzToMel(kFMin);
      const float mel_max = HertzToMel(kFMax);
      std::vector<float> mel_points(kNumMelBins + 2);
      for (int i = 0; i < mel_points.size(); ++i)
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (kNumMelBins + 1);

      std::vector<int> fft_bins(kNumMelBins + 2);
      for (int i = 0; i < fft_bins.size(); ++i)
        fft_bins[i] = static_cast<int>(std::floor((kFftLen)*MelToHertz(mel_points[i]) / kSampleRate));

      for (int m = 1; m <= kNumMelBins; ++m)
      {
        int f0 = fft_bins[m - 1], f1 = fft_bins[m], f2 = fft_bins[m + 1];
        for (int k = f0; k < f1; ++k)
          mel_filters[m - 1][k] =
              float(k - f0) / (f1 - f0);
        for (int k = f1; k < f2; ++k)
          mel_filters[m - 1][k] =
              float(f2 - k) / (f2 - f1);
      }
    }
  };

}

static FrontEndTables *g_frontend_tables = nullptr;
static float *g_fft_buf = nullptr;    // kFftLen * 2 floats
static float *g_power_spec = nullptr; // kFftLen / 2 + 1 floats
static float *g_mel_buf = nullptr;    // kNumFrames * kNumMelBins floats

void AudioToMelSpecInt8(const int16_t *in_audio,
                        int8_t out_mel[kNumFrames][kNumMelBins])
{
  if (g_frontend_tables == nullptr)
  {
    MicroPrintf("FrontEndTables not initialized!\n");
    return;
  }
  auto &tbl = *g_frontend_tables;

  // Use pre-allocated buffers in PSRAM for large arrays
  auto fft_buf = g_fft_buf;  // length kFftLen * 2
  auto power = g_power_spec; // length kFftLen / 2 + 1
  auto mel = reinterpret_cast<float (*)[kNumMelBins]>(g_mel_buf);

  // Track min/max for global linear → int8 scaling
  float global_min = 1e30f;
  float global_max = -1e30f;

  for (int frame = 0; frame < kNumFrames; ++frame)
  {
    // ---------- 1. Window & zero-pad ----------
    const int16_t *x = &in_audio[frame * kFrameStep];
    for (int i = 0; i < kFrameLen; ++i)
    {
      fft_buf[2 * i] = float(x[i]) * tbl.window[i]; // Re
      fft_buf[2 * i + 1] = 0.f;                     // Im
    }
    std::fill(&fft_buf[2 * kFrameLen], &fft_buf[2 * kFftLen], 0.f);

    // ---------- 2. FFT ----------
    dsps_fft2r_fc32(fft_buf, kFftLen); // :contentReference[oaicite:2]{index=2}
    dsps_bit_rev_fc32(fft_buf, kFftLen);
    dsps_cplx2reC_fc32(fft_buf, kFftLen); // convert packed real FFT

    // ---------- 3. Magnitude spectrum ----------
    power[0] = std::fabsf(fft_buf[0]); // DC
    for (int k = 1; k < kFftLen / 2; ++k)
    {
      float re = fft_buf[2 * k];
      float im = fft_buf[2 * k + 1];
      power[k] = std::sqrtf(re * re + im * im);
    }
    power[kFftLen / 2] = std::fabsf(fft_buf[1]); // Nyquist

    // ---------- 4. Mel projection ----------
    for (int m = 0; m < kNumMelBins; ++m)
    {
      float acc = 0.f;
      const auto &filter = tbl.mel_filters[m];
      for (int k = 0; k < kFftLen / 2 + 1; ++k)
        acc += power[k] * filter[k];
      // log-energy
      mel[frame][m] = std::logf(acc + 1e-6f);
      global_min = std::min(global_min, mel[frame][m]);
      global_max = std::max(global_max, mel[frame][m]);
    }
  }

  // ---------- 5. Linear->int8 quantisation ----------
  const float range = global_max - global_min + 1e-9f;
  for (int i = 0; i < kNumFrames; ++i)
    for (int j = 0; j < kNumMelBins; ++j)
      out_mel[i][j] = static_cast<int8_t>(
          (mel[i][j] - global_min) / range * 255.f - 128.f + 0.5f);
  MicroPrintf("Mel projection quantised");
}

// The name of this function is important for Arduino compatibility.
void setup()
{
  MicroPrintf("Setup");
  load_audio_to_psram();

  tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr)
  {
    MicroPrintf("Failed to allocate memory for tensor arena\n");
    return;
  }
  MicroPrintf("Tensor arena allocated");

  // Initialize front-end tables once (allocate in PSRAM to save internal DRAM)
  void *tbl_mem = heap_caps_malloc(sizeof(FrontEndTables), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tbl_mem == nullptr)
  {
    MicroPrintf("Failed to allocate memory for FrontEndTables\n");
    return;
  }
  g_frontend_tables = new (tbl_mem) FrontEndTables();
  MicroPrintf("FrontEndTables initialized (PSRAM)\n");

  // Allocate DSP working buffers in PSRAM
  size_t fft_buf_size = kFftLen * 2 * sizeof(float);
  size_t power_size = (kFftLen / 2 + 1) * sizeof(float);
  size_t mel_size = kNumFrames * kNumMelBins * sizeof(float);
  g_fft_buf = static_cast<float *>(heap_caps_malloc(fft_buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  g_power_spec = static_cast<float *>(heap_caps_malloc(power_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  g_mel_buf = static_cast<float *>(heap_caps_malloc(mel_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!g_fft_buf || !g_power_spec || !g_mel_buf)
  {
    MicroPrintf("Failed to allocate DSP buffers!\n");
    return;
  }

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
  MicroPrintf("Starting calculation of mel spectrogram");
  // Calculate mel spectrogram from audio data
  static int8_t mel_spectrogram[kNumFrames][kNumMelBins];
  AudioToMelSpecInt8(audio_psram, mel_spectrogram);
  MicroPrintf("Computed mel spectrogram: %dx%d\n", kNumFrames, kNumMelBins);

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
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    MicroPrintf("Invoke failed: %d\n", invoke_status);
    return;
  }

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
  if (max_index < kDisplayedClasses)
  {
    predicted_class = kClassNames[max_index];
  }
  else
  {
    predicted_class = "Unknown class";
  }

  MicroPrintf("Predicted class: %s (index: %d, score: %d)\n",
              predicted_class, max_index, static_cast<int>(max_score));
}
