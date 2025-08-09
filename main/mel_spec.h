#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    // Config
    int sample_rate; // 16000
    int n_fft;       // 512
    int win_length;  // 400
    int hop_length;  // 160
    int n_mels;      // 64
    float fmin;      // 125.0f
    float fmax;      // 7500.0f
    int frames_out;  // 96 (final)
    int frames_in;   // 98 (from librosa-style center=True padding)

    // Reusable data
    float *hann_win; // [win_length], periodic Hann
    float *win_full; // [n_fft], hann centered in n_fft
    float *mel_fb;   // [n_mels * (n_fft/2+1)], Slaney normalized triangles

    // Work buffers
    float *y_padded; // [15600 + 2*(n_fft/2)]
    float *fft_buf;  // [n_fft*2] complex interleaved
    float *mag_buf;  // [(n_fft/2+1)]
    float *mel_row;  // [n_mels]
    float *mel_in;   // [frames_in * n_mels]  (98 x 64)
    float *mel_out;  // [frames_out * n_mels] (96 x 64) after time-resample

    // Flags
    bool dsp_ready;
  } mel_ctx_t;

  // Create reusable context (allocates PSRAM buffers when possible). Returns NULL on failure.
  mel_ctx_t *mel_init(int sample_rate, int n_fft, int win_length, int hop_length,
                      int n_mels, float fmin, float fmax);

  // Compute int8 log-mel for one 0.975s window (15600 int16 samples).
  // Output: time-major [96,64] into out_int8.
  bool mel_compute_int8(mel_ctx_t *ctx, const int16_t *pcm15600, int8_t *out_int8);

  // Free all buffers.
  void mel_free(mel_ctx_t *ctx);

#ifdef __cplusplus
}
#endif
