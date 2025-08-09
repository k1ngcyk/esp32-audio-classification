#include "mel_spec.h"
#include <math.h>
#include <string.h>
#include <algorithm>

#include "esp_log.h"
#include "esp_heap_caps.h"

// ESP-DSP (FFT + helpers)
#include "esp_dsp.h"

static const char* TAG = "mel_spec";

// ---------- Math helpers (Slaney-style, matches process.py) ----------
static inline float hz_to_mel(float f_hz) {
  const float f_sp   = 200.0f / 3.0f;  // 66.666...
  const float brkfrq = 1000.0f;
  const float brkpt  = brkfrq / f_sp;
  const float logstep = logf(6.4f) / 27.0f;
  if (f_hz < brkfrq) {
    return f_hz / f_sp;
  }
  return brkpt + logf(f_hz / brkfrq) / logstep;
}
static inline float mel_to_hz(float mels) {
  const float f_sp   = 200.0f / 3.0f;
  const float brkfrq = 1000.0f;
  const float brkpt  = brkfrq / f_sp;
  const float logstep = logf(6.4f) / 27.0f;
  if (mels < brkpt) {
    return mels * f_sp;
  }
  return brkfrq * expf(logstep * (mels - brkpt));
}

static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ---------- Buffer helpers ----------
static void* psram_calloc(size_t count, size_t size) {
  return heap_caps_calloc(count, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}
static void* psram_malloc(size_t size) {
  return heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

// ---------- Build numpy.hanning (symmetric) of length win_length ----------
static bool build_hann_periodic(int win_length, float* out) {
  if (!out || win_length <= 0) return false;
  if (win_length == 1) { out[0] = 1.0f; return true; }
  const float denom = (float)(win_length - 1);
  for (int n = 0; n < win_length; ++n) {
    out[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / denom);
  }
  return true;
}

// ---------- Build win_full[n_fft] with hann centered ----------
static bool build_centered_window(int n_fft, int win_length, const float* hann, float* win_full) {
  if (!hann || !win_full) return false;
  memset(win_full, 0, sizeof(float) * n_fft);
  const int off = (n_fft - win_length) / 2; // 56 for 512-400
  for (int i = 0; i < win_length; ++i) {
    win_full[off + i] = hann[i];
  }
  return true;
}

// ---------- Build Slaney mel filterbank, norm='slaney', htk=False ----------
static float* build_mel_filterbank(int sr, int n_fft, int n_mels, float fmin, float fmax) {
  const int nbins = n_fft / 2 + 1;
  float* fb = (float*)psram_calloc(n_mels * nbins, sizeof(float));
  if (!fb) return nullptr;

  // Bin center freqs
  float* freqs = (float*)psram_malloc(nbins * sizeof(float));
  if (!freqs) { free(fb); return nullptr; }
  for (int i = 0; i < nbins; ++i) freqs[i] = (float)i * (float)sr / (float)n_fft;

  // Mel breakpoints
  float mel_lo = hz_to_mel(fmin);
  float mel_hi = hz_to_mel(fmax);
  const int npts = n_mels + 2;

  float* mpts = (float*)psram_malloc(npts * sizeof(float));
  float* hpts = (float*)psram_malloc(npts * sizeof(float));
  if (!mpts || !hpts) { free(freqs); free(fb); if (mpts) free(mpts); if (hpts) free(hpts); return nullptr; }
  for (int i = 0; i < npts; ++i) mpts[i] = mel_lo + (mel_hi - mel_lo) * (float)i / (float)(npts - 1);
  for (int i = 0; i < npts; ++i) hpts[i] = mel_to_hz(mpts[i]);

  // Triangles + Slaney area norm
  for (int m = 1; m <= n_mels; ++m) {
    float fL = hpts[m - 1], fC = hpts[m], fR = hpts[m + 1];
    const float width_hz = fR - fL;
    const float scale = (width_hz > 0.f) ? (2.0f / width_hz) : 1.0f;

    float* row = &fb[(m - 1) * nbins];
    for (int i = 0; i < nbins; ++i) {
      float f = freqs[i];
      float w = 0.f;
      if (f >= fL && f <= fC) {
        w = (f - fL) / fmaxf(1e-9f, (fC - fL));
      } else if (f > fC && f <= fR) {
        w = (fR - f) / fmaxf(1e-9f, (fR - fC));
      }
      row[i] = w * scale; // Slaney normalization
    }
  }

  free(freqs); free(mpts); free(hpts);
  return fb;
}

// ---------- librosa-style reflect padding by n_fft/2 with int16->[-1,1] scaling ----------
static void reflect_pad_center(const int16_t* pcm, int len, int pad, float* y) {
  // y size must be len + 2*pad
  const float scale = 1.0f / 32767.0f;
  // left reflect (exclude edge)
  for (int i = 0; i < pad; ++i) {
    int src = i + 1;
    if (src >= len) src = len - 1;
    float v = (float)pcm[src] * scale;
    y[pad - 1 - i] = clampf(v, -1.0f, 1.0f);
  }
  // middle
  for (int i = 0; i < len; ++i) {
    float v = (float)pcm[i] * scale;
    y[pad + i] = clampf(v, -1.0f, 1.0f);
  }
  // right reflect (exclude edge)
  for (int i = 0; i < pad; ++i) {
    int src = len - 2 - i;
    if (src < 0) src = 0;
    float v = (float)pcm[src] * scale;
    y[pad + len + i] = clampf(v, -1.0f, 1.0f);
  }
}

// ---------- Time linear resample [Tin x F] -> [Tout x F] ----------
static void time_resample_linear(const float* src, int Tin, int F, int Tout, float* dst) {
  if (Tin == Tout) {
    memcpy(dst, src, sizeof(float) * Tin * F);
    return;
  }
  for (int m = 0; m < F; ++m) {
    for (int t = 0; t < Tout; ++t) {
      float u = (Tout == 1) ? 0.f : (float)t / (float)(Tout - 1);       // 0..1
      float pos = u * (float)(Tin - 1);                                 // 0..Tin-1
      int t0 = (int)floorf(pos);
      int t1 = (t0 < Tin - 1) ? (t0 + 1) : t0;
      float a = pos - (float)t0;
      float v0 = src[t0 * F + m];
      float v1 = src[t1 * F + m];
      dst[t * F + m] = v0 + a * (v1 - v0);
    }
  }
}

// ---------- Public API ----------
mel_ctx_t* mel_init(int sample_rate, int n_fft, int win_length, int hop_length,
                    int n_mels, float fmin, float fmax) {
  mel_ctx_t* ctx = (mel_ctx_t*)heap_caps_calloc(1, sizeof(mel_ctx_t), MALLOC_CAP_8BIT);
  if (!ctx) return nullptr;

  ctx->sample_rate = sample_rate;
  ctx->n_fft       = n_fft;
  ctx->win_length  = win_length;
  ctx->hop_length  = hop_length;
  ctx->n_mels      = n_mels;
  ctx->fmin        = fmin;
  ctx->fmax        = fmax;
  ctx->frames_out  = 96;
  const int pad = n_fft / 2;                         // 256
  const int padded = 15600 + 2 * pad;                // 15600 + 512
  ctx->frames_in  = 1 + ( (padded - n_fft) / hop_length ); // 98

  // Init FFT tables
  if (dsps_fft2r_init_fc32(NULL, n_fft) != ESP_OK) { free(ctx); return nullptr; }
  ctx->dsp_ready = true;

  // Windows
  ctx->hann_win = (float*)psram_malloc(sizeof(float) * win_length);
  ctx->win_full = (float*)psram_malloc(sizeof(float) * n_fft);
  if (!ctx->hann_win || !ctx->win_full) { mel_free(ctx); return nullptr; }
  if (!build_hann_periodic(win_length, ctx->hann_win)) { mel_free(ctx); return nullptr; }
  if (!build_centered_window(n_fft, win_length, ctx->hann_win, ctx->win_full)) { mel_free(ctx); return nullptr; }

  // Mel filterbank
  ctx->mel_fb = build_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax);
  if (!ctx->mel_fb) { mel_free(ctx); return nullptr; }

  // Work buffers
  ctx->y_padded = (float*)psram_malloc(sizeof(float) * (15600 + 2 * pad));
  ctx->fft_buf  = (float*)psram_calloc(n_fft * 2, sizeof(float));
  ctx->mag_buf  = (float*)psram_malloc(sizeof(float) * (n_fft/2 + 1));
  ctx->mel_row  = (float*)psram_malloc(sizeof(float) * n_mels);
  ctx->mel_in   = (float*)psram_malloc(sizeof(float) * ctx->frames_in  * n_mels);
  ctx->mel_out  = (float*)psram_malloc(sizeof(float) * ctx->frames_out * n_mels);

  if (!ctx->y_padded || !ctx->fft_buf || !ctx->mag_buf ||
      !ctx->mel_row || !ctx->mel_in || !ctx->mel_out) {
    mel_free(ctx);
    return nullptr;
  }

  return ctx;
}

bool mel_compute_int8(mel_ctx_t* ctx, const int16_t* pcm15600, int8_t* out_int8) {
  if (!ctx || !pcm15600 || !out_int8) return false;

  const int pad = ctx->n_fft / 2;
  const int nbins = ctx->n_fft / 2 + 1;
  const float eps = 1e-6f;

  // 1) librosa-style reflect padding by n_fft/2 (center=True)
  reflect_pad_center(pcm15600, 15600, pad, ctx->y_padded);

  // 2) STFT frames over padded signal, each frame length = n_fft
  //    Multiply by win_full (n_fft with centered Hann); zero-imag; FFT; magnitude
  for (int f = 0; f < ctx->frames_in; ++f) {
    const int start = f * ctx->hop_length;

    // Build real frame into fft_buf (interleaved complex)
    // Re[k] = y_padded[start + k] * win_full[k], Im[k]=0
    float* X = ctx->fft_buf;
    for (int k = 0; k < ctx->n_fft; ++k) {
      float v = ctx->y_padded[start + k] * ctx->win_full[k];
      X[2*k + 0] = v;
      X[2*k + 1] = 0.f;
    }

    dsps_fft2r_fc32(X, ctx->n_fft);
    dsps_bit_rev2r_fc32(X, ctx->n_fft);
    dsps_cplx2reC_fc32(X, ctx->n_fft);

    // Magnitude spectrum [0 .. n_fft/2]
    ctx->mag_buf[0] = fabsf(X[0]);                // DC real
    ctx->mag_buf[nbins - 1] = fabsf(X[1]);        // Nyquist real
    for (int k = 1; k < nbins - 1; ++k) {
      float re = X[2*k + 0];
      float im = X[2*k + 1];
      ctx->mag_buf[k] = sqrtf(re*re + im*im);
    }

    // 3) mel projection + log
    for (int m = 0; m < ctx->n_mels; ++m) {
      const float* w = &ctx->mel_fb[m * nbins];
      float acc = 0.f;
      for (int i = 0; i < nbins; ++i) acc += w[i] * ctx->mag_buf[i];
      ctx->mel_row[m] = logf(acc + eps);
    }

    memcpy(&ctx->mel_in[f * ctx->n_mels], ctx->mel_row, sizeof(float) * ctx->n_mels);
  }

  // 4) Time-resample 98 -> 96 (linear, like your SciPy)
  time_resample_linear(ctx->mel_in, ctx->frames_in, ctx->n_mels, ctx->frames_out, ctx->mel_out);

  // 5) Per-spectrogram min-max -> int8: ((x-min)/(max-min))*255 - 128
  const int total = ctx->frames_out * ctx->n_mels;
  float vmin = ctx->mel_out[0], vmax = ctx->mel_out[0];
  for (int i = 1; i < total; ++i) {
    vmin = fminf(vmin, ctx->mel_out[i]);
    vmax = fmaxf(vmax, ctx->mel_out[i]);
  }
  const float scale = (vmax > vmin) ? (255.f / (vmax - vmin)) : 1.f;

  for (int i = 0; i < total; ++i) {
    float y = (ctx->mel_out[i] - vmin) * scale - 128.f;
    int v = (int)lrintf(y);
    v = (v < -128) ? -128 : (v > 127 ? 127 : v);
    out_int8[i] = (int8_t)v;
  }
  return true;
}

void mel_free(mel_ctx_t* ctx) {
  if (!ctx) return;
  if (ctx->hann_win) free(ctx->hann_win);
  if (ctx->win_full) free(ctx->win_full);
  if (ctx->mel_fb) free(ctx->mel_fb);
  if (ctx->y_padded) free(ctx->y_padded);
  if (ctx->fft_buf) free(ctx->fft_buf);
  if (ctx->mag_buf) free(ctx->mag_buf);
  if (ctx->mel_row) free(ctx->mel_row);
  if (ctx->mel_in) free(ctx->mel_in);
  if (ctx->mel_out) free(ctx->mel_out);
  free(ctx);
}
