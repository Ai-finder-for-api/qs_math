
#pragma once
// qs_math/media/include/qs_math_media.hpp (C++20)
// Signal processing, audio/video math, FFT, windowing, resampling.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <complex>
namespace qs::math::media {
using namespace qs::math;
using Cf64 = std::complex<f64>;
// ─── FFT (Cooley-Tukey, power-of-2) ──────────────────────────────────────────
void fft(std::vector<Cf64>& a, bool inverse=false) noexcept;
void rfft(const f64* in, i32 n, std::vector<Cf64>& out) noexcept;
void irfft(const std::vector<Cf64>& in, i32 n, std::vector<f64>& out) noexcept;
// ─── Window functions ─────────────────────────────────────────────────────────
enum class Window { Hann, Hamming, Blackman, FlatTop, Kaiser, Triangular, Rectangular };
void apply_window(f64* data, i32 n, Window w, f64 beta=8.6) noexcept;
[[nodiscard]] std::vector<f64> window_coeffs(i32 n, Window w, f64 beta=8.6) noexcept;
// ─── Power spectral density ───────────────────────────────────────────────────
[[nodiscard]] std::vector<f64> psd(const f64* data, i32 n, Window w=Window::Hann) noexcept;
// ─── Convolution / cross-correlation ─────────────────────────────────────────
std::vector<f64> convolve(const std::vector<f64>& a, const std::vector<f64>& b) noexcept;
std::vector<f64> cross_correlate(const std::vector<f64>& a, const std::vector<f64>& b) noexcept;
// ─── Digital IIR filters (biquad) ────────────────────────────────────────────
struct BiquadCoeffs { f64 b0,b1,b2,a1,a2; };
struct BiquadState  { f64 z1{0},z2{0}; };
[[nodiscard]] BiquadCoeffs biquad_lpf(f64 fc, f64 fs, f64 Q=0.707) noexcept;
[[nodiscard]] BiquadCoeffs biquad_hpf(f64 fc, f64 fs, f64 Q=0.707) noexcept;
[[nodiscard]] BiquadCoeffs biquad_bpf(f64 fc, f64 fs, f64 Q=1.0) noexcept;
[[nodiscard]] BiquadCoeffs biquad_peak(f64 fc, f64 fs, f64 gain_db, f64 Q=1.0) noexcept;
f64 biquad_process(const BiquadCoeffs& c, BiquadState& s, f64 x) noexcept;
void biquad_process_block(const BiquadCoeffs& c, BiquadState& s,
    const f64* in, f64* out, i32 n) noexcept;
// ─── Sinc resampler ───────────────────────────────────────────────────────────
void resample_sinc(const f64* in, i32 n_in, f64* out, i32 n_out,
    i32 kernel_taps=64) noexcept;
// ─── Mel scale ────────────────────────────────────────────────────────────────
[[nodiscard]] inline f64 hz_to_mel(f64 hz) noexcept { return 2595.0*std::log10(1+hz/700.0); }
[[nodiscard]] inline f64 mel_to_hz(f64 mel) noexcept { return 700.0*(std::pow(10.0,mel/2595.0)-1); }
// Mel filterbank: returns M×(N/2+1) filter matrix
std::vector<std::vector<f64>> mel_filterbank(i32 M, i32 N, f64 fs,
    f64 f_min=80, f64 f_max=8000) noexcept;
} // namespace qs::math::media
