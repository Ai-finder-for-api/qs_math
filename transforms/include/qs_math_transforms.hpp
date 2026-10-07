
#pragma once
// qs_math/transforms/include/qs_math_transforms.hpp (C++20)
// Mathematical transforms: FFT, DCT, WHT, wavelet, Z-transform, Laplace.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <complex>
#include <array>
namespace qs::math::xform {
using namespace qs::math;
using Cf64 = std::complex<f64>;
// ─── FFT (already in media, exposed here for signal-processing context) ───────
void fft_inplace(std::vector<Cf64>& a, bool inverse=false) noexcept;
// ─── Real FFT helper ──────────────────────────────────────────────────────────
[[nodiscard]] std::vector<Cf64> rfft(const std::vector<f64>& x) noexcept;
[[nodiscard]] std::vector<f64> irfft(const std::vector<Cf64>& X, i32 n) noexcept;
// ─── DCT-II and DCT-III ───────────────────────────────────────────────────────
void dct2(const f64* in, f64* out, i32 n) noexcept;
void dct3(const f64* in, f64* out, i32 n) noexcept; // inverse of DCT-II (scaled)
void dct4(const f64* in, f64* out, i32 n) noexcept;
// ─── 2D DCT (block-based, 8×8 for JPEG) ─────────────────────────────────────
void dct2_8x8(const f64 in[64], f64 out[64]) noexcept;
void idct2_8x8(const f64 in[64], f64 out[64]) noexcept;
// ─── Walsh-Hadamard transform ─────────────────────────────────────────────────
void wht(f64* data, i32 n) noexcept; // in-place, n must be power of 2
void iwht(f64* data, i32 n) noexcept;
// ─── Haar wavelet transform ───────────────────────────────────────────────────
void haar_fwd(f64* data, i32 n) noexcept; // n must be power of 2
void haar_inv(f64* data, i32 n) noexcept;
// ─── Daubechies D4 wavelet ───────────────────────────────────────────────────
void db4_fwd(f64* data, i32 n) noexcept;
void db4_inv(f64* data, i32 n) noexcept;
// ─── Chirp Z-transform (generalized FFT) ─────────────────────────────────────
[[nodiscard]] std::vector<Cf64> czt(const std::vector<Cf64>& x,
    i32 M, Cf64 W, Cf64 A) noexcept;
// ─── Z-transform evaluation ───────────────────────────────────────────────────
[[nodiscard]] Cf64 z_transform_eval(const f64* coefs, i32 n, Cf64 z) noexcept;
// ─── Hilbert transform ────────────────────────────────────────────────────────
[[nodiscard]] std::vector<f64> hilbert_envelope(const std::vector<f64>& x) noexcept;
[[nodiscard]] std::vector<f64> instantaneous_phase(const std::vector<f64>& x) noexcept;
// ─── Short-time Fourier transform ────────────────────────────────────────────
struct STFTResult {
    std::vector<std::vector<Cf64>> frames; // [time_frame][freq_bin]
    i32 n_frames{0}, n_bins{0};
};
[[nodiscard]] STFTResult stft(const std::vector<f64>& x,
    i32 frame_size=512, i32 hop=256) noexcept;
[[nodiscard]] std::vector<f64> istft(const STFTResult& s,
    i32 frame_size=512, i32 hop=256) noexcept;
} // namespace qs::math::xform
