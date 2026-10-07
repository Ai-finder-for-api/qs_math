#pragma once
// qs_math/simd_math.hpp — Actual SSE/AVX/NEON intrinsic wrappers (C++20)
// Replaces simd_stub.hpp with real implementations + compile-time dispatch.
// Include this instead of simd_stub.hpp for hot paths.

#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"

// Detect available SIMD
#if defined(__AVX2__)
#  include <immintrin.h>
#  define QS_HAS_AVX2 1
#  define QS_HAS_SSE4 1
#  define QS_HAS_SSE2 1
#elif defined(__AVX__)
#  include <immintrin.h>
#  define QS_HAS_AVX 1
#  define QS_HAS_SSE4 1
#  define QS_HAS_SSE2 1
#elif defined(__SSE4_1__)
#  include <smmintrin.h>
#  define QS_HAS_SSE4 1
#  define QS_HAS_SSE2 1
#elif defined(__SSE2__)
#  include <emmintrin.h>
#  define QS_HAS_SSE2 1
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define QS_HAS_NEON 1
#endif

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// Float4 — 128-bit SIMD float register abstraction
// Wraps __m128 / float32x4_t / fallback f32[4]
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) Float4 {
#if defined(QS_HAS_SSE2)
    using Reg = __m128;
    Reg r;
    Float4() noexcept : r(_mm_setzero_ps()) {}
    explicit Float4(Reg v) noexcept : r(v) {}
    Float4(f32 a, f32 b, f32 c, f32 d) noexcept : r(_mm_set_ps(d, c, b, a)) {}
    explicit Float4(f32 v) noexcept : r(_mm_set1_ps(v)) {}
    static Float4 load(const f32* p)        noexcept { Float4 v; v.r = _mm_loadu_ps(p); return v; }
    static Float4 load_aligned(const f32* p)noexcept { Float4 v; v.r = _mm_load_ps(p);  return v; }
    void store(f32* p)         const noexcept { _mm_storeu_ps(p, r); }
    void store_aligned(f32* p) const noexcept { _mm_store_ps(p, r);  }

    [[nodiscard]] Float4 operator+(const Float4& b) const noexcept { return Float4(_mm_add_ps(r, b.r)); }
    [[nodiscard]] Float4 operator-(const Float4& b) const noexcept { return Float4(_mm_sub_ps(r, b.r)); }
    [[nodiscard]] Float4 operator*(const Float4& b) const noexcept { return Float4(_mm_mul_ps(r, b.r)); }
    [[nodiscard]] Float4 operator/(const Float4& b) const noexcept { return Float4(_mm_div_ps(r, b.r)); }
    [[nodiscard]] Float4 operator-()               const noexcept { return Float4(_mm_sub_ps(_mm_setzero_ps(), r)); }

    [[nodiscard]] Float4 sqrt()  const noexcept { return Float4(_mm_sqrt_ps(r)); }
    [[nodiscard]] Float4 rsqrt() const noexcept { return Float4(_mm_rsqrt_ps(r)); }  // ~12-bit precision
    [[nodiscard]] Float4 rcp()   const noexcept { return Float4(_mm_rcp_ps(r)); }

    [[nodiscard]] Float4 min(const Float4& b) const noexcept { return Float4(_mm_min_ps(r, b.r)); }
    [[nodiscard]] Float4 max(const Float4& b) const noexcept { return Float4(_mm_max_ps(r, b.r)); }
    [[nodiscard]] Float4 abs()               const noexcept {
        __m128i mask = _mm_set1_epi32(0x7FFFFFFF);
        return Float4(_mm_and_ps(r, _mm_castsi128_ps(mask)));
    }

    // Horizontal ops
    [[nodiscard]] f32 hsum() const noexcept {
        __m128 s = _mm_add_ps(r, _mm_movehl_ps(r, r));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }
    [[nodiscard]] f32 hmax() const noexcept {
        __m128 s = _mm_max_ps(r, _mm_movehl_ps(r, r));
        s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }
    [[nodiscard]] f32 hmin() const noexcept {
        __m128 s = _mm_min_ps(r, _mm_movehl_ps(r, r));
        s = _mm_min_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }

    // Dot product of 4 components
    [[nodiscard]] f32 dot(const Float4& b) const noexcept { return (*this * b).hsum(); }

    // Shuffle (compile-time mask)
    template<int A, int B, int C, int D>
    [[nodiscard]] Float4 shuffle() const noexcept { return Float4(_mm_shuffle_ps(r, r, _MM_SHUFFLE(D,C,B,A))); }

    // Blend / select
    [[nodiscard]] Float4 blend(const Float4& b, const Float4& mask) const noexcept {
#if defined(QS_HAS_SSE4)
        return Float4(_mm_blendv_ps(r, b.r, mask.r));
#else
        return Float4(_mm_or_ps(_mm_andnot_ps(mask.r, r), _mm_and_ps(mask.r, b.r)));
#endif
    }

    // Element access
    [[nodiscard]] f32 x() const noexcept { return _mm_cvtss_f32(r); }
    [[nodiscard]] f32 y() const noexcept { return _mm_cvtss_f32(_mm_shuffle_ps(r,r,1)); }
    [[nodiscard]] f32 z() const noexcept { return _mm_cvtss_f32(_mm_shuffle_ps(r,r,2)); }
    [[nodiscard]] f32 w() const noexcept { return _mm_cvtss_f32(_mm_shuffle_ps(r,r,3)); }

#elif defined(QS_HAS_NEON)
    using Reg = float32x4_t;
    Reg r;
    Float4() noexcept : r(vdupq_n_f32(0.f)) {}
    explicit Float4(Reg v) noexcept : r(v) {}
    Float4(f32 a, f32 b, f32 c, f32 d) noexcept { f32 arr[4]{a,b,c,d}; r = vld1q_f32(arr); }
    explicit Float4(f32 v) noexcept : r(vdupq_n_f32(v)) {}
    static Float4 load(const f32* p)         noexcept { Float4 v; v.r = vld1q_f32(p); return v; }
    static Float4 load_aligned(const f32* p) noexcept { return load(p); }
    void store(f32* p)         const noexcept { vst1q_f32(p, r); }
    void store_aligned(f32* p) const noexcept { vst1q_f32(p, r); }

    [[nodiscard]] Float4 operator+(const Float4& b) const noexcept { return Float4(vaddq_f32(r, b.r)); }
    [[nodiscard]] Float4 operator-(const Float4& b) const noexcept { return Float4(vsubq_f32(r, b.r)); }
    [[nodiscard]] Float4 operator*(const Float4& b) const noexcept { return Float4(vmulq_f32(r, b.r)); }
    [[nodiscard]] Float4 operator/(const Float4& b) const noexcept { return Float4(vdivq_f32(r, b.r)); }
    [[nodiscard]] Float4 operator-()                const noexcept { return Float4(vnegq_f32(r)); }
    [[nodiscard]] Float4 sqrt()  const noexcept { return Float4(vsqrtq_f32(r)); }
    [[nodiscard]] Float4 rsqrt() const noexcept { return Float4(vrsqrteq_f32(r)); }
    [[nodiscard]] Float4 rcp()   const noexcept { return Float4(vrecpeq_f32(r)); }
    [[nodiscard]] Float4 min(const Float4& b) const noexcept { return Float4(vminq_f32(r, b.r)); }
    [[nodiscard]] Float4 max(const Float4& b) const noexcept { return Float4(vmaxq_f32(r, b.r)); }
    [[nodiscard]] Float4 abs()               const noexcept { return Float4(vabsq_f32(r)); }
    [[nodiscard]] f32 hsum() const noexcept { float32x2_t s = vadd_f32(vget_low_f32(r), vget_high_f32(r)); return vget_lane_f32(vpadd_f32(s,s),0); }
    [[nodiscard]] f32 dot(const Float4& b) const noexcept { return (*this * b).hsum(); }
    [[nodiscard]] f32 x() const noexcept { return vgetq_lane_f32(r,0); }
    [[nodiscard]] f32 y() const noexcept { return vgetq_lane_f32(r,1); }
    [[nodiscard]] f32 z() const noexcept { return vgetq_lane_f32(r,2); }
    [[nodiscard]] f32 w() const noexcept { return vgetq_lane_f32(r,3); }

#else
    // Scalar fallback
    f32 e[4]{0,0,0,0};
    Float4() noexcept = default;
    Float4(f32 a,f32 b,f32 c,f32 d) noexcept : e{a,b,c,d} {}
    explicit Float4(f32 v) noexcept : e{v,v,v,v} {}
    static Float4 load(const f32* p) noexcept { return Float4(p[0],p[1],p[2],p[3]); }
    static Float4 load_aligned(const f32* p) noexcept { return load(p); }
    void store(f32* p)         const noexcept { p[0]=e[0];p[1]=e[1];p[2]=e[2];p[3]=e[3]; }
    void store_aligned(f32* p) const noexcept { store(p); }
    [[nodiscard]] Float4 operator+(const Float4& b) const noexcept { return Float4(e[0]+b.e[0],e[1]+b.e[1],e[2]+b.e[2],e[3]+b.e[3]); }
    [[nodiscard]] Float4 operator-(const Float4& b) const noexcept { return Float4(e[0]-b.e[0],e[1]-b.e[1],e[2]-b.e[2],e[3]-b.e[3]); }
    [[nodiscard]] Float4 operator*(const Float4& b) const noexcept { return Float4(e[0]*b.e[0],e[1]*b.e[1],e[2]*b.e[2],e[3]*b.e[3]); }
    [[nodiscard]] Float4 operator/(const Float4& b) const noexcept { return Float4(e[0]/b.e[0],e[1]/b.e[1],e[2]/b.e[2],e[3]/b.e[3]); }
    [[nodiscard]] Float4 sqrt()  const noexcept { return Float4(std::sqrt(e[0]),std::sqrt(e[1]),std::sqrt(e[2]),std::sqrt(e[3])); }
    [[nodiscard]] Float4 rsqrt() const noexcept { return Float4(1.f/std::sqrt(e[0]),1.f/std::sqrt(e[1]),1.f/std::sqrt(e[2]),1.f/std::sqrt(e[3])); }
    [[nodiscard]] Float4 rcp()   const noexcept { return Float4(1.f/e[0],1.f/e[1],1.f/e[2],1.f/e[3]); }
    [[nodiscard]] Float4 min(const Float4& b) const noexcept { return Float4(std::min(e[0],b.e[0]),std::min(e[1],b.e[1]),std::min(e[2],b.e[2]),std::min(e[3],b.e[3])); }
    [[nodiscard]] Float4 max(const Float4& b) const noexcept { return Float4(std::max(e[0],b.e[0]),std::max(e[1],b.e[1]),std::max(e[2],b.e[2]),std::max(e[3],b.e[3])); }
    [[nodiscard]] Float4 abs()               const noexcept { return Float4(std::abs(e[0]),std::abs(e[1]),std::abs(e[2]),std::abs(e[3])); }
    [[nodiscard]] f32 hsum() const noexcept { return e[0]+e[1]+e[2]+e[3]; }
    [[nodiscard]] f32 dot(const Float4& b) const noexcept { return (*this*b).hsum(); }
    [[nodiscard]] f32 x() const noexcept { return e[0]; } [[nodiscard]] f32 y() const noexcept { return e[1]; }
    [[nodiscard]] f32 z() const noexcept { return e[2]; } [[nodiscard]] f32 w() const noexcept { return e[3]; }
#endif

    // Common helpers regardless of backend
    [[nodiscard]] Float4 madd(const Float4& b, const Float4& c) const noexcept { return *this * b + c; }
    [[nodiscard]] Float4 clamp(const Float4& lo, const Float4& hi) const noexcept { return min(hi).max(lo); }
    [[nodiscard]] Float4 saturate()  const noexcept { return clamp(Float4(0.f), Float4(1.f)); }
    [[nodiscard]] f32    length_sq() const noexcept { return dot(*this); }
    [[nodiscard]] f32    length()    const noexcept { return std::sqrt(length_sq()); }
    [[nodiscard]] Float4 normalize() const noexcept { return *this * Float4(1.f / length()); }

    // Vec4 interop
    [[nodiscard]] Vec4f to_vec4() const noexcept { return Vec4f{x(), y(), z(), w()}; }
    static Float4 from_vec4(const Vec4f& v)noexcept{ return Float4{v.x, v.y, v.z, v.w}; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Int4 — 128-bit integer SIMD register (i32 x4)
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) Int4 {
#if defined(QS_HAS_SSE2)
    __m128i r;
    Int4() noexcept : r(_mm_setzero_si128()) {}
    explicit Int4(__m128i v) noexcept : r(v) {}
    Int4(i32 a, i32 b, i32 c, i32 d) noexcept : r(_mm_set_epi32(d,c,b,a)) {}
    explicit Int4(i32 v) noexcept : r(_mm_set1_epi32(v)) {}
    static Int4 load(const i32* p) noexcept { Int4 v; v.r = _mm_loadu_si128((const __m128i*)p); return v; }
    void store(i32* p) const noexcept { _mm_storeu_si128((__m128i*)p, r); }

    [[nodiscard]] Int4 operator+(const Int4& b) const noexcept { return Int4(_mm_add_epi32(r, b.r)); }
    [[nodiscard]] Int4 operator-(const Int4& b) const noexcept { return Int4(_mm_sub_epi32(r, b.r)); }
    [[nodiscard]] Int4 operator*(const Int4& b) const noexcept {
#if defined(QS_HAS_SSE4)
        return Int4(_mm_mullo_epi32(r, b.r));
#else
        __m128i tmp1 = _mm_mul_epu32(r, b.r);
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(r,4), _mm_srli_si128(b.r,4));
        return Int4(_mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1,_MM_SHUFFLE(0,0,2,0)), _mm_shuffle_epi32(tmp2,_MM_SHUFFLE(0,0,2,0))));
#endif
    }
    [[nodiscard]] Int4 operator&(const Int4& b) const noexcept { return Int4(_mm_and_si128(r, b.r)); }
    [[nodiscard]] Int4 operator|(const Int4& b) const noexcept { return Int4(_mm_or_si128(r, b.r));  }
    [[nodiscard]] Int4 operator^(const Int4& b) const noexcept { return Int4(_mm_xor_si128(r, b.r)); }
    [[nodiscard]] Int4 operator<<(i32 n) const noexcept { return Int4(_mm_slli_epi32(r, n)); }
    [[nodiscard]] Int4 operator>>(i32 n) const noexcept { return Int4(_mm_srai_epi32(r, n)); }
    [[nodiscard]] Float4 to_float4() const noexcept { return Float4(_mm_cvtepi32_ps(r)); }
    [[nodiscard]] i32 x() const noexcept { return _mm_cvtsi128_si32(r); }
#elif defined(QS_HAS_NEON)
    int32x4_t r;
    Int4() noexcept : r(vdupq_n_s32(0)) {}
    explicit Int4(int32x4_t v) noexcept : r(v) {}
    Int4(i32 a,i32 b,i32 c,i32 d) noexcept { i32 arr[4]{a,b,c,d}; r = vld1q_s32(arr); }
    explicit Int4(i32 v) noexcept : r(vdupq_n_s32(v)) {}
    static Int4 load(const i32* p) noexcept { Int4 v; v.r = vld1q_s32(p); return v; }
    void store(i32* p) const noexcept { vst1q_s32(p, r); }
    [[nodiscard]] Int4 operator+(const Int4& b) const noexcept { return Int4(vaddq_s32(r,b.r)); }
    [[nodiscard]] Int4 operator-(const Int4& b) const noexcept { return Int4(vsubq_s32(r,b.r)); }
    [[nodiscard]] Int4 operator*(const Int4& b) const noexcept { return Int4(vmulq_s32(r,b.r)); }
    [[nodiscard]] Int4 operator&(const Int4& b) const noexcept { return Int4(vandq_s32(r,b.r)); }
    [[nodiscard]] Int4 operator|(const Int4& b) const noexcept { return Int4(vorrq_s32(r,b.r));  }
    [[nodiscard]] Int4 operator^(const Int4& b) const noexcept { return Int4(veorq_s32(r,b.r)); }
    [[nodiscard]] Float4 to_float4() const noexcept { return Float4(vcvtq_f32_s32(r)); }
    [[nodiscard]] i32 x() const noexcept { return vgetq_lane_s32(r,0); }
#else
    i32 e[4]{0,0,0,0};
    Int4() noexcept = default;
    Int4(i32 a,i32 b,i32 c,i32 d) noexcept : e{a,b,c,d} {}
    explicit Int4(i32 v) noexcept : e{v,v,v,v} {}
    static Int4 load(const i32* p) noexcept { return Int4(p[0],p[1],p[2],p[3]); }
    void store(i32* p) const noexcept { p[0]=e[0];p[1]=e[1];p[2]=e[2];p[3]=e[3]; }
    [[nodiscard]] Int4 operator+(const Int4& b) const noexcept { return Int4(e[0]+b.e[0],e[1]+b.e[1],e[2]+b.e[2],e[3]+b.e[3]); }
    [[nodiscard]] Int4 operator-(const Int4& b) const noexcept { return Int4(e[0]-b.e[0],e[1]-b.e[1],e[2]-b.e[2],e[3]-b.e[3]); }
    [[nodiscard]] Int4 operator*(const Int4& b) const noexcept { return Int4(e[0]*b.e[0],e[1]*b.e[1],e[2]*b.e[2],e[3]*b.e[3]); }
    [[nodiscard]] Int4 operator&(const Int4& b) const noexcept { return Int4(e[0]&b.e[0],e[1]&b.e[1],e[2]&b.e[2],e[3]&b.e[3]); }
    [[nodiscard]] Int4 operator|(const Int4& b) const noexcept { return Int4(e[0]|b.e[0],e[1]|b.e[1],e[2]|b.e[2],e[3]|b.e[3]); }
    [[nodiscard]] Int4 operator^(const Int4& b) const noexcept { return Int4(e[0]^b.e[0],e[1]^b.e[1],e[2]^b.e[2],e[3]^b.e[3]); }
    [[nodiscard]] Float4 to_float4() const noexcept { return Float4(f32(e[0]),f32(e[1]),f32(e[2]),f32(e[3])); }
    [[nodiscard]] i32 x() const noexcept { return e[0]; }
#endif
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD Mat4 multiply: 4x4 column-major f32 matrix (SSE2+ fast path)
// ─────────────────────────────────────────────────────────────────────────────
#if defined(QS_HAS_SSE2)
[[nodiscard]] inline Mat4f simd_mat4_mul(const Mat4f& A, const Mat4f& B) noexcept {
    // Load columns of B
    __m128 b0 = _mm_loadu_ps(&B.m[0][0]);
    __m128 b1 = _mm_loadu_ps(&B.m[1][0]);
    __m128 b2 = _mm_loadu_ps(&B.m[2][0]);
    __m128 b3 = _mm_loadu_ps(&B.m[3][0]);
    Mat4f R;
    for (int i = 0; i < 4; ++i) {
        __m128 a = _mm_loadu_ps(&A.m[i][0]);
        __m128 r = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(a,a,0x00), b0),
                       _mm_mul_ps(_mm_shuffle_ps(a,a,0x55), b1)),
            _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(a,a,0xAA), b2),
                       _mm_mul_ps(_mm_shuffle_ps(a,a,0xFF), b3)));
        _mm_storeu_ps(&R.m[i][0], r);
    }
    return R;
}

[[nodiscard]] inline Vec4f simd_mat4_vec4_mul(const Mat4f& M, const Vec4f& v) noexcept {
    __m128 vv = _mm_set_ps(v.w, v.z, v.y, v.x);
    __m128 r  = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(&M.m[0][0]), _mm_shuffle_ps(vv,vv,0x00)),
                   _mm_mul_ps(_mm_loadu_ps(&M.m[1][0]), _mm_shuffle_ps(vv,vv,0x55))),
        _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(&M.m[2][0]), _mm_shuffle_ps(vv,vv,0xAA)),
                   _mm_mul_ps(_mm_loadu_ps(&M.m[3][0]), _mm_shuffle_ps(vv,vv,0xFF))));
    Vec4f out; _mm_storeu_ps(&out.x, r); return out;
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Batch operations — common patterns on arrays
// ─────────────────────────────────────────────────────────────────────────────

// Dot product of two float arrays of length n (SIMD-vectorised)
[[nodiscard]] inline f32 simd_dot_array(const f32* a, const f32* b, u32 n) noexcept {
    Float4 acc(0.f);
    u32 i = 0;
    for (; i + 4 <= n; i += 4)
        acc = acc + Float4::load(a+i) * Float4::load(b+i);
    f32 r = acc.hsum();
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

// Clamp a float array in-place
inline void simd_clamp_array(f32* data, u32 n, f32 lo, f32 hi) noexcept {
    Float4 vlo(lo), vhi(hi);
    u32 i = 0;
    for (; i + 4 <= n; i += 4) {
        Float4 v = Float4::load(data+i);
        v.clamp(vlo, vhi).store(data+i);
    }
    for (; i < n; ++i) data[i] = data[i] < lo ? lo : (data[i] > hi ? hi : data[i]);
}

// Multiply-add: dst[i] = a[i]*b[i] + c[i]
inline void simd_madd_array(const f32* a, const f32* b, const f32* c, f32* dst, u32 n) noexcept {
    u32 i = 0;
    for (; i + 4 <= n; i += 4)
        (Float4::load(a+i) * Float4::load(b+i) + Float4::load(c+i)).store(dst+i);
    for (; i < n; ++i) dst[i] = a[i]*b[i] + c[i];
}

// Normalize an array of Vec3 (stride = 3 floats)
inline void simd_normalize_vec3_array(f32* vecs, u32 count) noexcept {
    for (u32 i = 0; i < count; ++i) {
        f32* v = vecs + i*3;
        Float4 f(v[0], v[1], v[2], 0.f);
        Float4 n = f.normalize();
        v[0] = n.x(); v[1] = n.y(); v[2] = n.z();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime capability query
// ─────────────────────────────────────────────────────────────────────────────
enum class SimdLevel { Scalar, SSE2, SSE4, AVX, AVX2, NEON };

[[nodiscard]] inline SimdLevel simd_capability() noexcept {
#if defined(QS_HAS_AVX2)
    return SimdLevel::AVX2;
#elif defined(QS_HAS_AVX)
    return SimdLevel::AVX;
#elif defined(QS_HAS_SSE4)
    return SimdLevel::SSE4;
#elif defined(QS_HAS_SSE2)
    return SimdLevel::SSE2;
#elif defined(QS_HAS_NEON)
    return SimdLevel::NEON;
#else
    return SimdLevel::Scalar;
#endif
}

[[nodiscard]] inline const char* simd_capability_name() noexcept {
    switch (simd_capability()) {
        case SimdLevel::AVX2:   return "AVX2";
        case SimdLevel::AVX:    return "AVX";
        case SimdLevel::SSE4:   return "SSE4.1";
        case SimdLevel::SSE2:   return "SSE2";
        case SimdLevel::NEON:   return "NEON";
        default:                return "Scalar";
    }
}

} // namespace qs
