#pragma once
// image_processing.hpp — Image math: convolution, resampling, color ops,
//                         channel math, histogram, dithering, format helpers,
//                         HDR tone mapping, exposure, white balance
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "color.hpp"
#include "color_science.hpp"
#include "scalar_math.hpp"

namespace qs::math::image {

// ---------------------------------------------------------------------------
// Pixel format helpers
// ---------------------------------------------------------------------------
enum class PixelFormat {
    R8, RG8, RGB8, RGBA8,
    R16F, RG16F, RGB16F, RGBA16F,
    R32F, RG32F, RGB32F, RGBA32F,
    BC1, BC3, BC4, BC5, BC6H, BC7
};

[[nodiscard]] constexpr i32 bytes_per_pixel(PixelFormat fmt) noexcept {
    switch(fmt){
        case PixelFormat::R8:     return 1;
        case PixelFormat::RG8:    return 2;
        case PixelFormat::RGB8:   return 3;
        case PixelFormat::RGBA8:  return 4;
        case PixelFormat::R16F:   return 2;
        case PixelFormat::RG16F:  return 4;
        case PixelFormat::RGBA16F:return 8;
        case PixelFormat::R32F:   return 4;
        case PixelFormat::RGBA32F:return 16;
        default: return 4;
    }
}

[[nodiscard]] constexpr bool is_hdr_format(PixelFormat fmt) noexcept {
    return fmt==PixelFormat::R16F||fmt==PixelFormat::RG16F||
           fmt==PixelFormat::RGB16F||fmt==PixelFormat::RGBA16F||
           fmt==PixelFormat::R32F||fmt==PixelFormat::RG32F||
           fmt==PixelFormat::RGB32F||fmt==PixelFormat::RGBA32F;
}

// Mip count from dimensions
[[nodiscard]] constexpr i32 mip_count(i32 w, i32 h) noexcept {
    i32 n=0; while(w>0||h>0){ ++n; w>>=1; h>>=1; }
    return n;
}

// Mip dimensions
[[nodiscard]] constexpr Vec2i mip_dims(i32 w, i32 h, i32 mip) noexcept {
    return {max(1,w>>mip), max(1,h>>mip)};
}

// ---------------------------------------------------------------------------
// Image sampling
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec4f sample_nearest(
    const Vec4f* pixels, i32 w, i32 h, Vec2f uv) noexcept
{
    i32 x=clamp((i32)(uv.x*w),0,w-1);
    i32 y=clamp((i32)(uv.y*h),0,h-1);
    return pixels[y*w+x];
}

[[nodiscard]] inline Vec4f sample_bilinear(
    const Vec4f* pixels, i32 w, i32 h, Vec2f uv) noexcept
{
    f32 fx=uv.x*w-0.5f, fy=uv.y*h-0.5f;
    i32 ix=(i32)fx, iy=(i32)fy;
    f32 bx=fx-ix, by=fy-iy;
    auto s=[&](i32 x,i32 y)->Vec4f{ return pixels[clamp(y,0,h-1)*w+clamp(x,0,w-1)]; };
    return s(ix,iy).lerp(s(ix+1,iy),bx).lerp(s(ix,iy+1).lerp(s(ix+1,iy+1),bx),by);
}

// Catmull-Rom bicubic
[[nodiscard]] inline Vec4f sample_bicubic(
    const Vec4f* pixels, i32 w, i32 h, Vec2f uv) noexcept
{
    f32 fx=uv.x*w-0.5f, fy=uv.y*h-0.5f;
    i32 ix=(i32)fx, iy=(i32)fy;
    f32 bx=fx-ix, by=fy-iy;
    auto s=[&](i32 x,i32 y)->Vec4f{ return pixels[clamp(y,0,h-1)*w+clamp(x,0,w-1)]; };
    // Catmull-Rom 1D
    auto cr1=[&](Vec4f p0,Vec4f p1,Vec4f p2,Vec4f p3,f32 t)->Vec4f{
        f32 t2=t*t,t3=t2*t;
        return (p0*((-t3+2.f*t2-t)*0.5f)+p1*((3.f*t3-5.f*t2+2.f)*0.5f)
               +p2*((-3.f*t3+4.f*t2+t)*0.5f)+p3*((t3-t2)*0.5f));
    };
    Vec4f rows[4];
    for(i32 j=0;j<4;++j) rows[j]=cr1(s(ix-1,iy-1+j),s(ix,iy-1+j),s(ix+1,iy-1+j),s(ix+2,iy-1+j),bx);
    return cr1(rows[0],rows[1],rows[2],rows[3],by);
}

// ---------------------------------------------------------------------------
// Convolution kernels
// ---------------------------------------------------------------------------
// Apply convolution kernel (float, arbitrary size)
inline void convolve2d(
    const Vec4f* src, Vec4f* dst, i32 w, i32 h,
    const f32* kernel, i32 k_size) noexcept // k_size×k_size kernel
{
    i32 half=k_size/2;
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        Vec4f acc{0.f};
        for(i32 ky=0;ky<k_size;++ky) for(i32 kx=0;kx<k_size;++kx){
            i32 sx=clamp(x+kx-half,0,w-1), sy=clamp(y+ky-half,0,h-1);
            acc=acc+src[sy*w+sx]*kernel[ky*k_size+kx];
        }
        dst[y*w+x]=acc;
    }
}

// Gaussian kernel generation
inline void gaussian_kernel(f32* out, i32 size, f32 sigma) noexcept {
    i32 half=size/2; f32 sum=0.f;
    for(i32 y=0;y<size;++y) for(i32 x=0;x<size;++x){
        f32 dx=x-half, dy=y-half;
        f32 v=exp(-(dx*dx+dy*dy)/(2.f*sigma*sigma));
        out[y*size+x]=v; sum+=v;
    }
    for(i32 i=0;i<size*size;++i) out[i]/=sum;
}

// Separable Gaussian blur
inline void gaussian_blur_h(const Vec4f* src, Vec4f* dst, i32 w, i32 h,
                              const f32* kernel1d, i32 k) noexcept {
    i32 half=k/2;
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        Vec4f acc{0.f};
        for(i32 i=0;i<k;++i){
            i32 sx=clamp(x+i-half,0,w-1);
            acc=acc+src[y*w+sx]*kernel1d[i];
        }
        dst[y*w+x]=acc;
    }
}
inline void gaussian_blur_v(const Vec4f* src, Vec4f* dst, i32 w, i32 h,
                              const f32* kernel1d, i32 k) noexcept {
    i32 half=k/2;
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        Vec4f acc{0.f};
        for(i32 i=0;i<k;++i){
            i32 sy=clamp(y+i-half,0,h-1);
            acc=acc+src[sy*w+x]*kernel1d[i];
        }
        dst[y*w+x]=acc;
    }
}

// Common kernels (3×3)
inline constexpr f32 KERNEL_SHARPEN[9]={ 0,-1,0,-1,5,-1,0,-1,0 };
inline constexpr f32 KERNEL_EMBOSS[9]={ -2,-1,0,-1,1,1,0,1,2 };
inline constexpr f32 KERNEL_EDGE_SOBEL_X[9]={ -1,0,1,-2,0,2,-1,0,1 };
inline constexpr f32 KERNEL_EDGE_SOBEL_Y[9]={ -1,-2,-1,0,0,0,1,2,1 };
inline constexpr f32 KERNEL_LAPLACIAN[9]={ 0,1,0,1,-4,1,0,1,0 };
inline constexpr f32 KERNEL_BOX3[9]={ 1/9.f,1/9.f,1/9.f,1/9.f,1/9.f,1/9.f,1/9.f,1/9.f,1/9.f };

// Edge detection (Sobel magnitude)
inline void sobel_edges(const Vec4f* src, f32* edges, i32 w, i32 h) noexcept {
    for(i32 y=1;y<h-1;++y) for(i32 x=1;x<w-1;++x){
        auto lum=[&](i32 xi,i32 yi)->f32{ return src[yi*w+xi].x*0.299f+src[yi*w+xi].y*0.587f+src[yi*w+xi].z*0.114f; };
        f32 gx=KERNEL_EDGE_SOBEL_X[0]*lum(x-1,y-1)+KERNEL_EDGE_SOBEL_X[2]*lum(x+1,y-1)
              +KERNEL_EDGE_SOBEL_X[3]*lum(x-1,y  )+KERNEL_EDGE_SOBEL_X[5]*lum(x+1,y  )
              +KERNEL_EDGE_SOBEL_X[6]*lum(x-1,y+1)+KERNEL_EDGE_SOBEL_X[8]*lum(x+1,y+1);
        f32 gy=KERNEL_EDGE_SOBEL_Y[0]*lum(x-1,y-1)+KERNEL_EDGE_SOBEL_Y[1]*lum(x,y-1)+KERNEL_EDGE_SOBEL_Y[2]*lum(x+1,y-1)
              +KERNEL_EDGE_SOBEL_Y[6]*lum(x-1,y+1)+KERNEL_EDGE_SOBEL_Y[7]*lum(x,y+1)+KERNEL_EDGE_SOBEL_Y[8]*lum(x+1,y+1);
        edges[y*w+x]=sqrt(gx*gx+gy*gy);
    }
}

// ---------------------------------------------------------------------------
// Image resampling
// ---------------------------------------------------------------------------
// Lanczos weight
[[nodiscard]] inline f32 lanczos(f32 x, i32 a=3) noexcept {
    if(abs(x)<EPS_F32) return 1.f;
    if(abs(x)>=a) return 0.f;
    return a*sin(PI*x)*sin(PI*x/a)/(PI*PI*x*x);
}

// Downscale by 2x (box filter)
inline void downscale2x(const Vec4f* src, Vec4f* dst, i32 w, i32 h) noexcept {
    i32 dw=max(1,w/2), dh=max(1,h/2);
    for(i32 y=0;y<dh;++y) for(i32 x=0;x<dw;++x){
        Vec4f s00=src[(y*2)*w+(x*2)], s10=src[(y*2)*w+min(x*2+1,w-1)];
        Vec4f s01=src[min(y*2+1,h-1)*w+(x*2)], s11=src[min(y*2+1,h-1)*w+min(x*2+1,w-1)];
        dst[y*dw+x]=(s00+s10+s01+s11)*0.25f;
    }
}

// ---------------------------------------------------------------------------
// Histogram equalization
// ---------------------------------------------------------------------------
inline void build_histogram(const f32* channel, i32 n,
                              u32* hist, i32 bins=256) noexcept {
    for(i32 i=0;i<bins;++i) hist[i]=0;
    for(i32 i=0;i<n;++i){
        i32 b=clamp((i32)(channel[i]*bins),0,bins-1);
        ++hist[b];
    }
}

inline void histogram_equalize_channel(f32* channel, i32 n, i32 bins=256) noexcept {
    u32 hist[256]={}; build_histogram(channel,n,hist,bins);
    u32 cdf[256]={}; cdf[0]=hist[0];
    for(i32 i=1;i<bins;++i) cdf[i]=cdf[i-1]+hist[i];
    u32 cdf_min=0; for(i32 i=0;i<bins;++i) if(cdf[i]>0){ cdf_min=cdf[i]; break; }
    for(i32 i=0;i<n;++i){
        i32 b=clamp((i32)(channel[i]*bins),0,bins-1);
        channel[i]=f32(cdf[b]-cdf_min)/f32(n-cdf_min);
    }
}

// ---------------------------------------------------------------------------
// Image ops: brightness, contrast, saturation, hue
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec4f adjust_brightness(Vec4f c, f32 amount) noexcept {
    return {c.x+amount,c.y+amount,c.z+amount,c.w};
}
[[nodiscard]] constexpr Vec4f adjust_contrast(Vec4f c, f32 contrast) noexcept {
    return {(c.x-0.5f)*contrast+0.5f,(c.y-0.5f)*contrast+0.5f,(c.z-0.5f)*contrast+0.5f,c.w};
}
[[nodiscard]] inline Vec4f adjust_saturation(Vec4f c, f32 saturation) noexcept {
    f32 lum=c.x*0.2126f+c.y*0.7152f+c.z*0.0722f;
    return {lerp(lum,c.x,saturation),lerp(lum,c.y,saturation),lerp(lum,c.z,saturation),c.w};
}
[[nodiscard]] inline Vec4f adjust_hue(Vec4f c, f32 hue_shift_deg) noexcept {
    ColorRGB rgb{c.x,c.y,c.z};
    ColorHSV hsv=ColorHSV::from_rgb(rgb);
    hsv.h=wrap(hsv.h+hue_shift_deg,0.f,360.f);
    ColorRGB out=hsv.to_rgb();
    return {out.r,out.g,out.b,c.w};
}

// Levels (black/white point + gamma)
[[nodiscard]] constexpr f32 apply_levels(f32 v, f32 in_black, f32 in_white,
                                          f32 gamma, f32 out_black, f32 out_white) noexcept {
    v=(v-in_black)/(in_white-in_black+EPS_F32);
    v=saturate(v);
    v=pow(v,1.f/gamma);
    return v*(out_white-out_black)+out_black;
}

// Curves (simple 4-point spline LUT)
inline void build_curves_lut(f32* lut, i32 lut_size,
                               Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3) noexcept {
    for(i32 i=0;i<lut_size;++i){
        f32 t=f32(i)/(lut_size-1);
        lut[i]=saturate(bezier_cubic(p0,p1,p2,p3,t).y);
    }
}

// ---------------------------------------------------------------------------
// Normal map from height map (CPU bake)
// ---------------------------------------------------------------------------
inline void height_to_normal(const f32* height, Vec4f* normals,
                               i32 w, i32 h, f32 strength=1.f) noexcept
{
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        auto s=[&](i32 xi,i32 yi)->f32{
            return height[clamp(yi,0,h-1)*w+clamp(xi,0,w-1)];
        };
        f32 dx=(s(x+1,y)-s(x-1,y))*0.5f*strength;
        f32 dy=(s(x,y+1)-s(x,y-1))*0.5f*strength;
        Vec3f n=Vec3f{-dx,-dy,1.f}.normalized();
        normals[y*w+x]={n.x*0.5f+0.5f,n.y*0.5f+0.5f,n.z*0.5f+0.5f,1.f};
    }
}

} // namespace qs::math::image
