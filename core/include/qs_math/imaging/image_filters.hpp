#pragma once
// qs_math/image_filters.hpp — Bilateral, guided, anisotropic diffusion, morphological ops (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cassert>
#include <array>
#include <numeric>

namespace qs {

// ─── Image buffer helper ──────────────────────────────────────────────────────
struct ImageBuf {
    std::vector<f32> data;
    u32 w{0},h{0},c{1}; // width,height,channels

    ImageBuf()=default;
    ImageBuf(u32 w,u32 h,u32 c,f32 fill=0):data(w*h*c,fill),w(w),h(h),c(c){}

    [[nodiscard]] f32& at(u32 x,u32 y,u32 ch=0)       { return data[(y*w+x)*c+ch]; }
    [[nodiscard]] f32  at(u32 x,u32 y,u32 ch=0) const { return data[(y*w+x)*c+ch]; }
    [[nodiscard]] f32  get_clamped(i32 x,i32 y,u32 ch=0) const {
        x=std::clamp(x,0,i32(w)-1); y=std::clamp(y,0,i32(h)-1);
        return data[(u32(y)*w+u32(x))*c+ch];
    }
    [[nodiscard]] Vec3f pixel(u32 x,u32 y) const {
        if(c>=3) return {at(x,y,0),at(x,y,1),at(x,y,2)};
        f32 v=at(x,y,0); return {v,v,v};
    }
    void set_pixel(u32 x,u32 y,Vec3f p){
        if(c>=1)at(x,y,0)=p.x; if(c>=2)at(x,y,1)=p.y; if(c>=3)at(x,y,2)=p.z;
    }
    [[nodiscard]] ImageBuf clone() const { return *this; }
    [[nodiscard]] u32 n_pixels() const { return w*h; }
};

// ─── Box blur ────────────────────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf box_blur(const ImageBuf& src, u32 radius) {
    ImageBuf dst=src.clone();
    u32 ksize=2*radius+1;
    f32 inv=1.f/f32(ksize);
    // Horizontal pass
    ImageBuf tmp=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 ch=0;ch<src.c;++ch) {
        f32 sum=0;
        for(u32 k=0;k<ksize;++k) sum+=src.get_clamped(i32(k)-i32(radius),i32(y),ch);
        for(u32 x=0;x<src.w;++x) {
            tmp.at(x,y,ch)=sum*inv;
            sum+=src.get_clamped(i32(x)+i32(radius)+1,i32(y),ch);
            sum-=src.get_clamped(i32(x)-i32(radius),  i32(y),ch);
        }
    }
    // Vertical pass
    for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
        f32 sum=0;
        for(u32 k=0;k<ksize;++k) sum+=tmp.get_clamped(i32(x),i32(k)-i32(radius),ch);
        for(u32 y=0;y<src.h;++y) {
            dst.at(x,y,ch)=sum*inv;
            sum+=tmp.get_clamped(i32(x),i32(y)+i32(radius)+1,ch);
            sum-=tmp.get_clamped(i32(x),i32(y)-i32(radius),  ch);
        }
    }
    return dst;
}

// ─── Gaussian blur ────────────────────────────────────────────────────────────
[[nodiscard]] inline std::vector<f32> gaussian_kernel_1d(f32 sigma, u32 radius) {
    std::vector<f32> k(2*radius+1);
    f32 s=0;
    for(u32 i=0;i<2*radius+1;++i){ f32 x=f32(i32(i)-i32(radius)); k[i]=std::exp(-x*x/(2*sigma*sigma)); s+=k[i]; }
    for(auto& v:k) v/=s;
    return k;
}

[[nodiscard]] inline ImageBuf gaussian_blur(const ImageBuf& src, f32 sigma) {
    u32 radius=u32(std::ceil(sigma*3));
    auto k=gaussian_kernel_1d(sigma,radius);
    ImageBuf tmp=src.clone(), dst=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 ch=0;ch<src.c;++ch)
        for(u32 x=0;x<src.w;++x) {
            f32 s=0;
            for(u32 i=0;i<k.size();++i) s+=k[i]*src.get_clamped(i32(x)+i32(i)-i32(radius),i32(y),ch);
            tmp.at(x,y,ch)=s;
        }
    for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch)
        for(u32 y=0;y<src.h;++y) {
            f32 s=0;
            for(u32 i=0;i<k.size();++i) s+=k[i]*tmp.get_clamped(i32(x),i32(y)+i32(i)-i32(radius),ch);
            dst.at(x,y,ch)=s;
        }
    return dst;
}

// ─── Bilateral filter ─────────────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf bilateral_filter(const ImageBuf& src,
    f32 sigma_space=5, f32 sigma_range=0.15f)
{
    u32 radius=u32(std::ceil(sigma_space*2));
    ImageBuf dst=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) {
        std::vector<f32> sums(src.c,0); f32 norm=0;
        Vec3f center=src.pixel(x,y);
        for(i32 dy=-i32(radius);dy<=i32(radius);++dy) for(i32 dx=-i32(radius);dx<=i32(radius);++dx) {
            f32 w_s=std::exp(-(dx*dx+dy*dy)/(2*sigma_space*sigma_space));
            Vec3f nb=src.pixel(u32(std::clamp(i32(x)+dx,0,i32(src.w)-1)),u32(std::clamp(i32(y)+dy,0,i32(src.h)-1)));
            Vec3f diff=nb-center; f32 d2=dot(diff,diff);
            f32 w_r=std::exp(-d2/(2*sigma_range*sigma_range));
            f32 w=w_s*w_r; norm+=w;
            for(u32 ch=0;ch<src.c;++ch) sums[ch]+=w*src.get_clamped(i32(x)+dx,i32(y)+dy,ch);
        }
        for(u32 ch=0;ch<src.c;++ch) dst.at(x,y,ch)=sums[ch]/(norm+1e-10f);
    }
    return dst;
}

// ─── Guided filter (fast edge-preserving smoothing) ──────────────────────────
[[nodiscard]] inline ImageBuf guided_filter(const ImageBuf& src, const ImageBuf& guide,
    u32 radius=8, f32 eps=0.01f)
{
    // Per-channel guided filter
    ImageBuf dst(src.w,src.h,src.c);
    for(u32 ch=0;ch<src.c;++ch) {
        // Compute mean_I, mean_p, corr_Ip, corr_II
        u32 r=radius;
        auto box=[&](const ImageBuf& img,u32 ich)->ImageBuf{
            ImageBuf t(img.w,img.h,1);
            for(u32 y=0;y<img.h;++y) for(u32 x=0;x<img.w;++x) t.at(x,y)=img.get_clamped(i32(x),i32(y),ich);
            return box_blur(t,r);
        };
        auto mean_I=box(guide,ch);
        auto mean_p=box(src,ch);
        // corr_Ip = mean(I*p)
        ImageBuf Ip(src.w,src.h,1); for(u32 i=0;i<src.n_pixels();++i) Ip.data[i]=guide.data[i*guide.c+ch]*src.data[i*src.c+ch];
        auto mean_Ip=box_blur(Ip,r);
        // corr_II = mean(I*I)
        ImageBuf II(src.w,src.h,1); for(u32 i=0;i<src.n_pixels();++i) II.data[i]=guide.data[i*guide.c+ch]*guide.data[i*guide.c+ch];
        auto mean_II=box_blur(II,r);
        // a = (corr_Ip - mean_I*mean_p)/(corr_II - mean_I²+eps)
        // b = mean_p - a*mean_I
        ImageBuf a(src.w,src.h,1),b(src.w,src.h,1);
        for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) {
            f32 mI=mean_I.at(x,y),mp=mean_p.at(x,y);
            f32 cov=mean_Ip.at(x,y)-mI*mp;
            f32 var=mean_II.at(x,y)-mI*mI;
            a.at(x,y)=cov/(var+eps);
            b.at(x,y)=mp-a.at(x,y)*mI;
        }
        auto mean_a=box_blur(a,r), mean_b=box_blur(b,r);
        for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x)
            dst.at(x,y,ch)=mean_a.at(x,y)*guide.get_clamped(i32(x),i32(y),ch)+mean_b.at(x,y);
    }
    return dst;
}

// ─── Anisotropic diffusion (Perona-Malik) ─────────────────────────────────────
[[nodiscard]] inline ImageBuf anisotropic_diffusion(const ImageBuf& src,
    u32 iters=20, f32 kappa=50, f32 gamma=0.1f, bool exponential=true)
{
    ImageBuf u=src.clone();
    for(u32 it=0;it<iters;++it) {
        ImageBuf u_new=u.clone();
        for(u32 y=1;y+1<u.h;++y) for(u32 x=1;x+1<u.w;++x) for(u32 ch=0;ch<u.c;++ch) {
            f32 v=u.at(x,y,ch);
            f32 dN=u.at(x,y-1,ch)-v,dS=u.at(x,y+1,ch)-v;
            f32 dW=u.at(x-1,y,ch)-v,dE=u.at(x+1,y,ch)-v;
            // Conductance
            auto c=[&](f32 d)->f32{
                return exponential?std::exp(-(d*d)/(kappa*kappa)):1.f/(1+(d*d)/(kappa*kappa));
            };
            u_new.at(x,y,ch)=v+gamma*(c(dN)*dN+c(dS)*dS+c(dW)*dW+c(dE)*dE);
        }
        u=u_new;
    }
    return u;
}

// ─── Total variation denoising (ROF model, split Bregman) ─────────────────────
[[nodiscard]] inline ImageBuf tv_denoise(const ImageBuf& src, f32 lambda=0.1f, u32 iters=100) {
    ImageBuf u=src.clone();
    ImageBuf bx(src.w,src.h,src.c,0),by(src.w,src.h,src.c,0);
    ImageBuf dx(src.w,src.h,src.c,0),dy(src.w,src.h,src.c,0);
    f32 mu=1.f/lambda;
    for(u32 it=0;it<iters;++it) {
        // u-update: (I + λ div ∇) u = f + λ div(d-b)  — simplified Jacobi
        ImageBuf u_new=u.clone();
        for(u32 y=1;y+1<src.h;++y) for(u32 x=1;x+1<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
            f32 dxb=dx.at(x,y,ch)-bx.at(x,y,ch), dyb=dy.at(x,y,ch)-by.at(x,y,ch);
            f32 div_db=(dxb-( x>0?dx.at(x-1,y,ch)-bx.at(x-1,y,ch):0))+(dyb-(y>0?dy.at(x,y-1,ch)-by.at(x,y-1,ch):0));
            u_new.at(x,y,ch)=(src.at(x,y,ch)+mu*div_db)/(1+mu*4*0.25f);
        }
        u=u_new;
        // d-update: shrinkage
        for(u32 y=0;y+1<src.h;++y) for(u32 x=0;x+1<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
            f32 gx=(u.at(x+1,y,ch)-u.at(x,y,ch))+bx.at(x,y,ch);
            f32 gy=(u.at(x,y+1,ch)-u.at(x,y,ch))+by.at(x,y,ch);
            f32 nm=std::sqrt(gx*gx+gy*gy);
            f32 shrink=std::max(0.f,1-1.f/(mu*nm+1e-10f));
            dx.at(x,y,ch)=shrink*gx; dy.at(x,y,ch)=shrink*gy;
        }
        // b-update
        for(u32 y=0;y+1<src.h;++y) for(u32 x=0;x+1<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
            bx.at(x,y,ch)+=(u.at(x+1,y,ch)-u.at(x,y,ch))-dx.at(x,y,ch);
            by.at(x,y,ch)+=(u.at(x,y+1,ch)-u.at(x,y,ch))-dy.at(x,y,ch);
        }
    }
    return u;
}

// ─── Morphological operations ─────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf erode(const ImageBuf& src, u32 radius=1) {
    ImageBuf dst=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
        f32 mn=1e30f;
        for(i32 dy=-i32(radius);dy<=i32(radius);++dy) for(i32 dx=-i32(radius);dx<=i32(radius);++dx)
            mn=std::min(mn,src.get_clamped(i32(x)+dx,i32(y)+dy,ch));
        dst.at(x,y,ch)=mn;
    }
    return dst;
}

[[nodiscard]] inline ImageBuf dilate(const ImageBuf& src, u32 radius=1) {
    ImageBuf dst=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
        f32 mx=-1e30f;
        for(i32 dy=-i32(radius);dy<=i32(radius);++dy) for(i32 dx=-i32(radius);dx<=i32(radius);++dx)
            mx=std::max(mx,src.get_clamped(i32(x)+dx,i32(y)+dy,ch));
        dst.at(x,y,ch)=mx;
    }
    return dst;
}

[[nodiscard]] inline ImageBuf morphological_open(const ImageBuf& src, u32 r=1) { return dilate(erode(src,r),r); }
[[nodiscard]] inline ImageBuf morphological_close(const ImageBuf& src, u32 r=1) { return erode(dilate(src,r),r); }
[[nodiscard]] inline ImageBuf morphological_gradient(const ImageBuf& src, u32 r=1) {
    auto d=dilate(src,r), e=erode(src,r);
    ImageBuf out=src.clone();
    for(u32 i=0;i<out.data.size();++i) out.data[i]=d.data[i]-e.data[i];
    return out;
}
[[nodiscard]] inline ImageBuf top_hat(const ImageBuf& src, u32 r=3) {
    auto opened=morphological_open(src,r);
    ImageBuf out=src.clone();
    for(u32 i=0;i<out.data.size();++i) out.data[i]=src.data[i]-opened.data[i];
    return out;
}

// ─── Edge detection ───────────────────────────────────────────────────────────
[[nodiscard]] inline std::pair<ImageBuf,ImageBuf> sobel(const ImageBuf& src) {
    ImageBuf gx(src.w,src.h,1),gy(src.w,src.h,1);
    for(u32 y=1;y+1<src.h;++y) for(u32 x=1;x+1<src.w;++x) {
        // Compute luma
        auto luma=[&](i32 px,i32 py)->f32{
            if(src.c==1) return src.get_clamped(px,py);
            return 0.2126f*src.get_clamped(px,py,0)+0.7152f*src.get_clamped(px,py,1)+0.0722f*src.get_clamped(px,py,2);
        };
        gx.at(x,y)=-luma(i32(x)-1,i32(y)-1)-2*luma(i32(x)-1,i32(y))-luma(i32(x)-1,i32(y)+1)
                   +luma(i32(x)+1,i32(y)-1)+2*luma(i32(x)+1,i32(y))+luma(i32(x)+1,i32(y)+1);
        gy.at(x,y)=-luma(i32(x)-1,i32(y)-1)-2*luma(i32(x),i32(y)-1)-luma(i32(x)+1,i32(y)-1)
                   +luma(i32(x)-1,i32(y)+1)+2*luma(i32(x),i32(y)+1)+luma(i32(x)+1,i32(y)+1);
    }
    return {gx,gy};
}

[[nodiscard]] inline ImageBuf sobel_magnitude(const ImageBuf& src) {
    auto [gx,gy]=sobel(src);
    ImageBuf out(src.w,src.h,1);
    for(u32 i=0;i<out.n_pixels();++i) out.data[i]=std::sqrt(gx.data[i]*gx.data[i]+gy.data[i]*gy.data[i]);
    return out;
}

// Canny edge detector
[[nodiscard]] inline ImageBuf canny(const ImageBuf& src, f32 sigma=1.5f,
    f32 low_thresh=0.05f, f32 high_thresh=0.15f)
{
    auto blurred=gaussian_blur(src,sigma);
    auto [gx,gy]=sobel(blurred);
    ImageBuf mag(src.w,src.h,1),angle(src.w,src.h,1);
    for(u32 i=0;i<src.n_pixels();++i) {
        mag.data[i]=std::sqrt(gx.data[i]*gx.data[i]+gy.data[i]*gy.data[i]);
        angle.data[i]=std::atan2(gy.data[i],gx.data[i]);
    }
    // Non-maximum suppression
    ImageBuf nms(src.w,src.h,1);
    for(u32 y=1;y+1<src.h;++y) for(u32 x=1;x+1<src.w;++x) {
        f32 a=angle.at(x,y); f32 m=mag.at(x,y);
        // Quantize angle to 4 directions
        i32 dx=i32(std::round(std::cos(a))),dy=i32(std::round(std::sin(a)));
        f32 m1=mag.get_clamped(i32(x)+dx,i32(y)+dy);
        f32 m2=mag.get_clamped(i32(x)-dx,i32(y)-dy);
        nms.at(x,y)=(m>=m1&&m>=m2)?m:0;
    }
    // Hysteresis thresholding
    ImageBuf edges(src.w,src.h,1);
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) {
        f32 v=nms.at(x,y);
        if(v>=high_thresh) edges.at(x,y)=1;
        else if(v>=low_thresh) {
            // Check if any neighbour is strong
            bool strong=false;
            for(i32 dy=-1;dy<=1&&!strong;++dy) for(i32 dx=-1;dx<=1&&!strong;++dx)
                if(nms.get_clamped(i32(x)+dx,i32(y)+dy)>=high_thresh) strong=true;
            edges.at(x,y)=strong?1.f:0.f;
        }
    }
    return edges;
}

// ─── Unsharp mask ────────────────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf unsharp_mask(const ImageBuf& src, f32 sigma=1.f, f32 amount=0.5f) {
    auto blurred=gaussian_blur(src,sigma);
    ImageBuf out=src.clone();
    for(u32 i=0;i<out.data.size();++i) out.data[i]=src.data[i]+amount*(src.data[i]-blurred.data[i]);
    return out;
}

// ─── Convolution with arbitrary kernel ────────────────────────────────────────
[[nodiscard]] inline ImageBuf convolve2d(const ImageBuf& src, const std::vector<f32>& kernel, u32 kw, u32 kh) {
    ImageBuf dst=src.clone();
    i32 kx=i32(kw)/2,ky=i32(kh)/2;
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
        f32 s=0;
        for(u32 j=0;j<kh;++j) for(u32 i=0;i<kw;++i)
            s+=kernel[j*kw+i]*src.get_clamped(i32(x)+i32(i)-kx,i32(y)+i32(j)-ky,ch);
        dst.at(x,y,ch)=s;
    }
    return dst;
}

// ─── Kuwahara filter (painterly effect) ──────────────────────────────────────
[[nodiscard]] inline ImageBuf kuwahara(const ImageBuf& src, u32 radius=5) {
    ImageBuf dst=src.clone();
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x) for(u32 ch=0;ch<src.c;++ch) {
        i32 r=i32(radius);
        // 4 quadrants: NW, NE, SW, SE
        struct Quad { f32 mean{0},var{0}; };
        auto compute_quad=[&](i32 x0,i32 y0,i32 x1,i32 y1)->Quad{
            f32 s=0,s2=0; u32 n=0;
            for(i32 qy=y0;qy<=y1;++qy) for(i32 qx=x0;qx<=x1;++qx){
                f32 v=src.get_clamped(i32(x)+qx,i32(y)+qy,ch);
                s+=v; s2+=v*v; ++n;
            }
            f32 m=s/n; return {m,s2/n-m*m};
        };
        std::array<Quad,4> quads={
            compute_quad(-r,-r,0,0),compute_quad(0,-r,r,0),
            compute_quad(-r,0,0,r), compute_quad(0,0,r,r)
        };
        u32 best=0; f32 min_var=quads[0].var;
        for(u32 q=1;q<4;++q) if(quads[q].var<min_var){min_var=quads[q].var;best=q;}
        dst.at(x,y,ch)=quads[best].mean;
    }
    return dst;
}

// ─── Frequency domain filtering ───────────────────────────────────────────────
// High-pass via Laplacian of Gaussian
[[nodiscard]] inline ImageBuf log_filter(const ImageBuf& src, f32 sigma=1.5f) {
    // LoG = -∇²G: implement as DOG (difference of Gaussians)
    auto g1=gaussian_blur(src,sigma), g2=gaussian_blur(src,sigma*1.6f);
    ImageBuf out=src.clone();
    for(u32 i=0;i<out.data.size();++i) out.data[i]=g1.data[i]-g2.data[i];
    return out;
}

// ─── Colour space operations ──────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf rgb_to_grayscale(const ImageBuf& src) {
    ImageBuf dst(src.w,src.h,1);
    for(u32 y=0;y<src.h;++y) for(u32 x=0;x<src.w;++x)
        dst.at(x,y)=0.2126f*src.get_clamped(i32(x),i32(y),0)+0.7152f*src.get_clamped(i32(x),i32(y),1)+0.0722f*src.get_clamped(i32(x),i32(y),2);
    return dst;
}

// ─── CLAHE (Contrast-Limited Adaptive Histogram Equalization) ─────────────────
[[nodiscard]] inline ImageBuf clahe(const ImageBuf& src, u32 tile_w=64, u32 tile_h=64, f32 clip_limit=40) {
    ImageBuf dst=src.clone();
    // Process each channel
    for(u32 ch=0;ch<src.c;++ch) {
        for(u32 ty=0;ty<src.h;ty+=tile_h) for(u32 tx=0;tx<src.w;tx+=tile_w) {
            u32 tw=std::min(tile_w,src.w-tx), th=std::min(tile_h,src.h-ty);
            // Build histogram
            std::array<u32,256> hist{}; hist.fill(0);
            for(u32 y=ty;y<ty+th;++y) for(u32 x=tx;x<tx+tw;++x)
                ++hist[u32(std::clamp(src.at(x,y,ch),0.f,1.f)*255)];
            // Clip and redistribute
            u32 limit=u32(clip_limit*tw*th/256);
            u32 excess=0;
            for(auto& h:hist){ if(h>limit){excess+=h-limit;h=limit;} }
            u32 add=excess/256;
            for(auto& h:hist) h+=add;
            // Build CDF
            std::array<f32,256> cdf{};
            u32 cum=0;
            for(u32 i=0;i<256;++i){ cum+=hist[i]; cdf[i]=f32(cum)/(tw*th); }
            // Apply equalization
            for(u32 y=ty;y<ty+th;++y) for(u32 x=tx;x<tx+tw;++x)
                dst.at(x,y,ch)=cdf[u32(std::clamp(src.at(x,y,ch),0.f,1.f)*255)];
        }
    }
    return dst;
}

} // namespace qs
