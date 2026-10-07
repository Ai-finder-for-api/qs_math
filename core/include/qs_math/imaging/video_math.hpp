#pragma once
// qs_math/video_math.hpp — Frame timing, motion vectors, temporal coherence, chroma subsampling (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "image_filters.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <numbers>

namespace qs {

// ─── Frame timing ────────────────────────────────────────────────────────────
struct FrameTiming {
    f64 fps{24};
    f64 frame_duration_s{1.0/24};
    u64 frame_number{0};
    f64 presentation_time_s{0};
    f64 decode_time_s{0};

    [[nodiscard]] f64 timestamp_s() const { return f64(frame_number)/fps; }
    [[nodiscard]] u64 pts_90khz() const { return u64(presentation_time_s*90000); }
    [[nodiscard]] u64 pts_27mhz() const { return u64(presentation_time_s*27000000); }

    // Convert between frame number and timecode HH:MM:SS:FF
    [[nodiscard]] std::array<u32,4> to_timecode() const {
        u64 fn=frame_number;
        u32 frames=u32(fn%u64(std::round(fps)));
        u64 total_sec=fn/u64(std::round(fps));
        u32 secs=u32(total_sec%60), mins=u32((total_sec/60)%60), hrs=u32(total_sec/3600);
        return {hrs,mins,secs,frames};
    }

    static FrameTiming from_timecode(u32 h,u32 m,u32 s,u32 f,f64 fps_) {
        FrameTiming t; t.fps=fps_;
        t.frame_number=u64((h*3600+m*60+s)*u32(std::round(fps_))+f);
        t.presentation_time_s=f64(t.frame_number)/fps_;
        return t;
    }

    // Drop-frame timecode correction (29.97 NTSC)
    [[nodiscard]] std::array<u32,4> to_drop_frame_timecode() const {
        u64 fn=frame_number;
        u64 d=fn/17982; u64 m_=fn%17982;
        u64 skip=2*(d*9+(m_>1?m_/1798:0));
        fn+=skip;
        u32 frames=u32(fn%30);
        u64 ts=fn/30;
        u32 ss=u32(ts%60),mm=u32((ts/60)%60),hh=u32(ts/3600);
        return {hh,mm,ss,frames};
    }
};

// Frame rate conversion: interpolation factor between two frames
[[nodiscard]] inline f32 frame_interpolation_factor(f64 src_fps, f64 dst_fps, u64 dst_frame) {
    f64 src_time=f64(dst_frame)/dst_fps;
    f64 src_frame=src_time*src_fps;
    return f32(src_frame-std::floor(src_frame));
}

// ─── Motion estimation ────────────────────────────────────────────────────────
struct MotionVector { i32 dx{0},dy{0}; f32 sad{0}; };

// Sum of Absolute Differences (SAD) for block matching
[[nodiscard]] inline f32 block_sad(const ImageBuf& ref, const ImageBuf& cur,
    u32 bx,u32 by,u32 bw,u32 bh,i32 dx,i32 dy)
{
    f32 sad=0;
    for(u32 y=0;y<bh;++y) for(u32 x=0;x<bw;++x) for(u32 ch=0;ch<ref.c;++ch)
        sad+=std::abs(cur.get_clamped(i32(bx+x),i32(by+y),ch)-ref.get_clamped(i32(bx+x)+dx,i32(by+y)+dy,ch));
    return sad;
}

// Full search block matching
[[nodiscard]] inline MotionVector full_search_bm(const ImageBuf& ref, const ImageBuf& cur,
    u32 bx,u32 by,u32 bw,u32 bh,u32 search_range)
{
    MotionVector best; best.sad=1e30f;
    for(i32 dy=-i32(search_range);dy<=i32(search_range);++dy)
        for(i32 dx=-i32(search_range);dx<=i32(search_range);++dx) {
            f32 sad=block_sad(ref,cur,bx,by,bw,bh,dx,dy);
            if(sad<best.sad){ best={dx,dy,sad}; }
        }
    return best;
}

// Diamond search (fast block matching)
[[nodiscard]] inline MotionVector diamond_search(const ImageBuf& ref, const ImageBuf& cur,
    u32 bx,u32 by,u32 bw,u32 bh,u32 max_range)
{
    MotionVector best; best.sad=block_sad(ref,cur,bx,by,bw,bh,0,0);
    static const i32 LARGE[8][2]={{0,-2},{0,2},{-2,0},{2,0},{-1,-1},{1,-1},{-1,1},{1,1}};
    static const i32 SMALL[4][2]={{0,-1},{0,1},{-1,0},{1,0}};
    bool improved=true;
    while(improved&&(std::abs(best.dx)<i32(max_range)&&std::abs(best.dy)<i32(max_range))) {
        improved=false;
        for(auto& d:LARGE) {
            i32 tx=best.dx+d[0],ty=best.dy+d[1];
            f32 sad=block_sad(ref,cur,bx,by,bw,bh,tx,ty);
            if(sad<best.sad){ best={tx,ty,sad}; improved=true; }
        }
    }
    for(auto& d:SMALL) {
        i32 tx=best.dx+d[0],ty=best.dy+d[1];
        f32 sad=block_sad(ref,cur,bx,by,bw,bh,tx,ty);
        if(sad<best.sad) best={tx,ty,sad};
    }
    return best;
}

// Sub-pixel motion estimation (half-pixel via bilinear interpolation)
[[nodiscard]] inline f32 subpixel_sad(const ImageBuf& ref, const ImageBuf& cur,
    u32 bx,u32 by,u32 bw,u32 bh,f32 dx,f32 dy)
{
    f32 sad=0;
    for(u32 y=0;y<bh;++y) for(u32 x=0;x<bw;++x) for(u32 ch=0;ch<ref.c;++ch) {
        f32 rx=f32(bx+x)+dx, ry=f32(by+y)+dy;
        // Bilinear sample from ref
        i32 rx0=i32(rx),ry0=i32(ry);
        f32 fx=rx-rx0,fy=ry-ry0;
        f32 r00=ref.get_clamped(rx0,  ry0,  ch),r10=ref.get_clamped(rx0+1,ry0,  ch);
        f32 r01=ref.get_clamped(rx0,  ry0+1,ch),r11=ref.get_clamped(rx0+1,ry0+1,ch);
        f32 ref_val=(r00*(1-fx)+r10*fx)*(1-fy)+(r01*(1-fx)+r11*fx)*fy;
        sad+=std::abs(cur.get_clamped(i32(bx+x),i32(by+y),ch)-ref_val);
    }
    return sad;
}

// Motion vector field (one MV per macro-block)
struct MVField {
    std::vector<MotionVector> mvs;
    u32 width_mbs{0}, height_mbs{0};
    u32 mb_size{16};

    MVField(u32 w,u32 h,u32 mb=16):width_mbs(w),height_mbs(h),mb_size(mb),mvs(w*h){}
    [[nodiscard]] MotionVector& at(u32 x,u32 y){ return mvs[y*width_mbs+x]; }
    [[nodiscard]] const MotionVector& at(u32 x,u32 y)const{ return mvs[y*width_mbs+x]; }

    // Median predictor for neighboring MVs
    [[nodiscard]] MotionVector median_predictor(u32 mbx,u32 mby) const {
        std::vector<i32> xs,ys;
        if(mbx>0){xs.push_back(at(mbx-1,mby).dx);ys.push_back(at(mbx-1,mby).dy);}
        if(mby>0){xs.push_back(at(mbx,mby-1).dx);ys.push_back(at(mbx,mby-1).dy);}
        if(mbx>0&&mby>0){xs.push_back(at(mbx-1,mby-1).dx);ys.push_back(at(mbx-1,mby-1).dy);}
        if(xs.empty()) return {0,0,0};
        std::sort(xs.begin(),xs.end()); std::sort(ys.begin(),ys.end());
        return {xs[xs.size()/2],ys[ys.size()/2],0};
    }

    // Estimate bits needed for MV coding (Exp-Golomb)
    [[nodiscard]] u32 coding_bits() const {
        u32 bits=0;
        for(const auto& mv:mvs) {
            auto eg=[](i32 v)->u32{ u32 k=u32(std::abs(v)*2+(v<0?1:0)); return 2*u32(std::log2(k+1)+1)-1; };
            bits+=eg(mv.dx)+eg(mv.dy);
        }
        return bits;
    }
};

// ─── Temporal coherence ───────────────────────────────────────────────────────

// Temporal median filter (across frames)
[[nodiscard]] inline ImageBuf temporal_median(const std::vector<ImageBuf>& frames) {
    assert(!frames.empty());
    u32 n=u32(frames.size()),w=frames[0].w,h=frames[0].h,c=frames[0].c;
    ImageBuf out(w,h,c);
    for(u32 y=0;y<h;++y) for(u32 x=0;x<w;++x) for(u32 ch=0;ch<c;++ch) {
        std::vector<f32> vals(n);
        for(u32 i=0;i<n;++i) vals[i]=frames[i].at(x,y,ch);
        std::sort(vals.begin(),vals.end());
        out.at(x,y,ch)=vals[n/2];
    }
    return out;
}

// Temporal exponential moving average (for flicker reduction)
inline void temporal_ema_update(ImageBuf& accumulator, const ImageBuf& frame, f32 alpha=0.1f) {
    for(u32 i=0;i<accumulator.data.size()&&i<frame.data.size();++i)
        accumulator.data[i]=alpha*frame.data[i]+(1-alpha)*accumulator.data[i];
}

// Scene change detection (histogram comparison)
[[nodiscard]] inline f32 scene_change_score(const ImageBuf& prev, const ImageBuf& cur, u32 bins=64) {
    std::vector<f32> hist_prev(bins,0),hist_cur(bins,0);
    for(u32 i=0;i<prev.n_pixels();++i) {
        // Luma of each pixel
        f32 lp=0,lc=0;
        for(u32 ch=0;ch<prev.c;++ch){lp+=prev.data[i*prev.c+ch];lc+=cur.data[i*cur.c+ch];}
        lp/=prev.c; lc/=cur.c;
        hist_prev[u32(std::clamp(lp,0.f,1.f)*(bins-1))]+=1;
        hist_cur [u32(std::clamp(lc,0.f,1.f)*(bins-1))]+=1;
    }
    f32 total=f32(prev.n_pixels());
    for(auto& v:hist_prev) v/=total; for(auto& v:hist_cur) v/=total;
    // Chi-squared distance
    f32 chi2=0;
    for(u32 i=0;i<bins;++i){ f32 s=hist_prev[i]+hist_cur[i]; if(s>1e-10f) chi2+=(hist_prev[i]-hist_cur[i])*(hist_prev[i]-hist_cur[i])/s; }
    return chi2;
}

// ─── Chroma subsampling ───────────────────────────────────────────────────────
// Convert RGB to YCbCr (BT.601)
[[nodiscard]] inline Vec3f rgb_to_ycbcr_601(Vec3f rgb) {
    return {
         0.299f*rgb.x+0.587f*rgb.y+0.114f*rgb.z,
        -0.169f*rgb.x-0.331f*rgb.y+0.500f*rgb.z+0.5f,
         0.500f*rgb.x-0.419f*rgb.y-0.081f*rgb.z+0.5f
    };
}
[[nodiscard]] inline Vec3f ycbcr_601_to_rgb(Vec3f ycbcr) {
    f32 Y=ycbcr.x,Cb=ycbcr.y-0.5f,Cr=ycbcr.z-0.5f;
    return {Y+1.402f*Cr, Y-0.344f*Cb-0.714f*Cr, Y+1.772f*Cb};
}

// YCbCr BT.709
[[nodiscard]] inline Vec3f rgb_to_ycbcr_709(Vec3f rgb) {
    return {
         0.2126f*rgb.x+0.7152f*rgb.y+0.0722f*rgb.z,
        -0.1146f*rgb.x-0.3854f*rgb.y+0.5000f*rgb.z+0.5f,
         0.5000f*rgb.x-0.4542f*rgb.y-0.0458f*rgb.z+0.5f
    };
}
[[nodiscard]] inline Vec3f ycbcr_709_to_rgb(Vec3f ycbcr) {
    f32 Y=ycbcr.x,Cb=ycbcr.y-0.5f,Cr=ycbcr.z-0.5f;
    return {Y+1.5748f*Cr, Y-0.1873f*Cb-0.4681f*Cr, Y+1.8556f*Cb};
}

// 4:2:0 subsample (H.264/HEVC style): decimate Cb/Cr by 2 in both directions
struct YUV420 {
    std::vector<f32> Y,Cb,Cr;
    u32 width{0},height{0};

    static YUV420 from_rgb(const ImageBuf& rgb) {
        YUV420 yuv; yuv.width=rgb.w; yuv.height=rgb.h;
        yuv.Y.resize(rgb.w*rgb.h);
        yuv.Cb.resize((rgb.w/2)*(rgb.h/2));
        yuv.Cr.resize((rgb.w/2)*(rgb.h/2));
        for(u32 y=0;y<rgb.h;++y) for(u32 x=0;x<rgb.w;++x) {
            Vec3f ycbcr=rgb_to_ycbcr_709(rgb.pixel(x,y));
            yuv.Y[y*rgb.w+x]=ycbcr.x;
            if(x%2==0&&y%2==0) {
                u32 ci=(y/2)*(rgb.w/2)+x/2;
                yuv.Cb[ci]=ycbcr.y;
                yuv.Cr[ci]=ycbcr.z;
            }
        }
        return yuv;
    }

    [[nodiscard]] ImageBuf to_rgb() const {
        ImageBuf rgb(width,height,3);
        for(u32 y=0;y<height;++y) for(u32 x=0;x<width;++x) {
            u32 ci=(y/2)*(width/2)+x/2;
            Vec3f ycbcr={Y[y*width+x], Cb.size()>ci?Cb[ci]:0.5f, Cr.size()>ci?Cr[ci]:0.5f};
            Vec3f col=ycbcr_709_to_rgb(ycbcr);
            rgb.set_pixel(x,y,{std::clamp(col.x,0.f,1.f),std::clamp(col.y,0.f,1.f),std::clamp(col.z,0.f,1.f)});
        }
        return rgb;
    }

    // PSNR between two YUV420 buffers (luma only)
    [[nodiscard]] f64 psnr_y(const YUV420& other) const {
        if(Y.size()!=other.Y.size()) return 0;
        f64 mse=0;
        for(u32 i=0;i<u32(Y.size());++i){ f64 d=Y[i]-other.Y[i]; mse+=d*d; }
        mse/=Y.size();
        return mse>1e-10?10*std::log10(1.0/mse):100.0;
    }

    [[nodiscard]] f64 ssim_y(const YUV420& other) const {
        if(Y.size()!=other.Y.size()) return 0;
        // Simplified SSIM
        f64 mu1=0,mu2=0,sig1=0,sig2=0,sig12=0;
        u32 n=u32(Y.size());
        for(u32 i=0;i<n;++i){ mu1+=Y[i]; mu2+=other.Y[i]; }
        mu1/=n; mu2/=n;
        for(u32 i=0;i<n;++i){
            sig1+=(Y[i]-mu1)*(Y[i]-mu1);
            sig2+=(other.Y[i]-mu2)*(other.Y[i]-mu2);
            sig12+=(Y[i]-mu1)*(other.Y[i]-mu2);
        }
        sig1/=n; sig2/=n; sig12/=n;
        static constexpr f64 C1=0.01*0.01,C2=0.03*0.03;
        return (2*mu1*mu2+C1)*(2*sig12+C2)/((mu1*mu1+mu2*mu2+C1)*(sig1+sig2+C2));
    }
};

// ─── Video codec bitrate math ─────────────────────────────────────────────────
// Target bits per frame given quality and complexity
[[nodiscard]] inline f64 cbr_bits_per_frame(f64 target_kbps, f64 fps) {
    return target_kbps*1000/fps;
}

// H.264/HEVC QP to estimated PSNR (empirical)
[[nodiscard]] inline f64 qp_to_psnr(f32 qp, bool is_hevc=false) {
    // Rough approximation: PSNR ≈ A - B*QP
    f64 A=is_hevc?45.0:44.0, B=is_hevc?0.7:0.75;
    return std::max(0.0,A-B*qp);
}

// Rate-distortion curve point (Shannon bound approximation)
[[nodiscard]] inline f64 rate_for_distortion(f64 distortion, f64 sigma2=0.1) {
    if(distortion>=sigma2) return 0;
    return 0.5*std::log2(sigma2/distortion);
}

// Bjontegaard delta metrics (BD-Rate / BD-PSNR)
struct BDPoint { f64 rate_bps; f64 psnr_dB; };
[[nodiscard]] inline f64 bd_psnr(const std::vector<BDPoint>& anchor,
    const std::vector<BDPoint>& test)
{
    // Piecewise cubic interpolation — simplified: average PSNR difference at same rate points
    assert(anchor.size()==test.size()&&anchor.size()>=4);
    f64 sum=0; u32 n=std::min(u32(anchor.size()),u32(test.size()));
    for(u32 i=0;i<n;++i) sum+=test[i].psnr_dB-anchor[i].psnr_dB;
    return sum/n;
}

// ─── Deinterlacing math ───────────────────────────────────────────────────────
[[nodiscard]] inline ImageBuf bob_deinterlace(const ImageBuf& field, bool is_top_field=true) {
    // Simple bob: duplicate field lines
    ImageBuf out(field.w,field.h*2,field.c);
    for(u32 y=0;y<field.h;++y) for(u32 x=0;x<field.w;++x) for(u32 ch=0;ch<field.c;++ch) {
        u32 oy=is_top_field?y*2:y*2+1;
        out.at(x,oy,ch)=field.at(x,y,ch);
        u32 oy2=is_top_field?y*2+1:y*2;
        if(oy2<out.h) out.at(x,oy2,ch)=field.at(x,y,ch);
    }
    return out;
}

[[nodiscard]] inline ImageBuf weave_fields(const ImageBuf& top, const ImageBuf& bot) {
    ImageBuf out(top.w,top.h+bot.h,top.c);
    for(u32 y=0;y<top.h;++y) for(u32 x=0;x<top.w;++x) for(u32 ch=0;ch<top.c;++ch) {
        out.at(x,y*2,ch)=top.at(x,y,ch);
        if(y<bot.h) out.at(x,y*2+1,ch)=bot.at(x,y,ch);
    }
    return out;
}

} // namespace qs
