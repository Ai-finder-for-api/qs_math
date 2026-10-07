#pragma once
// qs_math/imaging/image_transforms.hpp — Hough, Radon, log-polar, affine warp, optical flow
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Hough transform (line detection) ─────────────────────────────────────────
// Each edge point (x,y) votes for lines: ρ = x*cos(θ) + y*sin(θ)
template<int H, int W, int N_THETA, int N_RHO>
[[nodiscard]] constexpr std::array<std::array<int,N_RHO>,N_THETA> hough_lines(
    const std::array<std::array<bool,W>,H>& edges,
    f64 theta_min=0, f64 theta_max=3.14159265358979323846,
    f64 rho_max=-1) noexcept
{
    f64 rho_m=(rho_max<0)?std::sqrt(f64(H*H+W*W)):rho_max;
    std::array<std::array<int,N_RHO>,N_THETA> acc{};
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        if(!edges[y][x]) continue;
        for(int t=0;t<N_THETA;++t) {
            f64 theta=theta_min+(theta_max-theta_min)*t/N_THETA;
            f64 rho=x*std::cos(theta)+y*std::sin(theta);
            int ri=int((rho+rho_m)/(2*rho_m)*N_RHO);
            if(ri>=0&&ri<N_RHO) ++acc[t][ri];
        }
    }
    return acc;
}

// Hough circle transform: votes for (cx,cy,r)
template<int H, int W>
constexpr void hough_circles_vote(
    std::array<std::array<std::array<int,20>,W>,H>& acc, // [cy][cx][r_idx]
    const std::array<std::array<bool,W>,H>& edges,
    int r_min, int r_max) noexcept
{
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        if(!edges[y][x]) continue;
        for(int r=r_min;r<=r_max;++r) {
            int ri=(r-r_min)*20/(r_max-r_min+1);
            for(int t=0;t<32;++t) {
                f64 theta=2*3.14159265358979323846*t/32;
                int cx=int(x+r*std::cos(theta));
                int cy=int(y+r*std::sin(theta));
                if(cx>=0&&cx<W&&cy>=0&&cy<H) ++acc[cy][cx][ri];
            }
        }
    }
}

// ─── Radon transform (X-ray projection) ───────────────────────────────────────
// R(ρ,θ) = ∫∫ f(x,y) δ(x*cosθ + y*sinθ - ρ) dx dy
template<int H, int W, int N_PROJ>
[[nodiscard]] constexpr std::array<std::array<f64,H+W>,N_PROJ> radon_transform(
    const std::array<std::array<f64,W>,H>& img) noexcept
{
    std::array<std::array<f64,H+W>,N_PROJ> sinogram{};
    f64 cx=W*0.5,cy=H*0.5;
    int n_rho=H+W;
    f64 rho_max=std::sqrt(cx*cx+cy*cy);
    for(int t=0;t<N_PROJ;++t) {
        f64 theta=3.14159265358979323846*t/N_PROJ;
        f64 cth=std::cos(theta),sth=std::sin(theta);
        for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
            f64 rho=(x-cx)*cth+(y-cy)*sth;
            int ri=int((rho+rho_max)/(2*rho_max)*n_rho);
            if(ri>=0&&ri<n_rho) sinogram[t][ri]+=img[y][x];
        }
    }
    return sinogram;
}

// ─── Log-polar transform ──────────────────────────────────────────────────────
// Maps (x,y) → (log r, θ): useful for scale-rotation invariance
template<int H, int W, int LH, int LW>
[[nodiscard]] constexpr std::array<std::array<f64,LW>,LH> log_polar(
    const std::array<std::array<f64,W>,H>& img,
    f64 cx, f64 cy, f64 log_base=std::numbers::e) noexcept
{
    std::array<std::array<f64,LW>,LH> out{};
    f64 r_max=std::sqrt((std::max(cx,f64(W)-cx))*(std::max(cx,f64(W)-cx))
                       +(std::max(cy,f64(H)-cy))*(std::max(cy,f64(H)-cy)));
    f64 log_max=std::log(r_max)/std::log(log_base);
    for(int lr=0;lr<LH;++lr) for(int lth=0;lth<LW;++lth) {
        f64 log_r=log_max*lr/LH;
        f64 r=std::pow(log_base,log_r);
        f64 theta=2*3.14159265358979323846*lth/LW;
        f64 x=cx+r*std::cos(theta),y=cy+r*std::sin(theta);
        int xi=int(x),yi=int(y);
        if(xi>=0&&xi<W-1&&yi>=0&&yi<H-1) {
            f64 tx=x-xi,ty=y-yi;
            out[lr][lth]=img[yi][xi]*(1-tx)*(1-ty)+img[yi][xi+1]*tx*(1-ty)
                         +img[yi+1][xi]*(1-tx)*ty+img[yi+1][xi+1]*tx*ty;
        }
    }
    return out;
}

// ─── Affine warp ──────────────────────────────────────────────────────────────
// Transform matrix M (2×3): [dst_x,dst_y] = M*[src_x,src_y,1]
template<int DH, int DW, int SH, int SW>
[[nodiscard]] constexpr std::array<std::array<f64,DW>,DH> affine_warp(
    const std::array<std::array<f64,SW>,SH>& src,
    const std::array<std::array<f64,3>,2>& M_inv) noexcept // inverse map
{
    std::array<std::array<f64,DW>,DH> dst{};
    for(int dy=0;dy<DH;++dy) for(int dx=0;dx<DW;++dx) {
        f64 sx=M_inv[0][0]*dx+M_inv[0][1]*dy+M_inv[0][2];
        f64 sy=M_inv[1][0]*dx+M_inv[1][1]*dy+M_inv[1][2];
        int xi=int(sx),yi=int(sy);
        if(xi>=0&&xi<SW-1&&yi>=0&&yi<SH-1) {
            f64 tx=sx-xi,ty=sy-yi;
            dst[dy][dx]=src[yi][xi]*(1-tx)*(1-ty)+src[yi][xi+1]*tx*(1-ty)
                        +src[yi+1][xi]*(1-tx)*ty+src[yi+1][xi+1]*tx*ty;
        }
    }
    return dst;
}

// ─── Optical flow (Lucas-Kanade) ──────────────────────────────────────────────
// Solve [Σ Ix², Σ IxIy; Σ IxIy, Σ Iy²][u,v]' = -[Σ IxIt, Σ IyIt]
template<int H, int W>
[[nodiscard]] constexpr std::pair<f64,f64> lucas_kanade(
    const std::array<std::array<f64,W>,H>& I1,
    const std::array<std::array<f64,W>,H>& I2,
    int cx, int cy, int win=5) noexcept
{
    f64 Ix2=0,Iy2=0,IxIy=0,IxIt=0,IyIt=0;
    for(int dy=-win;dy<=win;++dy) for(int dx=-win;dx<=win;++dx) {
        int y=cy+dy,x=cx+dx;
        if(y<=0||y>=H-1||x<=0||x>=W-1) continue;
        f64 Ix=(I1[y][x+1]-I1[y][x-1])*0.5;
        f64 Iy=(I1[y+1][x]-I1[y-1][x])*0.5;
        f64 It=I2[y][x]-I1[y][x];
        Ix2+=Ix*Ix;Iy2+=Iy*Iy;IxIy+=Ix*Iy;IxIt+=Ix*It;IyIt+=Iy*It;
    }
    f64 det=Ix2*Iy2-IxIy*IxIy;
    if(std::abs(det)<1e-15) return{0,0};
    return{(-IxIt*Iy2+IyIt*IxIy)/det,(-IyIt*Ix2+IxIt*IxIy)/det};
}

// ─── Image pyramid ────────────────────────────────────────────────────────────
// Gaussian downsampling (half resolution)
template<int H, int W>
[[nodiscard]] constexpr std::array<std::array<f64,W/2>,H/2> pyramid_down(
    const std::array<std::array<f64,W>,H>& img) noexcept
{
    std::array<std::array<f64,W/2>,H/2> out{};
    // 5-tap Gaussian kernel weights: [1,4,6,4,1]/16
    for(int y=0;y<H/2;++y) for(int x=0;x<W/2;++x) {
        int sy=2*y,sx=2*x;
        f64 s=0,w=0;
        const int ky[]={-2,-1,0,1,2};
        const f64 kw[]={1,4,6,4,1};
        for(int dy=0;dy<5;++dy) for(int dx=0;dx<5;++dx) {
            int ny=sy+ky[dy],nx=sx+ky[dx];
            if(ny<0||ny>=H||nx<0||nx>=W) continue;
            s+=kw[dy]*kw[dx]*img[ny][nx];
            w+=kw[dy]*kw[dx];
        }
        out[y][x]=s/(w+1e-15);
    }
    return out;
}

// ─── SIFT-like keypoint scale ─────────────────────────────────────────────────
// DoG (difference of Gaussians) at scale σ
template<int H, int W>
[[nodiscard]] constexpr std::array<std::array<f64,W>,H> gaussian_blur(
    const std::array<std::array<f64,W>,H>& img, f64 sigma) noexcept
{
    std::array<std::array<f64,W>,H> out{};
    int r=int(3*sigma);
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        f64 s=0,w=0;
        for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
            int ny=y+dy,nx=x+dx;
            if(ny<0||ny>=H||nx<0||nx>=W) continue;
            f64 g=std::exp(-(dy*dy+dx*dx)/(2*sigma*sigma));
            s+=g*img[ny][nx]; w+=g;
        }
        out[y][x]=s/(w+1e-15);
    }
    return out;
}
} // namespace qs::math
