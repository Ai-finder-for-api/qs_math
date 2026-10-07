#pragma once
// qs_math/imaging/morphology.hpp — Morphological ops, distance transforms, skeletonization
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace qs::math {

// ─── Binary image morphology ──────────────────────────────────────────────────
template<int H, int W>
using BinImage = std::array<std::array<bool,W>,H>;
template<int H, int W>
using GrayImage = std::array<std::array<u8,W>,H>;
template<int H, int W>
using FloatImage = std::array<std::array<f32,W>,H>;

// Structuring element (2*R+1 × 2*R+1 disk)
template<int R>
[[nodiscard]] constexpr std::array<std::array<bool,2*R+1>,2*R+1> disk_se() noexcept {
    std::array<std::array<bool,2*R+1>,2*R+1> se{};
    for(int dy=-R;dy<=R;++dy) for(int dx=-R;dx<=R;++dx)
        se[dy+R][dx+R]=(dx*dx+dy*dy<=R*R);
    return se;
}

// Erosion: dst[p] = min over SE of src[p+offset]
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr BinImage<H,W> erode(
    const BinImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    BinImage<H,W> dst{};
    int rh=SH/2,rw=SW/2;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        bool v=true;
        for(int dy=-rh;dy<=rh&&v;++dy) for(int dx=-rw;dx<=rw&&v;++dx) {
            if(!se[dy+rh][dx+rw]) continue;
            int ny=y+dy,nx=x+dx;
            if(ny<0||ny>=H||nx<0||nx>=W||!src[ny][nx]) v=false;
        }
        dst[y][x]=v;
    }
    return dst;
}

// Dilation: dst[p] = max over SE of src[p-offset]
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr BinImage<H,W> dilate(
    const BinImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    BinImage<H,W> dst{};
    int rh=SH/2,rw=SW/2;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        bool v=false;
        for(int dy=-rh;dy<=rh&&!v;++dy) for(int dx=-rw;dx<=rw&&!v;++dx) {
            if(!se[dy+rh][dx+rw]) continue;
            int ny=y-dy,nx=x-dx;
            if(ny>=0&&ny<H&&nx>=0&&nx<W&&src[ny][nx]) v=true;
        }
        dst[y][x]=v;
    }
    return dst;
}

// Opening: erosion then dilation (removes small objects)
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr BinImage<H,W> opening(
    const BinImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    return dilate<H,W,SH,SW>(erode<H,W,SH,SW>(src,se),se);
}
// Closing: dilation then erosion (fills small holes)
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr BinImage<H,W> closing(
    const BinImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    return erode<H,W,SH,SW>(dilate<H,W,SH,SW>(src,se),se);
}

// ─── Grayscale morphology ─────────────────────────────────────────────────────
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr GrayImage<H,W> gray_erode(
    const GrayImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    GrayImage<H,W> dst{};
    int rh=SH/2,rw=SW/2;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        u8 mn=255;
        for(int dy=-rh;dy<=rh;++dy) for(int dx=-rw;dx<=rw;++dx) {
            if(!se[dy+rh][dx+rw]) continue;
            int ny=y+dy,nx=x+dx;
            if(ny>=0&&ny<H&&nx>=0&&nx<W&&src[ny][nx]<mn) mn=src[ny][nx];
        }
        dst[y][x]=mn;
    }
    return dst;
}
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr GrayImage<H,W> gray_dilate(
    const GrayImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    GrayImage<H,W> dst{};
    int rh=SH/2,rw=SW/2;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        u8 mx=0;
        for(int dy=-rh;dy<=rh;++dy) for(int dx=-rw;dx<=rw;++dx) {
            if(!se[dy+rh][dx+rw]) continue;
            int ny=y-dy,nx=x-dx;
            if(ny>=0&&ny<H&&nx>=0&&nx<W&&src[ny][nx]>mx) mx=src[ny][nx];
        }
        dst[y][x]=mx;
    }
    return dst;
}
// Morphological gradient: dilate - erode
template<int H, int W, int SH, int SW>
[[nodiscard]] constexpr GrayImage<H,W> morph_gradient(
    const GrayImage<H,W>& src,
    const std::array<std::array<bool,SW>,SH>& se) noexcept
{
    auto d=gray_dilate<H,W,SH,SW>(src,se);
    auto e=gray_erode<H,W,SH,SW>(src,se);
    GrayImage<H,W> g{};
    for(int y=0;y<H;++y) for(int x=0;x<W;++x)
        g[y][x]=u8(d[y][x]-e[y][x]);
    return g;
}

// ─── Euclidean distance transform (2D, exact) ─────────────────────────────────
// Meijster algorithm (O(HW))
template<int H, int W>
[[nodiscard]] constexpr FloatImage<H,W> distance_transform(const BinImage<H,W>& src) noexcept {
    FloatImage<H,W> dt{};
    constexpr f32 INF=1e9f;
    // Phase 1: column-wise
    std::array<std::array<f32,W>,H> g{};
    for(int x=0;x<W;++x) {
        // Forward scan
        g[0][x]=src[0][x]?0:INF;
        for(int y=1;y<H;++y) g[y][x]=src[y][x]?0:g[y-1][x]+1;
        // Backward scan
        for(int y=H-2;y>=0;--y) g[y][x]=std::min(g[y][x],g[y+1][x]+1);
    }
    // Phase 2: row-wise parabola envelope
    for(int y=0;y<H;++y) {
        std::array<int,W> s{},t{};
        std::array<f32,W> f{};
        for(int x=0;x<W;++x) f[x]=g[y][x]*g[y][x];
        int q=0; s[0]=0; t[0]=0;
        for(int u=1;u<W;++u) {
            while(q>=0) {
                f32 sep=(f32)(u*u-s[q]*s[q]+f[u]-f[s[q]])/(2*(u-s[q]));
                if(sep>t[q]) break;
                --q;
            }
            ++q; s[q]=u; t[q]=(int)((f32)(u*u-s[q-1<0?0:q-1]*s[q-1<0?0:q-1]+f[u]-f[s[q-1<0?0:q-1]])/(2*(u-s[q-1<0?0:q-1]))+0.5f);
        }
        q=0;
        for(int u=0;u<W;++u) {
            while(q<W-1&&t[q+1]<u) ++q;
            f32 d=u-s[q]; dt[y][u]=std::sqrt(d*d+f[s[q]]);
        }
    }
    return dt;
}

// ─── Connected components (4-connectivity, bitmask BFS) ───────────────────────
template<int H, int W>
[[nodiscard]] constexpr int connected_components(
    const BinImage<H,W>& src,
    std::array<std::array<int,W>,H>& labels) noexcept
{
    for(auto& r:labels) r.fill(-1);
    int n_comp=0;
    std::array<std::pair<int,int>,H*W> queue{};
    for(int sy=0;sy<H;++sy) for(int sx=0;sx<W;++sx) {
        if(!src[sy][sx]||labels[sy][sx]>=0) continue;
        int qh=0,qt=0;
        queue[qt++]={sy,sx};
        labels[sy][sx]=n_comp;
        while(qh<qt) {
            auto [y,x]=queue[qh++];
            const int dy[]={-1,1,0,0},dx[]={0,0,-1,1};
            for(int d=0;d<4;++d) {
                int ny=y+dy[d],nx=x+dx[d];
                if(ny>=0&&ny<H&&nx>=0&&nx<W&&src[ny][nx]&&labels[ny][nx]<0) {
                    labels[ny][nx]=n_comp;
                    queue[qt++]={ny,nx};
                }
            }
        }
        ++n_comp;
    }
    return n_comp;
}

// ─── Watershed segmentation seed ─────────────────────────────────────────────
// Marker-based: flood from seeds (simplified topographic scan)
template<int H, int W>
[[nodiscard]] constexpr std::array<std::array<int,W>,H> watershed_markers(
    const GrayImage<H,W>& grad,
    const std::array<std::array<int,W>,H>& seeds) noexcept
{
    auto labels=seeds;
    // Priority queue simulation: process in order of gradient magnitude
    // Simplified: iterative flooding from seeds
    for(int iter=0;iter<H*W;++iter) {
        bool changed=false;
        for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x) {
            if(labels[y][x]>=0) continue;
            const int dy[]={-1,1,0,0},dx[]={0,0,-1,1};
            int best_label=-1; u8 best_grad=255;
            for(int d=0;d<4;++d) {
                int ny=y+dy[d],nx=x+dx[d];
                if(labels[ny][nx]>=0&&grad[ny][nx]<best_grad) {
                    best_grad=grad[ny][nx]; best_label=labels[ny][nx];
                }
            }
            if(best_label>=0){labels[y][x]=best_label;changed=true;}
        }
        if(!changed) break;
    }
    return labels;
}
} // namespace qs::math
