
// qs_math/random/random_ext_impl.cpp (C++20)
#include "include/qs_math_random.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
namespace qs::math::rng {
constexpr f32 PI = 3.14159265358979323846f;

void stratified_1d(i32 n,f32* s,LCG& rng) noexcept {
    for(i32 i=0;i<n;++i) s[i]=((f32)i+rng.next_f32())/n;
    for(i32 i=n-1;i>0;--i){i32 j=rng.next_range(0,i+1);std::swap(s[i],s[j]);}
}
void stratified_2d(i32 nx,i32 ny,Vec2<f32>* s,LCG& rng) noexcept {
    for(i32 y=0;y<ny;++y)for(i32 x=0;x<nx;++x)
        s[y*nx+x]={((f32)x+rng.next_f32())/nx,((f32)y+rng.next_f32())/ny};
    i32 n=nx*ny;
    for(i32 i=n-1;i>0;--i){i32 j=rng.next_range(0,i+1);std::swap(s[i],s[j]);}
}
void latin_hypercube(i32 n,i32 dim,f64* out,LCG& rng) noexcept {
    for(i32 d=0;d<dim;++d){
        std::vector<i32> perm(n); std::iota(perm.begin(),perm.end(),0);
        for(i32 i=n-1;i>0;--i)std::swap(perm[i],perm[rng.next_range(0,i+1)]);
        for(i32 i=0;i<n;++i) out[i*dim+d]=(perm[i]+rng.next_f64())/n;
    }
}
static f32 radical_inv_base2(u32 n) noexcept {
    n=(n<<16)|(n>>16); n=((n&0x55555555)<<1)|((n>>1)&0x55555555);
    n=((n&0x33333333)<<2)|((n>>2)&0x33333333); n=((n&0x0f0f0f0f)<<4)|((n>>4)&0x0f0f0f0f);
    n=((n&0x00ff00ff)<<8)|((n>>8)&0x00ff00ff);
    return (f32)n*(1.0f/4294967296.0f);
}
Vec2<f32> hammersley(u32 i,u32 n) noexcept { return{(f32)i/n, radical_inv_base2(i)}; }
void hammersley_set(u32 n,Vec2<f32>* out) noexcept { for(u32 i=0;i<n;++i)out[i]=hammersley(i,n); }

Vec3<f32> fibonacci_sphere(u32 i,u32 n) noexcept {
    constexpr f32 PHI=1.6180339887498948482f;
    const f32 theta=std::acos(1-2.0f*(i+0.5f)/n);
    const f32 phi=2*PI*i/PHI;
    return{std::sin(theta)*std::cos(phi),std::cos(theta),std::sin(theta)*std::sin(phi)};
}
void fibonacci_sphere_set(u32 n,Vec3<f32>* out) noexcept { for(u32 i=0;i<n;++i)out[i]=fibonacci_sphere(i,n); }

Vec3<f32> sample_cosine_hemisphere(f32 u1,f32 u2) noexcept {
    const f32 r=std::sqrt(u1),phi=2*PI*u2;
    return{r*std::cos(phi),r*std::sin(phi),std::sqrt(std::max(0.0f,1-u1))};
}
Vec3<f32> sample_uniform_sphere(f32 u1,f32 u2) noexcept {
    const f32 z=1-2*u1,r=std::sqrt(std::max(0.0f,1-z*z)),phi=2*PI*u2;
    return{r*std::cos(phi),r*std::sin(phi),z};
}
Vec3<f32> sample_uniform_hemisphere(f32 u1,f32 u2) noexcept {
    const f32 z=u1,r=std::sqrt(std::max(0.0f,1-z*z)),phi=2*PI*u2;
    return{r*std::cos(phi),r*std::sin(phi),z};
}
f32 pdf_cosine_hemisphere(Vec3<f32> d) noexcept { return std::max(0.0f,d.z)/PI; }

void AliasTable::build(const f32* w,i32 n) noexcept {
    prob.resize(n); alias.resize(n,-1);
    std::vector<f32> norm(w,w+n);
    f32 sum=0;for(auto v:norm)sum+=v;
    for(auto&v:norm)v=v*n/sum;
    std::vector<i32> small,large;
    for(i32 i=0;i<n;++i)(norm[i]<1?small:large).push_back(i);
    while(!small.empty()&&!large.empty()){
        i32 s=small.back();small.pop_back();
        i32 l=large.back();large.pop_back();
        prob[s]=norm[s]; alias[s]=l;
        norm[l]=(norm[l]+norm[s])-1;
        (norm[l]<1?small:large).push_back(l);
    }
    for(auto i:small)prob[i]=1;
    for(auto i:large)prob[i]=1;
}
i32 AliasTable::sample(LCG& rng) const noexcept {
    i32 n=(i32)prob.size();
    const i32 i=rng.next_range(0,n);
    return rng.next_f32()<prob[i]?i:alias[i];
}

static const u32 SOBOL_V[4][32] = {
    {0x80000000,0x40000000,0x20000000,0x10000000,0x08000000,0x04000000,0x02000000,0x01000000,
     0x00800000,0x00400000,0x00200000,0x00100000,0x00080000,0x00040000,0x00020000,0x00010000,
     0x00008000,0x00004000,0x00002000,0x00001000,0x00000800,0x00000400,0x00000200,0x00000100,
     0x00000080,0x00000040,0x00000020,0x00000010,0x00000008,0x00000004,0x00000002,0x00000001},
    {0x80000000,0xc0000000,0xa0000000,0xf0000000,0x88000000,0xcc000000,0xaa000000,0xff000000,
     0x80800000,0xc0c00000,0xa0a00000,0xf0f00000,0x88880000,0xcccc0000,0xaaaa0000,0xffff0000,
     0x80008000,0xc000c000,0xa000a000,0xf000f000,0x88008800,0xcc00cc00,0xaa00aa00,0xff00ff00,
     0x80808080,0xc0c0c0c0,0xa0a0a0a0,0xf0f0f0f0,0x88888888,0xcccccccc,0xaaaaaaaa,0xffffffff},
    {0x80000000,0xc0000000,0x60000000,0x90000000,0xe8000000,0x5c000000,0x8e000000,0xc5000000,
     0x68800000,0x9cc00000,0xee600000,0x55900000,0x80680000,0xc09c0000,0x60ee0000,0x90550000,
     0xe8806800,0x5cc09c00,0x8e60ee00,0xc5905500,0x6880e880,0x9cc05cc0,0xee608e60,0x5590c590,
     0x80686880,0xc09c9cc0,0x60eeee60,0x90555590,0xe880e880,0x5cc05cc0,0x8e608e60,0xc590c590},
    {0x80000000,0x40000000,0x20000000,0x10000000,0x08000000,0x04000000,0x02000000,0x01000000,
     0x00800000,0x00400000,0x00200000,0x00100000,0x00080000,0x00040000,0x00020000,0x00010000,
     0x00008000,0x00004000,0x00002000,0x00001000,0x00000800,0x00000400,0x00000200,0x00000100,
     0x00000080,0x00000040,0x00000020,0x00000010,0x00000008,0x00000004,0x00000002,0x00000001}
};
static u32 sobol_dim(u32& state, const u32* V) noexcept {
    u32 idx=state++; u32 v=0;
    for(i32 b=0;idx;++b,idx>>=1) if(idx&1)v^=V[b];
    return v;
}
Vec2<f32> SobolSeq::next_2d() noexcept {
    static u32 s0=0,s1=0;
    f32 x=(sobol_dim(s0,SOBOL_V[0])^scramble[0])*(1.0f/4294967296.0f);
    f32 y=(sobol_dim(s1,SOBOL_V[1])^scramble[1])*(1.0f/4294967296.0f);
    n++; return{x,y};
}
std::array<f32,4> SobolSeq::next_4d() noexcept {
    static u32 s[4]={0,0,0,0};
    std::array<f32,4> r;
    for(i32 d=0;d<4;++d) r[d]=(sobol_dim(s[d],SOBOL_V[d])^scramble[d])*(1.0f/4294967296.0f);
    n++;return r;
}
} // namespace qs::math::rng
