#pragma once
// qs_math/compression_math.hpp — Entropy coding math, DCT, wavelet coefficients (C++20)
#include "scalar_types.hpp"
#include "fourier_math.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <numbers>

namespace qs {

// ─── Entropy measures for compression ────────────────────────────────────────
[[nodiscard]] inline f64 symbol_entropy_bits(const std::vector<u32>& counts) {
    u64 total=0; for(auto c:counts) total+=c;
    f64 H=0;
    for(auto c:counts) if(c>0){ f64 p=f64(c)/total; H-=p*std::log2(p); }
    return H;
}

// Kolmogorov-Ziv complexity lower bound (LZ78 phrase count)
[[nodiscard]] inline u32 lz78_phrase_count(const std::vector<u8>& data) {
    std::unordered_map<std::string,bool> dict;
    std::string cur;
    u32 phrases=0;
    for(u8 b:data) {
        cur+=char(b);
        if(!dict.count(cur)){ dict[cur]=true; cur=""; ++phrases; }
    }
    if(!cur.empty()) ++phrases;
    return phrases;
}

// ─── Huffman coding math ──────────────────────────────────────────────────────
struct HuffNode {
    u32 symbol{0}; f64 prob{0};
    i32 left{-1},right{-1};
    bool is_leaf{true};
};

struct HuffmanTree {
    std::vector<HuffNode> nodes;
    std::vector<std::vector<u8>> codes; // bit codes per symbol

    static HuffmanTree build(const std::vector<f64>& probs) {
        u32 n=u32(probs.size());
        HuffmanTree tree;
        tree.nodes.resize(n);
        for(u32 i=0;i<n;++i){ tree.nodes[i].symbol=i; tree.nodes[i].prob=probs[i]; }
        using P=std::pair<f64,u32>;
        std::priority_queue<P,std::vector<P>,std::greater<P>> pq;
        for(u32 i=0;i<n;++i) pq.push({probs[i],i});
        while(pq.size()>1) {
            auto [p1,i1]=pq.top(); pq.pop();
            auto [p2,i2]=pq.top(); pq.pop();
            HuffNode merged; merged.prob=p1+p2; merged.left=i32(i1); merged.right=i32(i2); merged.is_leaf=false;
            u32 idx=u32(tree.nodes.size()); tree.nodes.push_back(merged);
            pq.push({merged.prob,idx});
        }
        // Generate codes via DFS
        tree.codes.resize(n);
        std::function<void(i32,std::vector<u8>)> dfs=[&](i32 node,std::vector<u8> code){
            if(node<0) return;
            if(tree.nodes[u32(node)].is_leaf){ tree.codes[tree.nodes[u32(node)].symbol]=code; return; }
            auto lc=code; lc.push_back(0); dfs(tree.nodes[u32(node)].left,lc);
            auto rc=code; rc.push_back(1); dfs(tree.nodes[u32(node)].right,rc);
        };
        if(!tree.nodes.empty()) dfs(i32(tree.nodes.size()-1),{});
        return tree;
    }

    // Average code length (bits per symbol)
    [[nodiscard]] f64 average_length(const std::vector<f64>& probs) const {
        f64 L=0;
        for(u32 i=0;i<u32(codes.size())&&i<u32(probs.size());++i) L+=probs[i]*codes[i].size();
        return L;
    }

    // Redundancy vs Shannon entropy
    [[nodiscard]] f64 redundancy(const std::vector<f64>& probs) const {
        f64 H=0; for(auto p:probs) if(p>0) H-=p*std::log2(p);
        return average_length(probs)-H;
    }
};

// ─── Arithmetic coding (range coder math) ────────────────────────────────────
struct ArithmeticCoder {
    u64 low{0}, high{~u64(0)}, range{0};
    static constexpr u64 HALF=u64(1)<<63, QUARTER=u64(1)<<62;

    // Encode a symbol with given cumulative frequencies
    // cum[i] = cumulative frequency up to symbol i, total = cum.back()
    void encode_symbol(u32 sym, const std::vector<u32>& cum) {
        u64 total=cum.back();
        range=(high-low)/total;
        u64 lo=cum[sym],hi=sym+1<u32(cum.size())?cum[sym+1]:total;
        high=low+range*hi-1;
        low =low+range*lo;
    }

    // Decode: find symbol given cumulative frequencies
    [[nodiscard]] u32 decode_symbol(u64 value, const std::vector<u32>& cum) const {
        u64 total=cum.back();
        u64 scaled=(value-low)/(range/total+1);
        for(u32 s=0;s+1<u32(cum.size());++s) if(scaled<cum[s+1]) return s;
        return u32(cum.size())-2;
    }
};

// ─── ANS (Asymmetric Numeral Systems) math ────────────────────────────────────
// tANS encoding state transition
struct ANSState {
    u32 x{0}; // current state

    // Encode symbol s with frequency fs out of total M (power of 2)
    // Returns new state and output bits
    [[nodiscard]] std::pair<ANSState,u32> encode(u32 s, u32 fs, u32 M_log2) const {
        u32 M=1u<<M_log2;
        u32 x_max=((M/fs)<<M_log2)*fs;
        u32 out_bits=0;
        u32 new_x=x;
        while(new_x>=x_max){ out_bits=(out_bits<<1)|(new_x&1); new_x>>=1; }
        new_x=(new_x/fs)*M+(new_x%fs); // simplified symbol encoding
        return {ANSState{new_x},out_bits};
    }

    // Decode: returns symbol and new state
    [[nodiscard]] std::pair<u32,ANSState> decode(const std::vector<u32>& cum, u32 M_log2) const {
        u32 M=1u<<M_log2;
        u32 slot=x%M;
        // Find symbol from CDF
        u32 sym=0;
        for(;sym+1<u32(cum.size());++sym) if(slot<cum[sym+1]) break;
        u32 new_x=cum[sym]+(x/M); // simplified
        return {sym,ANSState{new_x}};
    }
};

// ─── 2D DCT (Discrete Cosine Transform) ──────────────────────────────────────
// 8×8 DCT as used in JPEG
[[nodiscard]] inline std::array<f32,64> dct_8x8(const std::array<f32,64>& block) {
    std::array<f32,64> out{};
    static constexpr f32 C0=1.f/std::numbers::sqrt2_v<f32>, C1=0.5f;
    for(u32 v=0;v<8;++v) for(u32 u=0;u<8;++u) {
        f32 sum=0;
        for(u32 y=0;y<8;++y) for(u32 x=0;x<8;++x)
            sum+=block[y*8+x]*std::cos((2*x+1)*u*f32(std::numbers::pi)/16)*std::cos((2*y+1)*v*f32(std::numbers::pi)/16);
        f32 cu=u==0?C0:1.f, cv=v==0?C0:1.f;
        out[v*8+u]=0.25f*cu*cv*sum;
    }
    return out;
}

[[nodiscard]] inline std::array<f32,64> idct_8x8(const std::array<f32,64>& dct) {
    std::array<f32,64> out{};
    static constexpr f32 C0=1.f/std::numbers::sqrt2_v<f32>;
    for(u32 y=0;y<8;++y) for(u32 x=0;x<8;++x) {
        f32 sum=0;
        for(u32 v=0;v<8;++v) for(u32 u=0;u<8;++u) {
            f32 cu=u==0?C0:1.f, cv=v==0?C0:1.f;
            sum+=cu*cv*dct[v*8+u]*std::cos((2*x+1)*u*f32(std::numbers::pi)/16)*std::cos((2*y+1)*v*f32(std::numbers::pi)/16);
        }
        out[y*8+x]=0.25f*sum;
    }
    return out;
}

// JPEG quantization table (luminance, quality 50)
static constexpr u32 JPEG_QTABLE_LUMA[64] = {
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55, 14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62, 18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99
};

// JPEG-like quantization + zigzag
[[nodiscard]] inline std::array<i32,64> jpeg_quantize(const std::array<f32,64>& dct, u32 quality=50) {
    f32 scale=quality<50?5000.f/quality:(200.f-2*quality);
    std::array<i32,64> out{};
    for(u32 i=0;i<64;++i) {
        f32 q=std::max(1.f,JPEG_QTABLE_LUMA[i]*scale/100.f);
        out[i]=i32(std::round(dct[i]/q));
    }
    return out;
}

// Zigzag scan order for 8×8 block
[[nodiscard]] inline std::array<u32,64> zigzag_order() {
    static const u32 ZZ[64]={
         0, 1, 5, 6,14,15,27,28,
         2, 4, 7,13,16,26,29,42,
         3, 8,12,17,25,30,41,43,
         9,11,18,24,31,40,44,53,
        10,19,23,32,39,45,52,54,
        20,22,33,38,46,51,55,60,
        21,34,37,47,50,56,59,61,
        35,36,48,49,57,58,62,63
    };
    std::array<u32,64> zz{}; for(u32 i=0;i<64;++i) zz[i]=ZZ[i]; return zz;
}

// Run-length encoding (for quantized DCT coefficients)
struct RLESymbol { u32 zeros; i32 value; };
[[nodiscard]] inline std::vector<RLESymbol> rle_encode(const std::array<i32,64>& quantized) {
    auto zz=zigzag_order();
    std::vector<RLESymbol> out;
    u32 zeros=0;
    for(u32 i=1;i<64;++i) { // skip DC
        i32 v=quantized[zz[i]];
        if(v==0) { ++zeros; if(zeros==16){ out.push_back({15,0}); zeros=0; } }
        else { out.push_back({zeros,v}); zeros=0; }
    }
    out.push_back({0,0}); // EOB
    return out;
}

// ─── LZ77 sliding window (compression statistics) ────────────────────────────
struct LZ77Match { u32 offset{0},length{0}; u8 literal{0}; };

[[nodiscard]] inline std::vector<LZ77Match> lz77_parse(const std::vector<u8>& data,
    u32 win=32768, u32 max_len=258)
{
    std::vector<LZ77Match> out;
    u32 n=u32(data.size()), pos=0;
    while(pos<n) {
        u32 best_len=0,best_off=0;
        u32 start=pos>win?pos-win:0;
        for(u32 i=start;i<pos;++i) {
            u32 len=0;
            while(len<max_len&&pos+len<n&&data[i+len]==data[pos+len]) ++len;
            if(len>best_len){ best_len=len; best_off=pos-i; }
        }
        if(best_len>=3) { out.push_back({best_off,best_len,0}); pos+=best_len; }
        else { out.push_back({0,0,data[pos++]}); }
    }
    return out;
}

// Compression ratio estimate
[[nodiscard]] inline f64 lz77_compression_ratio(const std::vector<LZ77Match>& parsed, u32 orig_size) {
    u32 encoded_bits=0;
    for(const auto& m:parsed) {
        if(m.length>=3) encoded_bits+=3+8+15; // flag + length + offset (approx)
        else encoded_bits+=9; // flag + literal
    }
    return f64(orig_size*8)/encoded_bits;
}

// ─── Wavelet coefficient statistics ──────────────────────────────────────────
// EZW (Embedded Zerotree Wavelet) significance map
enum class EZWSymbol : u8 { PosSignificant, NegSignificant, Zerotree, IsolatedZero };

[[nodiscard]] inline std::vector<EZWSymbol> ezw_pass(
    const std::vector<f32>& coeffs, f32 threshold, u32 width)
{
    u32 n=u32(coeffs.size());
    std::vector<EZWSymbol> map(n,EZWSymbol::IsolatedZero);
    for(u32 i=0;i<n;++i) {
        f32 v=std::abs(coeffs[i]);
        if(v>=threshold) { map[i]=coeffs[i]>0?EZWSymbol::PosSignificant:EZWSymbol::NegSignificant; }
        else {
            // Check descendants (simplified: children at 2i+1, 2i+2 for 1D)
            bool has_sig=false;
            for(u32 j=2*i+1;j<n&&j<4*i+5&&!has_sig;++j) if(std::abs(coeffs[j])>=threshold) has_sig=true;
            map[i]=has_sig?EZWSymbol::IsolatedZero:EZWSymbol::Zerotree;
        }
    }
    return map;
}

// SPIHT (Set Partitioning in Hierarchical Trees) rate-distortion estimate
[[nodiscard]] inline f64 spiht_rate_distortion(const std::vector<f32>& coeffs, u32 bits_budget) {
    // Sort coefficients by magnitude
    std::vector<f32> sorted=coeffs;
    std::sort(sorted.rbegin(),sorted.rend(),[](f32 a,f32 b){return std::abs(a)>std::abs(b);});
    f64 distortion=0;
    u32 bits_used=0;
    for(u32 i=0;i<u32(sorted.size());++i) {
        u32 bits_needed=u32(std::log2(std::abs(sorted[i])+1)+0.5)+2;
        if(bits_used+bits_needed>bits_budget){
            // Remaining coefficients contribute to distortion
            for(u32 j=i;j<u32(sorted.size());++j) distortion+=sorted[j]*sorted[j];
            break;
        }
        bits_used+=bits_needed;
    }
    return distortion;
}

} // namespace qs
