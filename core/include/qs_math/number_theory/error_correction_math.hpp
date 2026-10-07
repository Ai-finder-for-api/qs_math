#pragma once
// qs_math/error_correction_math.hpp — Hamming, Reed-Solomon decoder, LDPC helpers (C++20)
#include "scalar_types.hpp"
#include "finite_field_math.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <span>
#include <optional>
#include <bitset>

namespace qs {

// ─── Hamming code ─────────────────────────────────────────────────────────────

// Encode 4-bit data to 7-bit Hamming(7,4)
[[nodiscard]] inline u8 hamming74_encode(u8 data4) {
    u8 d1=(data4>>0)&1, d2=(data4>>1)&1, d3=(data4>>2)&1, d4=(data4>>3)&1;
    u8 p1=d1^d2^d4, p2=d1^d3^d4, p3=d2^d3^d4;
    return u8(p1|(p2<<1)|(d1<<2)|(p3<<3)|(d2<<4)|(d3<<5)|(d4<<6));
}

// Decode and correct 7-bit Hamming(7,4), returns corrected data nibble
[[nodiscard]] inline std::pair<u8,bool> hamming74_decode(u8 codeword) {
    u8 b1=(codeword>>0)&1, b2=(codeword>>1)&1, b3=(codeword>>2)&1;
    u8 b4=(codeword>>3)&1, b5=(codeword>>4)&1, b6=(codeword>>5)&1, b7=(codeword>>6)&1;
    u8 s1=b1^b3^b5^b7, s2=b2^b3^b6^b7, s3=b4^b5^b6^b7;
    u8 syndrome=u8(s1|(s2<<1)|(s3<<2));
    bool corrected=false;
    if(syndrome) { codeword^=(1<<(syndrome-1)); corrected=true; }
    return {u8(((codeword>>2)&1)|((codeword>>3)&2)|((codeword>>4)&4)|((codeword>>5)&8)), corrected};
}

// Extended Hamming (8,4) — adds overall parity
[[nodiscard]] inline u8 hamming84_encode(u8 data4) {
    u8 h7=hamming74_encode(data4);
    u8 parity=0;
    for(u32 i=0;i<7;++i) parity^=(h7>>i)&1;
    return u8(h7|(parity<<7));
}

// ─── Syndrome decoder ─────────────────────────────────────────────────────────

// Generic linear code syndrome computation: s = H * r (over GF(2))
[[nodiscard]] inline std::vector<u8> compute_syndrome(
    std::span<const u8> received,
    std::span<const std::vector<u8>> H_rows) // parity-check matrix rows
{
    u32 n=u32(received.size()), m=u32(H_rows.size());
    std::vector<u8> s(m,0);
    for(u32 i=0;i<m;++i)
        for(u32 j=0;j<n&&j<H_rows[i].size();++j)
            s[i]^=H_rows[i][j]*received[j];
    return s;
}

// ─── Reed-Solomon encoder ─────────────────────────────────────────────────────

struct RSEncoder {
    RSField field;
    u32 t{4};   // error correction capability (corrects t errors)
    std::vector<u64> generator; // generator polynomial

    void init(u32 errors=4) {
        t=errors; field.init();
        // g(x) = prod_{i=1}^{2t} (x - alpha^i)
        generator={1};
        for(u32 i=1;i<=2*t;++i){
            std::vector<u64> factor={field.gf.inv(field.alpha_powers[i]),1};
            // Actually: (x + alpha^i) in GF(256)
            factor[0]=field.alpha_powers[i];
            std::vector<u64> new_gen(generator.size()+1,0);
            for(u32 j=0;j<generator.size();++j){
                new_gen[j]=field.gf.add(new_gen[j],field.gf.mul(generator[j],factor[0]));
                new_gen[j+1]=field.gf.add(new_gen[j+1],generator[j]);
            }
            generator=new_gen;
        }
    }

    // Systematic encoding: append 2t parity symbols to k data symbols
    [[nodiscard]] std::vector<u64> encode(std::span<const u64> data) const {
        u32 k=u32(data.size());
        u32 n2=k+2*t;
        std::vector<u64> msg(n2,0);
        for(u32 i=0;i<k;++i) msg[i+2*t]=data[i];
        // Polynomial division: remainder when msg*x^2t divided by generator
        for(u32 i=n2-1;i>=2*t;--i){
            u64 coeff=msg[i];
            if(coeff==0) continue;
            for(u32 j=0;j<generator.size()&&i-j<n2;++j)
                msg[i-j]=field.gf.add(msg[i-j],field.gf.mul(generator[generator.size()-1-j],coeff));
            msg[i]=0;
        }
        // Reassemble systematic codeword
        std::vector<u64> cw(n2);
        for(u32 i=0;i<2*t;++i) cw[i]=msg[i];
        for(u32 i=0;i<k;++i) cw[2*t+i]=data[i];
        return cw;
    }
};

// ─── BerlekampMassey LFSR synthesis ──────────────────────────────────────────

[[nodiscard]] inline std::vector<u64> berlekamp_massey(
    std::span<const u64> syndromes, const GF2n& gf)
{
    u32 N=u32(syndromes.size());
    std::vector<u64> C={1}, B={1};
    u32 L=0, m=1; u64 b=1;
    for(u32 n2=0;n2<N;++n2){
        u64 d=syndromes[n2];
        for(u32 i=1;i<=L;++i) d=gf.add(d,gf.mul(C[i],syndromes[n2-i]));
        if(d==0){++m;continue;}
        std::vector<u64> T=C;
        u64 coeff=gf.mul(d,gf.inv(b));
        if(C.size()<B.size()+m) C.resize(B.size()+m,0);
        for(u32 i=0;i<B.size();++i) C[i+m]=gf.add(C[i+m],gf.mul(coeff,B[i]));
        if(2*L<=n2){L=n2+1-L;B=T;b=d;m=1;}else ++m;
    }
    return C;
}

// ─── Convolutional code (rate 1/2, K=3) ──────────────────────────────────────

struct ConvEncoder {
    u32 state{0};
    // G1=0b111, G2=0b101 (rate 1/2, K=3)
    static constexpr u32 G1=7, G2=5;

    [[nodiscard]] std::pair<u8,u8> encode_bit(u8 bit) {
        state=((state>>1)|(u32(bit)<<2))&7;
        return {u8(__builtin_popcount(state&G1)&1), u8(__builtin_popcount(state&G2)&1)};
    }

    [[nodiscard]] std::vector<u8> encode(std::span<const u8> bits) {
        std::vector<u8> out;
        for(u8 b:bits){auto[o1,o2]=encode_bit(b);out.push_back(o1);out.push_back(o2);}
        return out;
    }
};

// Viterbi decoder (rate 1/2, K=3)
[[nodiscard]] inline std::vector<u8> viterbi_decode(std::span<const u8> received) {
    static constexpr u32 N_STATES=4;
    static constexpr u32 G1=7, G2=5;
    u32 n_bits=u32(received.size())/2;
    std::vector<std::array<i32,N_STATES>> metrics(n_bits+1);
    std::vector<std::array<u32,N_STATES>> path(n_bits+1);
    metrics[0].fill(-1); metrics[0][0]=0;

    for(u32 t=0;t<n_bits;++t) {
        metrics[t+1].fill(-1);
        for(u32 s=0;s<N_STATES;++s) {
            if(metrics[t][s]<0) continue;
            for(u32 in=0;in<2;++in) {
                u32 ns=((s>>1)|(in<<(3-1)))&(N_STATES-1);
                u32 full=(s|(in<<2))&7;
                u8 o1=__builtin_popcount(full&G1)&1, o2=__builtin_popcount(full&G2)&1;
                i32 bm=i32(o1==(received[2*t]&1)?0:1)+i32(o2==(received[2*t+1]&1)?0:1);
                i32 m=metrics[t][s]+(2-bm);
                if(m>metrics[t+1][ns]){metrics[t+1][ns]=m;path[t+1][ns]=s;}
            }
        }
    }
    // Traceback
    std::vector<u8> decoded(n_bits);
    u32 s=0; i32 best=-1;
    for(u32 i=0;i<N_STATES;++i) if(metrics[n_bits][i]>best){best=metrics[n_bits][i];s=i;}
    for(i32 t=i32(n_bits)-1;t>=0;--t){
        u32 ps=path[t+1][s];
        decoded[t]=u8((s>>(3-1))&1); // input bit that caused transition
        s=ps;
    }
    return decoded;
}

// ─── LDPC factor graph ────────────────────────────────────────────────────────

struct LDPCParams {
    u32 n{},k{},m{};  // codeword length, info bits, check nodes
    std::vector<std::vector<u32>> H_row_to_col; // sparse parity-check rows
    std::vector<std::vector<u32>> H_col_to_row; // transpose

    [[nodiscard]] f32 belief_propagation_iteration(
        std::vector<f32>& llr, u32 max_iter=50) const
    {
        // Sum-product on factor graph
        std::vector<f32> check_to_var(n*m, 0.f);
        std::vector<f32> var_to_check(n*m, 0.f);
        for(u32 iter=0;iter<max_iter;++iter){
            // Check node update (min-sum approximation)
            for(u32 i=0;i<m;++i){
                f32 sign=1.f, min1=1e10f, min2=1e10f;
                for(u32 j:H_row_to_col[i]){
                    f32 m2=llr[j]+check_to_var[i*n+j];
                    sign*=m2<0?-1.f:1.f;
                    f32 am=std::abs(m2);
                    if(am<min1){min2=min1;min1=am;}else if(am<min2)min2=am;
                }
                for(u32 j:H_row_to_col[i])
                    check_to_var[i*n+j]=sign*(std::abs(llr[j]+check_to_var[i*n+j])==min1?min2:min1)*
                        (llr[j]+check_to_var[i*n+j]<0?-1.f:1.f)*sign;
            }
            // Variable node update
            for(u32 j=0;j<n;++j){
                f32 total=llr[j];
                for(u32 i:H_col_to_row[j]) total+=check_to_var[i*n+j];
                for(u32 i:H_col_to_row[j]) var_to_check[i*n+j]=total-check_to_var[i*n+j];
            }
        }
        return 0.f;
    }
};

} // namespace qs
