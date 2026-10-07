#pragma once
// qs_math/hash_math.hpp — Spatial hashing, consistent hashing, locality sensitive (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <unordered_map>
#include <array>
#include <cmath>
#include <functional>
#include <algorithm>

namespace qs {

// ─── Integer hashing ─────────────────────────────────────────────────────────

[[nodiscard]] inline u32 hash_u32(u32 x) {
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = (x >> 16) ^ x;
    return x;
}

[[nodiscard]] inline u64 hash_u64(u64 x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

[[nodiscard]] inline u32 hash_combine(u32 a, u32 b) {
    return a ^ (b + 0x9e3779b9u + (a<<6) + (a>>2));
}

// ─── Spatial hash map ────────────────────────────────────────────────────────

template<typename T>
struct SpatialHashGrid {
    f32 cell_size{1.f};
    std::unordered_map<u64, std::vector<T>> cells;

    [[nodiscard]] u64 key(i32 x, i32 y, i32 z) const {
        u64 ux=u64(x+100000), uy=u64(y+100000), uz=u64(z+100000);
        return hash_u64(ux*73856093ULL^uy*19349663ULL^uz*83492791ULL);
    }

    [[nodiscard]] std::array<i32,3> cell_of(const Vec3f& p) const {
        return {i32(std::floor(p.x/cell_size)),
                i32(std::floor(p.y/cell_size)),
                i32(std::floor(p.z/cell_size))};
    }

    void insert(const Vec3f& p, const T& val) {
        auto [cx,cy,cz] = cell_of(p);
        cells[key(cx,cy,cz)].push_back(val);
    }

    void query_radius(const Vec3f& p, f32 r, std::vector<T>& out) const {
        i32 steps = i32(std::ceil(r/cell_size));
        auto [cx,cy,cz] = cell_of(p);
        for (i32 dz=-steps;dz<=steps;++dz)
        for (i32 dy=-steps;dy<=steps;++dy)
        for (i32 dx=-steps;dx<=steps;++dx) {
            auto it = cells.find(key(cx+dx,cy+dy,cz+dz));
            if (it!=cells.end())
                for (const auto& v:it->second) out.push_back(v);
        }
    }

    void clear() { cells.clear(); }
};

// ─── Consistent hashing (ring) ───────────────────────────────────────────────

struct ConsistentHashRing {
    struct Node { u32 hash; u32 id; };
    std::vector<Node> ring;
    u32 replicas{100};

    void add_node(u32 id) {
        for (u32 r=0;r<replicas;++r) {
            u32 h = hash_combine(id, hash_u32(r));
            ring.push_back({h,id});
        }
        std::sort(ring.begin(),ring.end(),[](auto&a,auto&b){return a.hash<b.hash;});
    }

    void remove_node(u32 id) {
        ring.erase(std::remove_if(ring.begin(),ring.end(),
            [id](const Node& n){return n.id==id;}),ring.end());
    }

    [[nodiscard]] u32 get_node(u32 key) const {
        if (ring.empty()) return 0;
        auto it = std::lower_bound(ring.begin(),ring.end(),Node{key,0},
            [](const Node&a,const Node&b){return a.hash<b.hash;});
        if (it==ring.end()) it=ring.begin();
        return it->id;
    }
};

// ─── LSH (Locality-Sensitive Hashing) ────────────────────────────────────────

struct LSHTable {
    u32 num_hashes{8};
    u32 num_tables{4};
    u32 dim{32};
    f32 width{4.f};

    std::vector<std::vector<f32>> random_vectors; // [num_tables*num_hashes][dim]
    std::vector<f32> random_offsets;

    void init(u32 d, u64 seed=42) {
        dim=d;
        random_vectors.resize(num_tables*num_hashes, std::vector<f32>(d));
        random_offsets.resize(num_tables*num_hashes);
        // Simple LCG for deterministic init
        u64 s=seed;
        auto rng=[&]{ s=s*6364136223846793005ull+1442695040888963407ull; return f32(s>>33)/f32(1u<<31)-1.f; };
        for (auto& rv : random_vectors) for (auto& v : rv) v=rng();
        for (auto& o : random_offsets) o=(rng()+1.f)*0.5f*width;
    }

    [[nodiscard]] u64 hash_vector(const std::vector<f32>& v, u32 table) const {
        u64 h=0;
        for (u32 i=0;i<num_hashes;++i) {
            const auto& rv=random_vectors[table*num_hashes+i];
            f32 proj=0.f;
            for (u32 j=0;j<std::min(u32(v.size()),dim);++j) proj+=v[j]*rv[j];
            u32 bucket=u32(std::floor((proj+random_offsets[table*num_hashes+i])/width));
            h=hash_combine(u32(h), bucket);
        }
        return h;
    }

    [[nodiscard]] bool likely_similar(
        const std::vector<f32>& a, const std::vector<f32>& b) const
    {
        for (u32 t=0;t<num_tables;++t)
            if (hash_vector(a,t)==hash_vector(b,t)) return true;
        return false;
    }
};

// ─── Min-hash (Jaccard similarity) ───────────────────────────────────────────

struct MinHash {
    u32 num_hashes{128};
    std::vector<u32> a, b; // random coefficients for universal hashing

    void init(u32 n, u64 seed=0) {
        num_hashes=n; a.resize(n); b.resize(n);
        u64 s=seed^0xdeadbeefcafe;
        auto rng=[&]{s=s*6364136223846793005ull+1;return u32(s>>32);};
        for (u32 i=0;i<n;++i){a[i]=rng()|1;b[i]=rng();}
    }

    [[nodiscard]] std::vector<u32> compute(const std::vector<u32>& set) const {
        std::vector<u32> sig(num_hashes, u32(-1));
        for (u32 x : set)
            for (u32 i=0;i<num_hashes;++i) {
                u32 h = a[i]*x+b[i];
                sig[i]=std::min(sig[i],h);
            }
        return sig;
    }

    [[nodiscard]] f32 jaccard_estimate(
        const std::vector<u32>& sig_a, const std::vector<u32>& sig_b) const
    {
        u32 eq=0;
        for (u32 i=0;i<num_hashes&&i<u32(sig_a.size())&&i<u32(sig_b.size());++i)
            if (sig_a[i]==sig_b[i]) ++eq;
        return f32(eq)/f32(num_hashes);
    }
};

// ─── Polynomial rolling hash ──────────────────────────────────────────────────

struct RollingHash {
    static constexpr u64 BASE=131, MOD=1000000007ULL;
    u64 hash{0}, power{1};
    u32 window{0};
    std::vector<u8> buf;

    void push(u8 c) {
        hash=(hash*BASE+c)%MOD;
        buf.push_back(c);
        power=(power*BASE)%MOD;
        ++window;
    }

    void pop() {
        if (buf.empty()) return;
        hash=(hash - buf.front()*power%MOD + MOD*2)%MOD;
        power=(power*876124*BASE)%MOD; // modular inverse approx
        buf.erase(buf.begin());
        --window;
    }

    [[nodiscard]] u64 get() const { return hash; }
};

// ─── Perlin hash permutation ──────────────────────────────────────────────────

static constexpr u8 PERM256[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,
    120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,
    33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,
    230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,
    1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,
    116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,
    124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,
    47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,
    44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,
    19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,
    249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
    176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,
    67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

[[nodiscard]] inline u8 perm(u32 i) { return PERM256[i & 255]; }
[[nodiscard]] inline u8 perm2(u32 x, u32 y) { return perm(x+perm(y)); }
[[nodiscard]] inline u8 perm3(u32 x, u32 y, u32 z) { return perm(x+perm2(y,z)); }

} // namespace qs
