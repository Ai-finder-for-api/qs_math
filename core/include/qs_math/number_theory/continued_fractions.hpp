#pragma once
// qs_math/continued_fractions.hpp — CF convergents, Stern-Brocot, best rational approx (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <cassert>
#include <optional>

namespace qs {

// ─── Continued fraction expansion ────────────────────────────────────────────

// Simple CF expansion of x: x = a0 + 1/(a1 + 1/(a2 + ...))
[[nodiscard]] inline std::vector<i64> cf_expand(f64 x, u32 max_terms=20, f64 eps=1e-14) {
    std::vector<i64> a;
    for (u32 i=0;i<max_terms;++i) {
        i64 ai = i64(std::floor(x));
        a.push_back(ai);
        f64 frac = x - f64(ai);
        if (frac < eps) break;
        x = 1./frac;
    }
    return a;
}

// CF expansion of rational p/q
[[nodiscard]] inline std::vector<i64> cf_rational(i64 p, i64 q) {
    std::vector<i64> a;
    while (q) {
        a.push_back(p/q);
        i64 r=p%q; p=q; q=r;
    }
    return a;
}

// ─── Convergents ─────────────────────────────────────────────────────────────

struct Convergent {
    i64 p{}, q{};  // numerator/denominator
    [[nodiscard]] f64 value() const { return f64(p)/f64(q); }
};

[[nodiscard]] inline std::vector<Convergent> cf_convergents(std::span<const i64> a) {
    std::vector<Convergent> cvg;
    i64 h_prev=1,h_curr=a[0], k_prev=0,k_curr=1;
    cvg.push_back({h_curr,k_curr});
    for (u32 i=1;i<u32(a.size());++i) {
        i64 h_next=a[i]*h_curr+h_prev;
        i64 k_next=a[i]*k_curr+k_prev;
        cvg.push_back({h_next,k_next});
        h_prev=h_curr; h_curr=h_next;
        k_prev=k_curr; k_curr=k_next;
    }
    return cvg;
}

// Best rational approximation to x with denominator <= max_denom
[[nodiscard]] inline Convergent best_rational_approx(f64 x, i64 max_denom) {
    auto a = cf_expand(x);
    auto cvg = cf_convergents(a);
    Convergent best{i64(std::round(x)),1};
    for (const auto& c : cvg) {
        if (c.q > max_denom) break;
        if (std::abs(c.value()-x) < std::abs(best.value()-x)) best=c;
    }
    return best;
}

// ─── Stern-Brocot tree ───────────────────────────────────────────────────────

struct SBNode {
    i64 p{}, q{};
    [[nodiscard]] f64 value() const { return f64(p)/f64(q); }
};

// Mediant of two fractions
[[nodiscard]] inline SBNode mediant(SBNode a, SBNode b) {
    return {a.p+b.p, a.q+b.q};
}

// Search Stern-Brocot tree for best approximation to x in [0,1]
[[nodiscard]] inline SBNode stern_brocot_search(f64 x, u32 max_steps=50) {
    SBNode lo{0,1}, hi{1,1};
    for (u32 i=0;i<max_steps;++i) {
        SBNode mid = mediant(lo,hi);
        f64 mv = mid.value();
        if (std::abs(mv-x)<1e-12) return mid;
        if (mv < x) lo=mid;
        else        hi=mid;
    }
    return mediant(lo,hi);
}

// Stern-Brocot rank (path in tree as L/R sequence)
[[nodiscard]] inline std::vector<bool> stern_brocot_path(f64 x, u32 depth=30) {
    std::vector<bool> path;
    SBNode lo{0,1},hi{1,1};
    for (u32 i=0;i<depth;++i) {
        SBNode mid=mediant(lo,hi);
        if (std::abs(mid.value()-x)<1e-13) break;
        if (mid.value()<x){path.push_back(true);lo=mid;}  // R
        else              {path.push_back(false);hi=mid;} // L
    }
    return path;
}

// ─── Farey sequence ───────────────────────────────────────────────────────────

// Generate Farey sequence F_n (all fractions p/q with 0<=p<=q<=n, gcd=1)
[[nodiscard]] inline std::vector<std::pair<i64,i64>> farey_sequence(i64 n) {
    std::vector<std::pair<i64,i64>> F;
    i64 a=0,b=1,c=1,d=n;
    F.push_back({a,b});
    while(c<=n){
        F.push_back({c,d});
        i64 k=(n+b)/d;
        i64 a2=c, b2=d;
        c=k*c-a; d=k*d-b;
        a=a2; b=b2;
    }
    return F;
}

// Farey neighbours of p/q in F_n
[[nodiscard]] inline std::pair<std::pair<i64,i64>,std::pair<i64,i64>>
farey_neighbours(i64 p, i64 q, i64 n)
{
    // Left: p'/q' with q*p' - p*q' = -1 and q+q' > n minimized
    i64 x=p, y=q;
    i64 best_lp=0,best_lq=1,best_rp=1,best_rq=1;
    // Simple Farey property: neighbours satisfy |p*q'-p'*q|=1
    for (i64 qr=1;qr<=n;++qr) {
        // Extended Euclidean
        // Actually use mediant property
        break; // placeholder
    }
    return {{best_lp,best_lq},{best_rp,best_rq}};
}

// ─── Periodic continued fractions ────────────────────────────────────────────

// Check if CF is eventually periodic (all sqrt of integers are)
// Detect period via Floyd's cycle detection on CF expansion
[[nodiscard]] inline std::optional<u32> cf_period(const std::vector<i64>& a) {
    if (a.size() < 2) return std::nullopt;
    for (u32 period=1;period<u32(a.size()/2);++period) {
        bool ok=true;
        for (u32 i=1;i+period<u32(a.size());++i)
            if (a[i]!=a[i+period]){ok=false;break;}
        if (ok) return period;
    }
    return std::nullopt;
}

// CF expansion of sqrt(n)
[[nodiscard]] inline std::pair<i64,std::vector<i64>> cf_sqrt(i64 n) {
    i64 a0=i64(std::floor(std::sqrt(f64(n))));
    if (a0*a0==n) return {a0,{}};
    std::vector<i64> period;
    i64 m=0, d=1, a=a0;
    do {
        m=d*a-m; d=(n-m*m)/d; a=(a0+m)/d;
        period.push_back(a);
    } while (a!=2*a0);
    return {a0,period};
}

// Pell equation x^2 - n*y^2 = 1, fundamental solution via CF
[[nodiscard]] inline std::optional<std::pair<i64,i64>> pell_fundamental(i64 n) {
    auto [a0,period]=cf_sqrt(n);
    if (period.empty()) return std::nullopt; // perfect square
    // Convergent at index (period length - 1)
    std::vector<i64> cf_terms={a0};
    u32 len=u32(period.size());
    for (u32 i=0;i<len;++i) cf_terms.push_back(period[i%len]);
    auto cvg=cf_convergents(cf_terms);
    // Fundamental solution is last convergent (period-1 or period-length)
    u32 idx = (len%2==1)?(len-1):(2*len-1);
    if (idx>=cvg.size()) idx=u32(cvg.size())-1;
    return std::make_pair(cvg[idx].p, cvg[idx].q);
}

// ─── Continued fraction arithmetic (Gosper) ──────────────────────────────────

// Evaluate continued fraction given terms on-the-fly (floor(x))
struct CFEvaluator {
    i64 a{0},b{1},c{1},d{0}; // state: (ax+b)/(cx+d)

    void ingest(i64 term) {
        // Shift: x <- 1/(x-term)  =>  (a,b,c,d) <- (c,d,a-term*c,b-term*d)
        i64 na=c,nb=d,nc=a-term*c,nd=b-term*d;
        a=na;b=nb;c=nc;d=nd;
    }

    [[nodiscard]] std::optional<i64> output() const {
        if (c==0&&d==0) return std::nullopt;
        if (c==0||d==0) return std::nullopt;
        i64 qa=a/c, qb=b/d;
        if (qa==qb) return qa;
        return std::nullopt;
    }

    void emit(i64 q) {
        i64 na=c,nb=d,nc=a-q*c,nd=b-q*d;
        a=na;b=nb;c=nc;d=nd;
    }
};

} // namespace qs
