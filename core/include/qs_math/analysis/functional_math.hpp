#pragma once
// qs_math/functional_math.hpp — Compose, curry, memoize, fixed-point, monad-like (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <span>
#include <numeric>

namespace qs {

// ─── Function composition ─────────────────────────────────────────────────────

template<typename F, typename G>
[[nodiscard]] auto compose(F&& f, G&& g) {
    return [f=std::forward<F>(f), g=std::forward<G>(g)](auto&&... args) {
        return f(g(std::forward<decltype(args)>(args)...));
    };
}

template<typename F, typename... Gs>
[[nodiscard]] auto compose_n(F&& f, Gs&&... gs) {
    if constexpr(sizeof...(Gs)==0) return f;
    else return compose(std::forward<F>(f), compose_n(std::forward<Gs>(gs)...));
}

// Pipe: pipe(x, f, g, h) = h(g(f(x)))
template<typename T>
[[nodiscard]] T pipe(T x) { return x; }

template<typename T, typename F, typename... Fs>
[[nodiscard]] auto pipe(T&& x, F&& f, Fs&&... fs) {
    return pipe(f(std::forward<T>(x)), std::forward<Fs>(fs)...);
}

// ─── Memoization ──────────────────────────────────────────────────────────────

template<typename Key, typename Val>
struct Memo {
    std::unordered_map<Key,Val> cache;
    std::function<Val(Key)> fn;

    explicit Memo(std::function<Val(Key)> f) : fn(std::move(f)) {}

    [[nodiscard]] Val operator()(const Key& k) {
        auto it=cache.find(k);
        if(it!=cache.end()) return it->second;
        return cache[k]=fn(k);
    }

    void clear() { cache.clear(); }
    [[nodiscard]] u32 cache_size() const { return u32(cache.size()); }
};

template<typename Key, typename Val>
[[nodiscard]] auto memoize(std::function<Val(Key)> f) {
    return Memo<Key,Val>{std::move(f)};
}

// ─── Fixed-point combinator (Y combinator) ────────────────────────────────────

// Usage: auto fib = fix_point([](auto self, u32 n) { return n<=1?n:self(n-1)+self(n-2); });
template<typename F>
struct FixPoint {
    F f;
    template<typename... Args>
    auto operator()(Args&&... args) const {
        return f(*this, std::forward<Args>(args)...);
    }
};

template<typename F>
[[nodiscard]] auto fix_point(F&& f) { return FixPoint<std::decay_t<F>>{std::forward<F>(f)}; }

// ─── Partial application ──────────────────────────────────────────────────────

template<typename F, typename... Bound>
[[nodiscard]] auto partial(F&& f, Bound&&... bound) {
    return [f=std::forward<F>(f), ...bound=std::forward<Bound>(bound)](auto&&... rest) {
        return f(bound..., std::forward<decltype(rest)>(rest)...);
    };
}

// ─── Iterator-style functional ops on spans ───────────────────────────────────

template<typename T, typename F>
[[nodiscard]] std::vector<std::invoke_result_t<F,T>> fmap(std::span<const T> xs, F&& f) {
    std::vector<std::invoke_result_t<F,T>> out;
    out.reserve(xs.size());
    for(const auto& x:xs) out.push_back(f(x));
    return out;
}

template<typename T, typename F>
[[nodiscard]] std::vector<T> filter(std::span<const T> xs, F&& pred) {
    std::vector<T> out;
    for(const auto& x:xs) if(pred(x)) out.push_back(x);
    return out;
}

template<typename T, typename Acc, typename F>
[[nodiscard]] Acc fold_left(std::span<const T> xs, Acc init, F&& f) {
    for(const auto& x:xs) init=f(std::move(init),x);
    return init;
}

template<typename T, typename Acc, typename F>
[[nodiscard]] Acc fold_right(std::span<const T> xs, Acc init, F&& f) {
    for(i32 i=i32(xs.size())-1;i>=0;--i) init=f(xs[i],std::move(init));
    return init;
}

// Scan (running fold)
template<typename T, typename Acc, typename F>
[[nodiscard]] std::vector<Acc> scan_left(std::span<const T> xs, Acc init, F&& f) {
    std::vector<Acc> out; out.push_back(init);
    for(const auto& x:xs){init=f(std::move(init),x);out.push_back(init);}
    return out;
}

// Zip two spans
template<typename A, typename B>
[[nodiscard]] std::vector<std::pair<A,B>> zip(std::span<const A> as, std::span<const B> bs) {
    std::vector<std::pair<A,B>> out;
    u32 n=u32(std::min(as.size(),bs.size()));
    for(u32 i=0;i<n;++i) out.push_back({as[i],bs[i]});
    return out;
}

// ─── Numeric integration via functional composition ───────────────────────────

[[nodiscard]] inline f64 integrate_simpson(
    const std::function<f64(f64)>& f, f64 a, f64 b, u32 n=100)
{
    if(n%2) ++n;
    f64 h=(b-a)/f64(n);
    f64 s=f(a)+f(b);
    for(u32 i=1;i<n;++i) s+=(i%2==0?2.:4.)*f(a+f64(i)*h);
    return s*h/3.;
}

[[nodiscard]] inline f64 integrate_gauss5(
    const std::function<f64(f64)>& f, f64 a, f64 b)
{
    // 5-point Gauss-Legendre nodes/weights on [-1,1]
    static constexpr f64 nodes[5]={-0.9061798459,-0.5384693101,0.,0.5384693101,0.9061798459};
    static constexpr f64 wts[5]={0.2369268851,0.4786286705,0.5688888889,0.4786286705,0.2369268851};
    f64 mid=(a+b)*0.5,half=(b-a)*0.5,s=0.;
    for(u32 i=0;i<5;++i) s+=wts[i]*f(mid+half*nodes[i]);
    return half*s;
}

// Adaptive integration (Simpson with error control)
[[nodiscard]] inline f64 adaptive_integrate(
    const std::function<f64(f64)>& f, f64 a, f64 b,
    f64 tol=1e-8, u32 max_depth=20)
{
    std::function<f64(f64,f64,f64,f64,u32)> rec=[&](f64 a2,f64 b2,f64 fa,f64 fb,u32 depth)->f64{
        f64 m=(a2+b2)*0.5, fm=f(m);
        f64 whole=(b2-a2)/6.*(fa+4.*fm+fb);
        f64 left =(m-a2)/6.*(fa+4.*f((a2+m)*0.5)+fm);
        f64 right=(b2-m)/6.*(fm+4.*f((m+b2)*0.5)+fb);
        f64 delta=left+right-whole;
        if(depth>=max_depth||std::abs(delta)<15.*tol) return left+right+delta/15.;
        return rec(a2,m,fa,fm,depth+1)+rec(m,b2,fm,fb,depth+1);
    };
    return rec(a,b,f(a),f(b),0);
}

// ─── Maybe / Result monad ─────────────────────────────────────────────────────

template<typename T>
struct Maybe {
    std::optional<T> val;

    [[nodiscard]] static Maybe just(T v) { return {std::optional<T>{std::move(v)}}; }
    [[nodiscard]] static Maybe nothing() { return {std::nullopt}; }
    [[nodiscard]] bool has_value() const { return val.has_value(); }
    [[nodiscard]] const T& get() const { return *val; }

    template<typename F>
    [[nodiscard]] auto bind(F&& f) const -> decltype(f(val.value())) {
        if(!val) return decltype(f(*val))::nothing();
        return f(*val);
    }

    template<typename F>
    [[nodiscard]] auto fmap(F&& f) const -> Maybe<decltype(f(*val))> {
        if(!val) return Maybe<decltype(f(*val))>::nothing();
        return Maybe<decltype(f(*val))>::just(f(*val));
    }
};

// ─── Function differentiation (finite difference) ─────────────────────────────

[[nodiscard]] inline f64 finite_diff_forward(const std::function<f64(f64)>& f, f64 x, f64 h=1e-7) {
    return (f(x+h)-f(x))/h;
}
[[nodiscard]] inline f64 finite_diff_central(const std::function<f64(f64)>& f, f64 x, f64 h=1e-5) {
    return (f(x+h)-f(x-h))/(2.*h);
}
[[nodiscard]] inline f64 finite_diff_second(const std::function<f64(f64)>& f, f64 x, f64 h=1e-4) {
    return (f(x+h)-2.*f(x)+f(x-h))/(h*h);
}

// Newton's method
[[nodiscard]] inline std::optional<f64> newton_solve(
    const std::function<f64(f64)>& f, f64 x0,
    f64 tol=1e-12, u32 max_iter=100)
{
    f64 x=x0;
    for(u32 i=0;i<max_iter;++i){
        f64 fx=f(x), dfx=finite_diff_central(f,x);
        if(std::abs(dfx)<1e-15) return std::nullopt;
        x-=fx/dfx;
        if(std::abs(fx)<tol) return x;
    }
    return std::nullopt;
}

// Gradient descent (multivariate)
[[nodiscard]] inline std::vector<f64> gradient_descent(
    const std::function<f64(std::span<const f64>)>& f,
    std::vector<f64> x0, f64 lr=0.01, u32 max_iter=1000, f64 tol=1e-8)
{
    u32 n=u32(x0.size());
    for(u32 iter=0;iter<max_iter;++iter){
        std::vector<f64> grad(n);
        f64 fx=f(x0);
        for(u32 i=0;i<n;++i){
            std::vector<f64> xh=x0; xh[i]+=1e-5;
            grad[i]=(f(xh)-fx)/1e-5;
        }
        f64 gnorm=0.; for(auto g:grad) gnorm+=g*g;
        if(std::sqrt(gnorm)<tol) break;
        for(u32 i=0;i<n;++i) x0[i]-=lr*grad[i];
    }
    return x0;
}

} // namespace qs
