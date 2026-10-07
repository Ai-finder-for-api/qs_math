#pragma once
// qs_math/reactive_math.hpp — Signal graphs, reactive variables, dependency tracking (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <algorithm>
#include <any>

namespace qs {

// ─── Reactive scalar ─────────────────────────────────────────────────────────

// Forward declarations
struct ReactiveNode;
using NodeRef = std::shared_ptr<ReactiveNode>;

struct ReactiveNode {
    f32  value{};
    bool dirty{true};
    std::string name{};
    std::vector<std::weak_ptr<ReactiveNode>> dependents;
    std::vector<std::shared_ptr<ReactiveNode>> dependencies;
    std::function<f32()> compute_fn{};

    void mark_dirty() {
        dirty=true;
        for(auto& wp:dependents) if(auto d=wp.lock()) d->mark_dirty();
    }

    f32 get() {
        if(dirty&&compute_fn){value=compute_fn();dirty=false;}
        return value;
    }

    void set(f32 v) { value=v; dirty=false; mark_dirty(); }
};

// ─── Reactive graph builder ───────────────────────────────────────────────────

struct ReactiveGraph {
    std::vector<NodeRef> nodes;

    [[nodiscard]] NodeRef var(f32 v, const std::string& name="") {
        auto n=std::make_shared<ReactiveNode>();
        n->value=v; n->dirty=false; n->name=name;
        nodes.push_back(n);
        return n;
    }

    [[nodiscard]] NodeRef derived(
        std::vector<NodeRef> deps, std::function<f32()> fn,
        const std::string& name="")
    {
        auto n=std::make_shared<ReactiveNode>();
        n->name=name; n->compute_fn=std::move(fn);
        n->dependencies=deps;
        for(auto& d:deps) d->dependents.push_back(n);
        nodes.push_back(n);
        return n;
    }

    // Convenience: map single node through function
    [[nodiscard]] NodeRef map(NodeRef src, const std::function<f32(f32)>& fn) {
        return derived({src},[src,fn]()->f32{return fn(src->get());});
    }

    // Combine two nodes
    [[nodiscard]] NodeRef combine(NodeRef a, NodeRef b, const std::function<f32(f32,f32)>& fn) {
        return derived({a,b},[a,b,fn]()->f32{return fn(a->get(),b->get());});
    }

    // Topological evaluation order
    [[nodiscard]] std::vector<NodeRef> topo_sort() const {
        std::vector<NodeRef> order;
        std::unordered_set<ReactiveNode*> visited;
        std::function<void(ReactiveNode*)> dfs=[&](ReactiveNode* n){
            if(visited.count(n)) return;
            visited.insert(n);
            for(auto& d:n->dependencies) dfs(d.get());
            order.push_back(std::shared_ptr<ReactiveNode>(n,[](ReactiveNode*){}));
        };
        for(auto& n:nodes) dfs(n.get());
        return order;
    }

    void invalidate_all() { for(auto& n:nodes) n->mark_dirty(); }
};

// ─── Signal stream ────────────────────────────────────────────────────────────

template<typename T>
struct Signal {
    T current{};
    std::vector<std::function<void(const T&)>> subscribers;

    void emit(const T& v) {
        current=v;
        for(auto& s:subscribers) s(v);
    }

    void subscribe(std::function<void(const T&)> fn) {
        subscribers.push_back(std::move(fn));
    }

    // Map: returns a new signal transformed by fn
    template<typename U>
    [[nodiscard]] Signal<U> map(std::function<U(const T&)> fn) {
        Signal<U> out;
        subscribe([&out,fn](const T& v){out.emit(fn(v));});
        return out;
    }

    // Filter: only emit when predicate holds
    [[nodiscard]] Signal<T> filter(std::function<bool(const T&)> pred) {
        Signal<T> out;
        subscribe([&out,pred](const T& v){if(pred(v))out.emit(v);});
        return out;
    }
};

// ─── Smooth reactive value (spring-driven) ────────────────────────────────────

struct SmoothReactive {
    f32 target{0.f};
    f32 current{0.f};
    f32 velocity{0.f};
    f32 stiffness{100.f};
    f32 damping{20.f};

    void set_target(f32 t) { target=t; }

    f32 update(f32 dt) {
        f32 F=stiffness*(target-current)-damping*velocity;
        velocity+=F*dt;
        current+=velocity*dt;
        return current;
    }

    [[nodiscard]] bool settled(f32 eps=0.001f) const {
        return std::abs(target-current)<eps&&std::abs(velocity)<eps;
    }
};

// ─── Reactive expression tree ─────────────────────────────────────────────────

enum class ExprOp { Const, Var, Add, Sub, Mul, Div, Abs, Sin, Cos, Pow, Min, Max, Lerp };

struct Expr {
    ExprOp op{ExprOp::Const};
    f32 const_val{0.f};
    u32 var_idx{0};
    std::vector<std::shared_ptr<Expr>> children;

    [[nodiscard]] f32 eval(std::span<const f32> vars) const {
        switch(op){
        case ExprOp::Const: return const_val;
        case ExprOp::Var:   return var_idx<vars.size()?vars[var_idx]:0.f;
        case ExprOp::Add:   return children[0]->eval(vars)+children[1]->eval(vars);
        case ExprOp::Sub:   return children[0]->eval(vars)-children[1]->eval(vars);
        case ExprOp::Mul:   return children[0]->eval(vars)*children[1]->eval(vars);
        case ExprOp::Div:   { f32 d=children[1]->eval(vars); return d!=0.f?children[0]->eval(vars)/d:0.f; }
        case ExprOp::Abs:   return std::abs(children[0]->eval(vars));
        case ExprOp::Sin:   return std::sin(children[0]->eval(vars));
        case ExprOp::Cos:   return std::cos(children[0]->eval(vars));
        case ExprOp::Pow:   return std::pow(children[0]->eval(vars),children[1]->eval(vars));
        case ExprOp::Min:   return std::min(children[0]->eval(vars),children[1]->eval(vars));
        case ExprOp::Max:   return std::max(children[0]->eval(vars),children[1]->eval(vars));
        case ExprOp::Lerp:  { f32 t=children[2]->eval(vars);
            return children[0]->eval(vars)*(1.f-t)+children[1]->eval(vars)*t; }
        }
        return 0.f;
    }
};

// Builder helpers
inline auto expr_const(f32 v) { auto e=std::make_shared<Expr>(); e->op=ExprOp::Const; e->const_val=v; return e; }
inline auto expr_var(u32 i)   { auto e=std::make_shared<Expr>(); e->op=ExprOp::Var;   e->var_idx=i;   return e; }
inline auto expr_add(std::shared_ptr<Expr> a, std::shared_ptr<Expr> b){
    auto e=std::make_shared<Expr>(); e->op=ExprOp::Add; e->children={a,b}; return e;}
inline auto expr_mul(std::shared_ptr<Expr> a, std::shared_ptr<Expr> b){
    auto e=std::make_shared<Expr>(); e->op=ExprOp::Mul; e->children={a,b}; return e;}
inline auto expr_sin(std::shared_ptr<Expr> a){
    auto e=std::make_shared<Expr>(); e->op=ExprOp::Sin; e->children={a}; return e;}
inline auto expr_lerp(std::shared_ptr<Expr> a, std::shared_ptr<Expr> b, std::shared_ptr<Expr> t){
    auto e=std::make_shared<Expr>(); e->op=ExprOp::Lerp; e->children={a,b,t}; return e;}

// Symbolic differentiation of expression tree (partial wrt variable idx)
[[nodiscard]] inline std::shared_ptr<Expr> expr_diff(const std::shared_ptr<Expr>& e, u32 var_idx);

[[nodiscard]] inline std::shared_ptr<Expr> expr_diff(const std::shared_ptr<Expr>& e, u32 vi) {
    switch(e->op){
    case ExprOp::Const: return expr_const(0.f);
    case ExprOp::Var:   return expr_const(e->var_idx==vi?1.f:0.f);
    case ExprOp::Add:   return expr_add(expr_diff(e->children[0],vi), expr_diff(e->children[1],vi));
    case ExprOp::Mul: {
        // Product rule: d(uv) = du*v + u*dv
        auto du=expr_diff(e->children[0],vi), dv=expr_diff(e->children[1],vi);
        return expr_add(expr_mul(du,e->children[1]), expr_mul(e->children[0],dv));
    }
    case ExprOp::Sin: {
        auto ex=std::make_shared<Expr>(); ex->op=ExprOp::Cos; ex->children={e->children[0]};
        return expr_mul(ex, expr_diff(e->children[0],vi));
    }
    case ExprOp::Cos: {
        auto ex=std::make_shared<Expr>(); ex->op=ExprOp::Sin; ex->children={e->children[0]};
        auto neg=expr_mul(expr_const(-1.f),ex);
        return expr_mul(neg, expr_diff(e->children[0],vi));
    }
    default: return expr_const(0.f); // unimplemented
    }
}

} // namespace qs
