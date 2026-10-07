#pragma once
// qs_math/symbolic_math.hpp — CAS expression nodes, simplification, LaTeX emit (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <algorithm>
#include <optional>

namespace qs {

// ─── Expression node ──────────────────────────────────────────────────────────

enum class ExprKind {
    Num, Var, Add, Sub, Mul, Div, Pow,
    Sin, Cos, Tan, Exp, Log, Sqrt, Abs, Neg
};

struct Expr {
    ExprKind kind{ExprKind::Num};
    f64 num{0.};
    std::string var{};
    std::vector<std::shared_ptr<Expr>> args;

    [[nodiscard]] static std::shared_ptr<Expr> make_num(f64 v) {
        auto e=std::make_shared<Expr>(); e->kind=ExprKind::Num; e->num=v; return e; }
    [[nodiscard]] static std::shared_ptr<Expr> make_var(const std::string& s) {
        auto e=std::make_shared<Expr>(); e->kind=ExprKind::Var; e->var=s; return e; }
    [[nodiscard]] static std::shared_ptr<Expr> make_binop(ExprKind k, std::shared_ptr<Expr> a, std::shared_ptr<Expr> b) {
        auto e=std::make_shared<Expr>(); e->kind=k; e->args={a,b}; return e; }
    [[nodiscard]] static std::shared_ptr<Expr> make_unary(ExprKind k, std::shared_ptr<Expr> a) {
        auto e=std::make_shared<Expr>(); e->kind=k; e->args={a}; return e; }

    [[nodiscard]] bool is_num(f64 v) const { return kind==ExprKind::Num&&std::abs(num-v)<1e-12; }
    [[nodiscard]] bool is_var(const std::string& s) const { return kind==ExprKind::Var&&var==s; }
};

using ExprPtr = std::shared_ptr<Expr>;

// Convenience builders
inline ExprPtr enum(f64 v)                         { return Expr::make_num(v); }
inline ExprPtr evar(const std::string& s)           { return Expr::make_var(s); }
inline ExprPtr eadd(ExprPtr a, ExprPtr b)           { return Expr::make_binop(ExprKind::Add,a,b); }
inline ExprPtr esub(ExprPtr a, ExprPtr b)           { return Expr::make_binop(ExprKind::Sub,a,b); }
inline ExprPtr emul(ExprPtr a, ExprPtr b)           { return Expr::make_binop(ExprKind::Mul,a,b); }
inline ExprPtr ediv(ExprPtr a, ExprPtr b)           { return Expr::make_binop(ExprKind::Div,a,b); }
inline ExprPtr epow(ExprPtr a, ExprPtr b)           { return Expr::make_binop(ExprKind::Pow,a,b); }
inline ExprPtr esin(ExprPtr a)                      { return Expr::make_unary(ExprKind::Sin,a); }
inline ExprPtr ecos(ExprPtr a)                      { return Expr::make_unary(ExprKind::Cos,a); }
inline ExprPtr eexp(ExprPtr a)                      { return Expr::make_unary(ExprKind::Exp,a); }
inline ExprPtr elog(ExprPtr a)                      { return Expr::make_unary(ExprKind::Log,a); }
inline ExprPtr esqrt(ExprPtr a)                     { return Expr::make_unary(ExprKind::Sqrt,a); }
inline ExprPtr eneg(ExprPtr a)                      { return Expr::make_unary(ExprKind::Neg,a); }

// ─── Evaluate ────────────────────────────────────────────────────────────────

using VarMap = std::unordered_map<std::string,f64>;

[[nodiscard]] inline f64 eval_expr(const ExprPtr& e, const VarMap& vars) {
    switch(e->kind){
    case ExprKind::Num:  return e->num;
    case ExprKind::Var:  { auto it=vars.find(e->var); return it!=vars.end()?it->second:0.; }
    case ExprKind::Add:  return eval_expr(e->args[0],vars)+eval_expr(e->args[1],vars);
    case ExprKind::Sub:  return eval_expr(e->args[0],vars)-eval_expr(e->args[1],vars);
    case ExprKind::Mul:  return eval_expr(e->args[0],vars)*eval_expr(e->args[1],vars);
    case ExprKind::Div:  { f64 d=eval_expr(e->args[1],vars); return d?eval_expr(e->args[0],vars)/d:0.; }
    case ExprKind::Pow:  return std::pow(eval_expr(e->args[0],vars),eval_expr(e->args[1],vars));
    case ExprKind::Sin:  return std::sin(eval_expr(e->args[0],vars));
    case ExprKind::Cos:  return std::cos(eval_expr(e->args[0],vars));
    case ExprKind::Tan:  return std::tan(eval_expr(e->args[0],vars));
    case ExprKind::Exp:  return std::exp(eval_expr(e->args[0],vars));
    case ExprKind::Log:  return std::log(eval_expr(e->args[0],vars));
    case ExprKind::Sqrt: return std::sqrt(eval_expr(e->args[0],vars));
    case ExprKind::Abs:  return std::abs(eval_expr(e->args[0],vars));
    case ExprKind::Neg:  return -eval_expr(e->args[0],vars);
    }
    return 0.;
}

// ─── Symbolic differentiation ─────────────────────────────────────────────────

[[nodiscard]] ExprPtr diff_expr(const ExprPtr& e, const std::string& wrt);

[[nodiscard]] inline ExprPtr diff_expr(const ExprPtr& e, const std::string& wrt) {
    auto Z=enum(0.), O=enum(1.);
    switch(e->kind){
    case ExprKind::Num: return Z;
    case ExprKind::Var: return e->var==wrt?O:Z;
    case ExprKind::Add: return eadd(diff_expr(e->args[0],wrt),diff_expr(e->args[1],wrt));
    case ExprKind::Sub: return esub(diff_expr(e->args[0],wrt),diff_expr(e->args[1],wrt));
    case ExprKind::Mul: return eadd(emul(diff_expr(e->args[0],wrt),e->args[1]),
                                    emul(e->args[0],diff_expr(e->args[1],wrt)));
    case ExprKind::Div: return ediv(esub(emul(diff_expr(e->args[0],wrt),e->args[1]),
                                         emul(e->args[0],diff_expr(e->args[1],wrt))),
                                    epow(e->args[1],enum(2.)));
    case ExprKind::Pow: // d/dx u^v = v*u^(v-1)*du/dx + u^v*ln(u)*dv/dx
        return eadd(emul(emul(e->args[1],epow(e->args[0],esub(e->args[1],O))),diff_expr(e->args[0],wrt)),
                    emul(emul(epow(e->args[0],e->args[1]),elog(e->args[0])),diff_expr(e->args[1],wrt)));
    case ExprKind::Sin: return emul(ecos(e->args[0]),diff_expr(e->args[0],wrt));
    case ExprKind::Cos: return eneg(emul(esin(e->args[0]),diff_expr(e->args[0],wrt)));
    case ExprKind::Exp: return emul(e,diff_expr(e->args[0],wrt));
    case ExprKind::Log: return ediv(diff_expr(e->args[0],wrt),e->args[0]);
    case ExprKind::Sqrt: return ediv(diff_expr(e->args[0],wrt),emul(enum(2.),esqrt(e->args[0])));
    case ExprKind::Neg: return eneg(diff_expr(e->args[0],wrt));
    default: return Z;
    }
}

// ─── Simplification rules ────────────────────────────────────────────────────

[[nodiscard]] ExprPtr simplify(const ExprPtr& e);

[[nodiscard]] inline ExprPtr simplify(const ExprPtr& e) {
    if(e->args.empty()) return e;
    std::vector<ExprPtr> sargs;
    for(auto& a:e->args) sargs.push_back(simplify(a));
    // Constant folding
    bool all_num=true;
    for(auto& a:sargs) if(a->kind!=ExprKind::Num){all_num=false;break;}
    if(all_num){
        VarMap vm;
        auto tmp=std::make_shared<Expr>(*e); tmp->args=sargs;
        return enum(eval_expr(tmp,vm));
    }
    switch(e->kind){
    case ExprKind::Add:
        if(sargs[0]->is_num(0.)) return sargs[1];
        if(sargs[1]->is_num(0.)) return sargs[0];
        break;
    case ExprKind::Sub:
        if(sargs[1]->is_num(0.)) return sargs[0];
        break;
    case ExprKind::Mul:
        if(sargs[0]->is_num(0.)||sargs[1]->is_num(0.)) return enum(0.);
        if(sargs[0]->is_num(1.)) return sargs[1];
        if(sargs[1]->is_num(1.)) return sargs[0];
        break;
    case ExprKind::Div:
        if(sargs[0]->is_num(0.)) return enum(0.);
        if(sargs[1]->is_num(1.)) return sargs[0];
        break;
    case ExprKind::Pow:
        if(sargs[1]->is_num(0.)) return enum(1.);
        if(sargs[1]->is_num(1.)) return sargs[0];
        if(sargs[0]->is_num(1.)) return enum(1.);
        break;
    default: break;
    }
    auto r=std::make_shared<Expr>(*e); r->args=sargs; return r;
}

// ─── LaTeX emission ───────────────────────────────────────────────────────────

[[nodiscard]] inline std::string to_latex(const ExprPtr& e) {
    switch(e->kind){
    case ExprKind::Num:  {
        std::ostringstream ss; ss<<e->num; return ss.str(); }
    case ExprKind::Var:  return e->var;
    case ExprKind::Add:  return to_latex(e->args[0])+" + "+to_latex(e->args[1]);
    case ExprKind::Sub:  return to_latex(e->args[0])+" - "+to_latex(e->args[1]);
    case ExprKind::Mul:  return "("+to_latex(e->args[0])+") \\cdot ("+to_latex(e->args[1])+")";
    case ExprKind::Div:  return "\\frac{"+to_latex(e->args[0])+"}{"+to_latex(e->args[1])+"}";
    case ExprKind::Pow:  return "{"+to_latex(e->args[0])+"}^{"+to_latex(e->args[1])+"}";
    case ExprKind::Sin:  return "\\sin("+to_latex(e->args[0])+")";
    case ExprKind::Cos:  return "\\cos("+to_latex(e->args[0])+")";
    case ExprKind::Tan:  return "\\tan("+to_latex(e->args[0])+")";
    case ExprKind::Exp:  return "e^{"+to_latex(e->args[0])+"}";
    case ExprKind::Log:  return "\\ln("+to_latex(e->args[0])+")";
    case ExprKind::Sqrt: return "\\sqrt{"+to_latex(e->args[0])+"}";
    case ExprKind::Abs:  return "\\left|"+to_latex(e->args[0])+"\\right|";
    case ExprKind::Neg:  return "-"+to_latex(e->args[0]);
    }
    return "";
}

// ─── String emission ──────────────────────────────────────────────────────────

[[nodiscard]] inline std::string to_string(const ExprPtr& e) {
    switch(e->kind){
    case ExprKind::Num: { std::ostringstream ss; ss<<e->num; return ss.str(); }
    case ExprKind::Var: return e->var;
    case ExprKind::Add: return "("+to_string(e->args[0])+" + "+to_string(e->args[1])+")";
    case ExprKind::Sub: return "("+to_string(e->args[0])+" - "+to_string(e->args[1])+")";
    case ExprKind::Mul: return "("+to_string(e->args[0])+" * "+to_string(e->args[1])+")";
    case ExprKind::Div: return "("+to_string(e->args[0])+" / "+to_string(e->args[1])+")";
    case ExprKind::Pow: return "pow("+to_string(e->args[0])+", "+to_string(e->args[1])+")";
    case ExprKind::Sin: return "sin("+to_string(e->args[0])+")";
    case ExprKind::Cos: return "cos("+to_string(e->args[0])+")";
    case ExprKind::Exp: return "exp("+to_string(e->args[0])+")";
    case ExprKind::Log: return "log("+to_string(e->args[0])+")";
    case ExprKind::Sqrt:return "sqrt("+to_string(e->args[0])+")";
    case ExprKind::Abs: return "abs("+to_string(e->args[0])+")";
    case ExprKind::Neg: return "(-"+to_string(e->args[0])+")";
    default: return "?";
    }
}

// ─── Expression pattern matching ─────────────────────────────────────────────

[[nodiscard]] inline bool expr_equal(const ExprPtr& a, const ExprPtr& b) {
    if(a->kind!=b->kind) return false;
    if(a->kind==ExprKind::Num) return std::abs(a->num-b->num)<1e-12;
    if(a->kind==ExprKind::Var) return a->var==b->var;
    if(a->args.size()!=b->args.size()) return false;
    for(u32 i=0;i<a->args.size();++i) if(!expr_equal(a->args[i],b->args[i])) return false;
    return true;
}

// Substitute variable with expression
[[nodiscard]] inline ExprPtr substitute(const ExprPtr& e, const std::string& var, const ExprPtr& val) {
    if(e->kind==ExprKind::Var&&e->var==var) return val;
    if(e->args.empty()) return e;
    auto r=std::make_shared<Expr>(*e);
    r->args.clear();
    for(const auto& a:e->args) r->args.push_back(substitute(a,var,val));
    return r;
}

// Count occurrences of variable
[[nodiscard]] inline u32 count_var(const ExprPtr& e, const std::string& var) {
    if(e->kind==ExprKind::Var&&e->var==var) return 1;
    u32 c=0; for(const auto& a:e->args) c+=count_var(a,var);
    return c;
}

// ─── Gradient vector ──────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<ExprPtr> gradient(
    const ExprPtr& f, const std::vector<std::string>& vars)
{
    std::vector<ExprPtr> grad;
    for(const auto& v:vars) grad.push_back(simplify(diff_expr(f,v)));
    return grad;
}

// Hessian matrix
[[nodiscard]] inline std::vector<std::vector<ExprPtr>> hessian(
    const ExprPtr& f, const std::vector<std::string>& vars)
{
    u32 n=u32(vars.size());
    std::vector<std::vector<ExprPtr>> H(n,std::vector<ExprPtr>(n));
    auto grad=gradient(f,vars);
    for(u32 i=0;i<n;++i)
        for(u32 j=0;j<n;++j)
            H[i][j]=simplify(diff_expr(grad[i],vars[j]));
    return H;
}

} // namespace qs
