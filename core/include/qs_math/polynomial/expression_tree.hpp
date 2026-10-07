#pragma once
// qs_math/expression_tree.hpp — AST for math expressions, evaluation, symbolic differentiation (C++20)
#include "scalar_types.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <cassert>

namespace qs {

// ─── Expression node types ────────────────────────────────────────────────────
enum class ExprType : u8 {
    Num,    // constant
    Var,    // variable
    Add, Sub, Mul, Div, Pow, Neg,
    Sin, Cos, Tan, Exp, Log, Sqrt, Abs,
    Atan2,  // two-arg
};

struct Expr {
    ExprType type;
    f64      val{0};       // for Num
    std::string var_name;  // for Var
    std::vector<std::shared_ptr<Expr>> children;

    // Factory methods
    static std::shared_ptr<Expr> num(f64 v) {
        auto e=std::make_shared<Expr>(); e->type=ExprType::Num; e->val=v; return e;
    }
    static std::shared_ptr<Expr> var(std::string name) {
        auto e=std::make_shared<Expr>(); e->type=ExprType::Var; e->var_name=std::move(name); return e;
    }
    static std::shared_ptr<Expr> make(ExprType t, std::shared_ptr<Expr> a, std::shared_ptr<Expr> b=nullptr) {
        auto e=std::make_shared<Expr>(); e->type=t; e->children.push_back(a);
        if(b) e->children.push_back(b);
        return e;
    }
};

using ExprPtr=std::shared_ptr<Expr>;
using VarMap=std::unordered_map<std::string,f64>;

// Convenience operators
inline ExprPtr operator+(ExprPtr a,ExprPtr b){return Expr::make(ExprType::Add,a,b);}
inline ExprPtr operator-(ExprPtr a,ExprPtr b){return Expr::make(ExprType::Sub,a,b);}
inline ExprPtr operator*(ExprPtr a,ExprPtr b){return Expr::make(ExprType::Mul,a,b);}
inline ExprPtr operator/(ExprPtr a,ExprPtr b){return Expr::make(ExprType::Div,a,b);}
inline ExprPtr pow_expr(ExprPtr a,ExprPtr b) {return Expr::make(ExprType::Pow,a,b);}
inline ExprPtr neg_expr(ExprPtr a)           {return Expr::make(ExprType::Neg,a);}
inline ExprPtr sin_expr(ExprPtr a)           {return Expr::make(ExprType::Sin,a);}
inline ExprPtr cos_expr(ExprPtr a)           {return Expr::make(ExprType::Cos,a);}
inline ExprPtr exp_expr(ExprPtr a)           {return Expr::make(ExprType::Exp,a);}
inline ExprPtr log_expr(ExprPtr a)           {return Expr::make(ExprType::Log,a);}
inline ExprPtr sqrt_expr(ExprPtr a)          {return Expr::make(ExprType::Sqrt,a);}
inline ExprPtr num(f64 v)                    {return Expr::num(v);}
inline ExprPtr var(std::string s)            {return Expr::var(std::move(s));}

// ─── Evaluation ───────────────────────────────────────────────────────────────
[[nodiscard]] inline f64 eval(const ExprPtr& e, const VarMap& vars={}) {
    switch(e->type) {
        case ExprType::Num: return e->val;
        case ExprType::Var: {
            auto it=vars.find(e->var_name);
            if(it==vars.end()) throw std::runtime_error("Undefined variable: "+e->var_name);
            return it->second;
        }
        case ExprType::Add:  return eval(e->children[0],vars)+eval(e->children[1],vars);
        case ExprType::Sub:  return eval(e->children[0],vars)-eval(e->children[1],vars);
        case ExprType::Mul:  return eval(e->children[0],vars)*eval(e->children[1],vars);
        case ExprType::Div:  return eval(e->children[0],vars)/eval(e->children[1],vars);
        case ExprType::Pow:  return std::pow(eval(e->children[0],vars),eval(e->children[1],vars));
        case ExprType::Neg:  return -eval(e->children[0],vars);
        case ExprType::Sin:  return std::sin(eval(e->children[0],vars));
        case ExprType::Cos:  return std::cos(eval(e->children[0],vars));
        case ExprType::Tan:  return std::tan(eval(e->children[0],vars));
        case ExprType::Exp:  return std::exp(eval(e->children[0],vars));
        case ExprType::Log:  return std::log(eval(e->children[0],vars));
        case ExprType::Sqrt: return std::sqrt(eval(e->children[0],vars));
        case ExprType::Abs:  return std::abs(eval(e->children[0],vars));
        case ExprType::Atan2:return std::atan2(eval(e->children[0],vars),eval(e->children[1],vars));
        default: return 0;
    }
}

// ─── Symbolic differentiation d/dx ────────────────────────────────────────────
[[nodiscard]] inline ExprPtr diff(const ExprPtr& e, const std::string& x);

[[nodiscard]] inline ExprPtr simplify(const ExprPtr& e) {
    // Basic constant folding
    if(e->type==ExprType::Add||e->type==ExprType::Sub||e->type==ExprType::Mul||e->type==ExprType::Div) {
        auto a=simplify(e->children[0]), b=simplify(e->children[1]);
        if(a->type==ExprType::Num&&b->type==ExprType::Num) {
            f64 v=0;
            switch(e->type){
                case ExprType::Add: v=a->val+b->val; break;
                case ExprType::Sub: v=a->val-b->val; break;
                case ExprType::Mul: v=a->val*b->val; break;
                case ExprType::Div: v=a->val/b->val; break;
                default: break;
            }
            return num(v);
        }
        // x + 0 = x, x * 1 = x, x * 0 = 0, etc.
        if(e->type==ExprType::Add&&b->type==ExprType::Num&&b->val==0) return a;
        if(e->type==ExprType::Add&&a->type==ExprType::Num&&a->val==0) return b;
        if(e->type==ExprType::Mul&&b->type==ExprType::Num&&b->val==1) return a;
        if(e->type==ExprType::Mul&&a->type==ExprType::Num&&a->val==1) return b;
        if(e->type==ExprType::Mul&&b->type==ExprType::Num&&b->val==0) return num(0);
        if(e->type==ExprType::Mul&&a->type==ExprType::Num&&a->val==0) return num(0);
        if(e->type==ExprType::Sub&&b->type==ExprType::Num&&b->val==0) return a;
        auto r=Expr::make(e->type,a,b); return r;
    }
    if(e->children.size()==1) {
        auto a=simplify(e->children[0]);
        if(a->type==ExprType::Num) {
            switch(e->type){
                case ExprType::Neg:  return num(-a->val);
                case ExprType::Sin:  return num(std::sin(a->val));
                case ExprType::Cos:  return num(std::cos(a->val));
                case ExprType::Exp:  return num(std::exp(a->val));
                case ExprType::Log:  return num(std::log(a->val));
                case ExprType::Sqrt: return num(std::sqrt(a->val));
                default: break;
            }
        }
        return Expr::make(e->type,a);
    }
    return e;
}

[[nodiscard]] inline ExprPtr diff(const ExprPtr& e, const std::string& x) {
    switch(e->type) {
        case ExprType::Num: return num(0);
        case ExprType::Var: return num(e->var_name==x?1:0);
        case ExprType::Neg: return neg_expr(diff(e->children[0],x));
        case ExprType::Add: return simplify(diff(e->children[0],x)+diff(e->children[1],x));
        case ExprType::Sub: return simplify(diff(e->children[0],x)-diff(e->children[1],x));
        case ExprType::Mul: { // product rule
            auto a=e->children[0],b=e->children[1];
            return simplify(diff(a,x)*b+a*diff(b,x));
        }
        case ExprType::Div: { // quotient rule
            auto a=e->children[0],b=e->children[1];
            return simplify((diff(a,x)*b-a*diff(b,x))/(b*b));
        }
        case ExprType::Pow: { // power rule (constant exponent) and chain rule
            auto f_=e->children[0],g=e->children[1];
            if(g->type==ExprType::Num) {
                f64 n=g->val;
                return simplify(num(n)*pow_expr(f_,num(n-1))*diff(f_,x));
            }
            // General: (f^g)' = f^g * (g'*ln(f) + g*f'/f)
            return simplify(pow_expr(f_,g)*(diff(g,x)*log_expr(f_)+g*diff(f_,x)/f_));
        }
        case ExprType::Sin: return simplify(cos_expr(e->children[0])*diff(e->children[0],x));
        case ExprType::Cos: return simplify(neg_expr(sin_expr(e->children[0]))*diff(e->children[0],x));
        case ExprType::Exp: return simplify(exp_expr(e->children[0])*diff(e->children[0],x));
        case ExprType::Log: return simplify(diff(e->children[0],x)/e->children[0]);
        case ExprType::Sqrt: return simplify(diff(e->children[0],x)/(num(2)*sqrt_expr(e->children[0])));
        default: return num(0);
    }
}

// ─── String conversion ────────────────────────────────────────────────────────
[[nodiscard]] inline std::string to_string(const ExprPtr& e) {
    switch(e->type) {
        case ExprType::Num: {
            std::ostringstream ss; ss<<e->val; return ss.str();
        }
        case ExprType::Var: return e->var_name;
        case ExprType::Neg: return "(-"+to_string(e->children[0])+")";
        case ExprType::Add: return "("+to_string(e->children[0])+"+"+to_string(e->children[1])+")";
        case ExprType::Sub: return "("+to_string(e->children[0])+"-"+to_string(e->children[1])+")";
        case ExprType::Mul: return "("+to_string(e->children[0])+"*"+to_string(e->children[1])+")";
        case ExprType::Div: return "("+to_string(e->children[0])+"/"+to_string(e->children[1])+")";
        case ExprType::Pow: return "("+to_string(e->children[0])+"^"+to_string(e->children[1])+")";
        case ExprType::Sin: return "sin("+to_string(e->children[0])+")";
        case ExprType::Cos: return "cos("+to_string(e->children[0])+")";
        case ExprType::Exp: return "exp("+to_string(e->children[0])+")";
        case ExprType::Log: return "log("+to_string(e->children[0])+")";
        case ExprType::Sqrt:return "sqrt("+to_string(e->children[0])+")";
        default: return "?";
    }
}

// ─── Compile to a callable (for performance) ──────────────────────────────────
[[nodiscard]] inline std::function<f64(const VarMap&)> compile(ExprPtr e) {
    return [e](const VarMap& vars){ return eval(e,vars); };
}

// ─── Gradient (vector of partial derivatives) ─────────────────────────────────
[[nodiscard]] inline std::vector<ExprPtr> gradient(const ExprPtr& e, const std::vector<std::string>& vars) {
    std::vector<ExprPtr> grad;
    grad.reserve(vars.size());
    for(const auto& v:vars) grad.push_back(simplify(diff(e,v)));
    return grad;
}

// ─── Simple recursive descent parser: num, var, +, -, *, /, ^ ─────────────────
// Parse an expression string like "x^2 + sin(y) * 3.14"
struct ExprParser {
    std::string src;
    u32 pos{0};

    void skip_ws(){ while(pos<src.size()&&src[pos]==' ') ++pos; }
    char peek(){ skip_ws(); return pos<src.size()?src[pos]:0; }
    char get() { skip_ws(); return pos<src.size()?src[pos++]:0; }
    bool accept(char c){ skip_ws(); if(pos<src.size()&&src[pos]==c){++pos;return true;}return false; }

    ExprPtr parse_expr()   { auto a=parse_term();  while(peek()=='+'||peek()=='-'){ char op=get(); a=Expr::make(op=='+'?ExprType::Add:ExprType::Sub,a,parse_term()); } return a; }
    ExprPtr parse_term()   { auto a=parse_pow();   while(peek()=='*'||peek()=='/'){ char op=get(); a=Expr::make(op=='*'?ExprType::Mul:ExprType::Div,a,parse_pow());  } return a; }
    ExprPtr parse_pow()    { auto a=parse_unary(); if(peek()=='^'){get();return pow_expr(a,parse_pow());} return a; }
    ExprPtr parse_unary()  { if(accept('-'))return neg_expr(parse_primary()); accept('+'); return parse_primary(); }
    ExprPtr parse_primary() {
        if(accept('(')){ auto e=parse_expr(); accept(')'); return e; }
        if(std::isdigit(peek())||peek()=='.'){ return parse_num(); }
        return parse_func_or_var();
    }
    ExprPtr parse_num() {
        skip_ws(); std::string s;
        while(pos<src.size()&&(std::isdigit(src[pos])||src[pos]=='.')) s+=src[pos++];
        if(pos<src.size()&&src[pos]=='e'){ s+=src[pos++]; if(pos<src.size()&&(src[pos]=='+'||src[pos]=='-')) s+=src[pos++]; while(pos<src.size()&&std::isdigit(src[pos])) s+=src[pos++]; }
        return num(std::stod(s));
    }
    ExprPtr parse_func_or_var() {
        skip_ws(); std::string name;
        while(pos<src.size()&&(std::isalpha(src[pos])||src[pos]=='_'||std::isdigit(src[pos]))) name+=src[pos++];
        if(accept('(')) {
            auto arg=parse_expr(); accept(',');
            std::shared_ptr<Expr> arg2;
            if(peek()!=')'&&name=="atan2") arg2=parse_expr();
            accept(')');
            if(name=="sin") return sin_expr(arg);
            if(name=="cos") return cos_expr(arg);
            if(name=="exp") return exp_expr(arg);
            if(name=="log"||name=="ln") return log_expr(arg);
            if(name=="sqrt") return sqrt_expr(arg);
            if(name=="abs") return Expr::make(ExprType::Abs,arg);
            if(name=="atan2"&&arg2) return Expr::make(ExprType::Atan2,arg,arg2);
            return arg; // fallback
        }
        if(name=="pi") return num(std::numbers::pi_v<f64>);
        if(name=="e")  return num(std::numbers::e_v<f64>);
        return var(name);
    }

    static ExprPtr parse(const std::string& s) { ExprParser p; p.src=s; return p.parse_expr(); }
};

} // namespace qs
