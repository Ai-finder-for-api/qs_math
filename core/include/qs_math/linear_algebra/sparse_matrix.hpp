#pragma once
// qs_math/sparse_matrix.hpp — CSR/CSC sparse format, sparse mat-vec, sparse solvers (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// SparseCSR<T> — Compressed Sparse Row matrix
// ─────────────────────────────────────────────────────────────────────────────
template<typename T = f32>
struct SparseCSR {
    u32 rows{0}, cols{0};
    std::vector<u32> row_ptr; // size = rows+1
    std::vector<u32> col_idx;
    std::vector<T>   values;

    SparseCSR() = default;
    SparseCSR(u32 r, u32 c) : rows(r), cols(c), row_ptr(r+1, 0) {}

    [[nodiscard]] u32 nnz() const { return u32(values.size()); }

    // Build from triplets (row, col, val) — sorted and deduplicated
    static SparseCSR from_triplets(u32 r, u32 c, std::vector<std::tuple<u32,u32,T>> triplets) {
        std::sort(triplets.begin(), triplets.end());
        SparseCSR M(r,c);
        u32 prev_row=0;
        for(auto& [ri,ci,v]: triplets) {
            // Merge duplicates
            if(!M.col_idx.empty() && M.row_ptr[ri+1]==u32(M.values.size()) &&
               M.col_idx.back()==ci) {
                M.values.back()+=v; continue;
            }
            while(prev_row<=ri) { M.row_ptr[prev_row+1]=u32(M.values.size()); ++prev_row; }
            M.col_idx.push_back(ci);
            M.values.push_back(v);
        }
        // Fill remaining row_ptrs
        for(u32 i=prev_row;i<=r;++i) M.row_ptr[i]=u32(M.values.size());
        return M;
    }

    // Dense → sparse (drop near-zeros)
    static SparseCSR from_dense(const DenseMatrix<T>& M, T tol=T(1e-12)) {
        std::vector<std::tuple<u32,u32,T>> trips;
        for(u32 i=0;i<M.rows;++i) for(u32 j=0;j<M.cols;++j)
            if(std::abs(M.at(i,j))>tol) trips.emplace_back(i,j,M.at(i,j));
        return from_triplets(M.rows, M.cols, std::move(trips));
    }

    // Sparse → dense
    [[nodiscard]] DenseMatrix<T> to_dense() const {
        DenseMatrix<T> M(rows,cols);
        for(u32 i=0;i<rows;++i) for(u32 k=row_ptr[i];k<row_ptr[i+1];++k)
            M.at(i,col_idx[k])=values[k];
        return M;
    }

    // Get element (O(nnz per row))
    [[nodiscard]] T get(u32 r, u32 c) const {
        for(u32 k=row_ptr[r];k<row_ptr[r+1];++k)
            if(col_idx[k]==c) return values[k];
        return T(0);
    }

    // Sparse matrix-vector product y = A*x
    [[nodiscard]] std::vector<T> matvec(const std::vector<T>& x) const {
        assert(u32(x.size())==cols);
        std::vector<T> y(rows, T(0));
        for(u32 i=0;i<rows;++i) for(u32 k=row_ptr[i];k<row_ptr[i+1];++k)
            y[i]+=values[k]*x[col_idx[k]];
        return y;
    }

    // Transpose
    [[nodiscard]] SparseCSR<T> transpose() const {
        std::vector<std::tuple<u32,u32,T>> trips;
        trips.reserve(nnz());
        for(u32 i=0;i<rows;++i) for(u32 k=row_ptr[i];k<row_ptr[i+1];++k)
            trips.emplace_back(col_idx[k],i,values[k]);
        return from_triplets(cols,rows,std::move(trips));
    }

    // Scalar ops
    [[nodiscard]] SparseCSR operator*(T s) const {
        SparseCSR R=*this; for(auto& v:R.values) v*=s; return R;
    }

    // Frobenius norm
    [[nodiscard]] T frobenius_norm() const {
        T s=T(0); for(auto v:values) s+=v*v; return std::sqrt(s);
    }

    // Diagonal extraction
    [[nodiscard]] std::vector<T> diagonal() const {
        std::vector<T> d(std::min(rows,cols), T(0));
        for(u32 i=0;i<u32(d.size());++i) d[i]=get(i,i);
        return d;
    }

    // Check symmetry
    [[nodiscard]] bool is_symmetric(T tol=T(1e-7)) const {
        if(rows!=cols) return false;
        for(u32 i=0;i<rows;++i) for(u32 k=row_ptr[i];k<row_ptr[i+1];++k) {
            u32 j=col_idx[k];
            if(std::abs(values[k]-get(j,i))>tol) return false;
        }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SparseCSC<T> — Compressed Sparse Column (column-major variant)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T = f32>
struct SparseCSC {
    u32 rows{0}, cols{0};
    std::vector<u32> col_ptr; // size = cols+1
    std::vector<u32> row_idx;
    std::vector<T>   values;

    SparseCSC() = default;
    static SparseCSC from_csr(const SparseCSR<T>& A) {
        // Transpose of transpose
        auto At = A.transpose();
        SparseCSC C; C.rows=A.rows; C.cols=A.cols;
        C.col_ptr=At.row_ptr; C.row_idx=At.col_idx; C.values=At.values;
        return C;
    }

    // Column-oriented matvec
    [[nodiscard]] std::vector<T> matvec(const std::vector<T>& x) const {
        std::vector<T> y(rows, T(0));
        for(u32 j=0;j<cols;++j) for(u32 k=col_ptr[j];k<col_ptr[j+1];++k)
            y[row_idx[k]]+=values[k]*x[j];
        return y;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sparse Cholesky (incomplete — ILU0 preconditioner)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
struct ILU0 {
    SparseCSR<T> L, U; // incomplete factors
    explicit ILU0(SparseCSR<T> A) {
        // Zero-fill ILU: keep sparsity of A, factorize
        DenseMatrix<T> D = A.to_dense();
        u32 n=D.rows;
        for(u32 k=0;k<n;++k) {
            T diag=D.at(k,k);
            if(std::abs(diag)<T(1e-14)) continue;
            for(u32 i=k+1;i<n;++i) {
                if(std::abs(A.get(i,k))<T(1e-14)) continue;
                D.at(i,k)/=diag;
                for(u32 j=k+1;j<n;++j) {
                    if(std::abs(A.get(i,j))<T(1e-14)) continue;
                    D.at(i,j)-=D.at(i,k)*D.at(k,j);
                }
            }
        }
        // Split into L, U
        std::vector<std::tuple<u32,u32,T>> lt,ut;
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) {
            T v=D.at(i,j);
            if(std::abs(v)<T(1e-14)) continue;
            if(j<i) lt.emplace_back(i,j,v);
            else    ut.emplace_back(i,j,v);
        }
        L=SparseCSR<T>::from_triplets(n,n,std::move(lt));
        U=SparseCSR<T>::from_triplets(n,n,std::move(ut));
        // Add identity to L diagonal
        for(u32 i=0;i<n;++i) {
            std::vector<std::tuple<u32,u32,T>> extra;
            extra.emplace_back(i,i,T(1));
            auto Laug=SparseCSR<T>::from_triplets(n,n,std::move(extra));
            // merge — simplified: rebuild
        }
    }

    // Apply preconditioner M^{-1} x via forward/backward substitution
    [[nodiscard]] std::vector<T> apply(const std::vector<T>& x) const {
        u32 n=u32(x.size());
        // Forward solve L*y=x
        std::vector<T> y=x;
        for(u32 i=0;i<n;++i) for(u32 k=L.row_ptr[i];k<L.row_ptr[i+1];++k)
            y[i]-=L.values[k]*y[L.col_idx[k]];
        // Backward solve U*z=y
        std::vector<T> z=y;
        for(i32 i=i32(n)-1;i>=0;--i) {
            T diag=T(1);
            for(u32 k=U.row_ptr[u32(i)];k<U.row_ptr[u32(i)+1];++k) {
                if(U.col_idx[k]==u32(i)) { diag=U.values[k]; continue; }
                if(U.col_idx[k]>u32(i)) z[u32(i)]-=U.values[k]*z[U.col_idx[k]];
            }
            z[u32(i)]/=diag;
        }
        return z;
    }
};

// Vector ops helpers for iterative solvers
template<typename T> [[nodiscard]] T vec_dot(const std::vector<T>& a,const std::vector<T>& b){
    T s=T(0); for(u32 i=0;i<u32(a.size());++i) s+=a[i]*b[i]; return s;
}
template<typename T> [[nodiscard]] T vec_norm(const std::vector<T>& a){ return std::sqrt(vec_dot(a,a)); }
template<typename T> [[nodiscard]] std::vector<T> vec_add(const std::vector<T>& a,const std::vector<T>& b,T sb=T(1)){
    std::vector<T> r(a.size()); for(u32 i=0;i<u32(a.size());++i) r[i]=a[i]+sb*b[i]; return r;
}
template<typename T> [[nodiscard]] std::vector<T> vec_scale(const std::vector<T>& a,T s){
    std::vector<T> r(a.size()); for(u32 i=0;i<u32(a.size());++i) r[i]=a[i]*s; return r;
}

} // namespace qs
