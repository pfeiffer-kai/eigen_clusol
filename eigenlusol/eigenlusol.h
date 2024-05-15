// This file is part of EIGENLUSOL.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef _EIGENLUSOL_
#define _EIGENLUSOL_

#pragma once

#include "typedefs.h"
#include "options.h"
#include <clusol/clusol.h>

/* This code describes an interface betwee eigen and (c)lusol (Gill, Murray, Saunders, Wright, 1984)
 * https://github.com/nwh/lusol
 * add  "#ifdef __cplusplus extern "C"{ #endif' compiler flag to clusol.h
 * compile and install libclusol.so and clusol.h in /usr/local/lib and /usr/local/include/clusol, respectively
 */

// ToDo:
// Can the interface between Eigen::Sparse and here be made faster? With a simple pointer or similar, currently full copy

namespace cpplusol {

    template <typename T> void print(string name, T* a, int len)
    {
        cout << name << ":\n";
        for (int i=0;i<len;i++)
        {
            cout << (a)[i] << " ";
        }
        cout << endl;
    }

    class eigenlusol
    {
        // Remark:
        // lusol convention, indices start from 1
        // so +1 indexing if going into lusol
        // so -1 indexing if coming from lusol

        public:
            eigenlusol(options _opt = cpplusol::options()) :
                opt(_opt)
            {
                // if (!hp) { cout << "eigenlusol: no workspace available" << endl; throw; }
                allocate(); // {'TPP','TRP','TCP','TSP', 'TBP}));
            }

            void reset();

            // sparse LU decomposition of A
            // if construct==true, this computes the upper triangular factors L and U such that
            // A = P' * L * U * Q'
            // if construct==false
            // A = L * U
            // and P and Q are not computed
            void factorize(const mat& A, bool construct=true);
            void factorize_partial(const mat& A, const int j, const int rg=1, bool construct=true);
            int factorize_bin(const mat& A, const veci& cols, int& nrRows, bool construct=true, int n1_=-1, int n2_=-1, int np_=-1);
            inline void addCol(const mat& A, const int j);
            inline int repCol(const mat& A, const int j, const int _jrep, const int _mode1 = 1, const int _mode2 = 1);

            // computes the (implicit: construct false, explicit: construct true) NS of the matrix A
            // Z = P' * inv(L') * [0; I] with A' = P' L U Q'
            void computeNS(const mat& A, shared_ptr<mat> Zio=NULL, bool construct=false);
            // apply NS on the right of matrix B -> BZ (not in place)
            mat applyNSOnTheRight(const mat& B);
            void applyNSOnTheRight(mat& B, vector<mat>& _storage);
            // apply NS on the left of vector b -> Zb (not in place)
            mat applyNSTOnTheLeft(mat& B);
            vec applyNSOnTheLeft(const vec& b);

            // solve A x = B (or b) with the exisiting decomposition
            // note that it is up to the user to keep track of whether the current instance of eigenlusol was used to
            // - factorize a matrix directly (A)
            // - factorize it indirectly through computeNS (A')
            void solveInPlace(mat& B);
            void solveInPlace(vec& b);
            vec solveInPlaceT(const vec& b); // solves ATA of the existing decomposition of A (?)
            vec solveInPlaceTT(const vec& b); // solves with the transpose of the existing decomposition

            // return initial decomposition (in terms of updated decomposition)
            inline Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>& P0() { return _P0; }
            inline mat& L0() { return _L0; }
            inline mat& U0() { return _U0; }
            inline Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>& Q0() { return _Q0; }
            // return current decomposition
            inline Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>& P() { if (!_P) { constructP(); } return *_P; }
            inline mat& L() { if (!_L) { constructL0(true); } return *_L; }
            inline mat& U() { if (!_U) { constructU(true); } return *_U; }
            inline Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>& Q() { if (!_Q) { constructQ(); } return *_Q; }

            inline mat& Z() { if (!_Z) { constructZ(); } return *_Z; }
            inline int64_t* qq() { return q; }
            inline int64_t* pp() { return p; }
            inline int64_t* invqq() { return iqinv; }
            inline int64_t* invpp() { return ipinv; }

            int rank() { return info->nrank; }
            int rank0() { return _rank0; }

            void constructL0(bool permute=false);
            bool constructU(bool permute=false);
            // constructs U and u for the turnbull algorithm with the assumption A = PT L [U u] QT
            bool constructU_partial(bool permute=false, const int uidx=-1, vec* u=NULL);
            bool constructU_partial(mat& U2, int n1, int n2, int np, veci& elimCols, veci& chosenColsP, veci& chosenColsPinv, int tid);
            void constructu_partial(vec& u);
            void getv(vec& u);
            void constructP();
            void constructQ();

            double getUcond();
            int lenRow(int idx);

            void constructZ(shared_ptr<mat> Zio = NULL);

            void computeInvLP();

// #if SAFEGUARD
            // matrix associated with decomposition
            mat _A;
// #endif

            void printPb();

            struct lusolinfo {
                int64_t inform; 
                int64_t nsing; 
                int64_t jsing;
                int64_t minlen;
                int64_t maxlen;
                int64_t nupdat;
                int64_t nrank;
                int64_t ndens1;
                int64_t ndens2;
                int64_t jumin;
                int64_t numL0;
                int64_t lenL0;
                int64_t lenU0;
                int64_t lenL; 
                int64_t lenU;
                int64_t lrow; 
                int64_t ncp;
                int64_t mersum;
                int64_t nUtri;
                int64_t nLtri;
                double Amax;
                double Lmax;
                double Umax;
                double DUmax;
                double DUmin;
                double Akmax; 
                double growth;
                double resid; 
            }* info = NULL;

            void assignInfo();

            int getm() { return (int)(*m); }

            unique_ptr<mat> _L = NULL; // L factor
            unique_ptr<mat> _U = NULL; // U factor
            unique_ptr<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> > _P = NULL;
            unique_ptr<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> > _Q = NULL;

            int _rank0;
            mat _L0;
            mat _U0;
            Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> _P0;
            Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> _Q0;

            void updateInvpq();

            options& getOpt() { return opt; }

#if TIMEMEASUREMENTS
            std::vector<cpplusol::time> times;
#endif

        private:

            void allocate();
            void init(const mat& A);
            void allocate(const mat& A);

            unique_ptr<mat> _Z = NULL; // U factor

            // temps for solves
            // helpful if same solve / projection operations are done several times
            // FIXME: disable on user input
            unique_ptr<mat> _invLP = NULL; // inv(L) * P
            unique_ptr<mat> _QinvUinvLP = NULL; // inverse of _A: Q inv(U) inv(L) P

            cpplusol::tripletHandler th, th2;

            // problem and solver variables
            int64_t* m;
            int64_t* n;
            int64_t* nelem;
            int64_t* lena;
            int maxmn;
            int64_t* luparm;
            double* parmlu;
            double* a;
            int64_t* indc;
            int64_t* indr;
            int64_t* p;
            int64_t* q;
            int64_t* lenc;
            int64_t* lenr;
            int64_t* locc;
            int64_t* locr;
            int64_t* iploc;
            int64_t* iqloc;
            int64_t* ipinv; // inverse of p
            int64_t* iqinv; // inverse of q
            double* w;
            int64_t* inform;

            double* v; // in case of addcol
            int64_t* vidx; // in case of sparse addcol

            // vL[r] contains the position of the first triplet of L corresponding to row r
            veci vL;

            options opt;
    };

    inline void eigenlusol::addCol(const mat& A, const int j)
    {
        int64_t* mode = new int64_t[1]; mode[0] = 1;
        double* diag = new double[1];
        double* vnorm = new double[1];

        n[0] += 1;

        for (int i = 0; i < min(*m, *n); i++)
        {
            v[i] = 0;
            locc[i] = 0;
        }
        for (int i = min(*m,*n); i < max(*m,*n); i++) locc[i] = 0;
        for (mat::InnerIterator it(A,j); it; ++it)
        {
            v[it.row()] = it.value();
        }

        clu8adc(mode, m, n, v, w, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, inform, diag, vnorm); 
        assignInfo();
        if (opt.verbose >= MAT) cout << "clu8adc finished inform " << inform[0] << " diag " << diag[0] << " vnorm " << vnorm[0] << endl;
    }

    inline int eigenlusol::repCol(const mat& A, const int j, const int _jrep, const int _mode1, const int _mode2)
    {
        int64_t* jrep = new int64_t[1]; jrep[0] = (int64_t)_jrep + 1; // +1: lusol convention
        int64_t* mode1 = new int64_t[1]; mode1[0] = (int64_t)_mode1; // 1: old column is assumed to be non-zero
        int64_t* mode2 = new int64_t[1]; mode2[0] = (int64_t)_mode2; // if 0: new column is assumed to be zero, 1: the new column is assumed to be non-zero, 3: the new column is assumed to be linear dependent and will not be permuted
        double* diag = new double[1];
        double* vnorm = new double[1];
        // minimum entry of L that will affect v during bartels update Lv = vnew
        int64_t* lmin = new int64_t[1]; lmin[0] = luparm[22]+1; // lenL0

        for (int i = 0; i < max(*m, *n); i++)
        {
            // if (_mode2 > 0)
            v[i] = 0;
            locc[i] = 0;
        }
        // for (int i = min(*m,*n); i < max(*m,*n); i++)
        // {
        //     if (_mode2 > 0) v[i] = 0;
        //     locc[i] = 0;
        // }
        if (_mode2 > 0) 
        { 
            // cout << "(A,j " << j << ")\n" << A.col(j).transpose() << endl;
            // check whether there are nnz's in the column
            // lusol has numerical difficulties (lu7rnk fatal error) if the input row is zero
            // if (A.outerIndexPtr()[j+1] - A.outerIndexPtr()[j] <= 0)
            //     mode2[0] = 0; // this is not necessarily reliable if A is not pruned
            // else
            {
                // double colNorm = 0;
                for (mat::InnerIterator it(A,j); it; ++it) 
                {
                    // if (abs(it.value()) > 1e-5) // pruning for numerical stability of decomposition?
                    {
                        v[it.row()] = it.value();
                        if (vL[it.row()] >= 0 && vL[it.row()] < lmin[0])
                            lmin[0] = vL[it.row()];
                        // colNorm += abs(it.value());
                    }
                }
                // if (colNorm < 1e-12)
                // if (colNorm < 1e-31)
                //     mode2[0] = 0; // the new column is assumed to be zero
            }
            // cout << "eigenlusol::repcol:vL\n" << vL.transpose() << endl;
            // cout << "\nLMIN " << lmin[0] << endl;
        }
        else
        {
            // cout << "(A,_jrep " << _jrep << ")\n" << A.col(_jrep).transpose() << endl;
            for (mat::InnerIterator it(A,_jrep); it; ++it) 
            {
                // if (abs(it.value()) > 1e-5) // pruning for numerical stability of decomposition?
                {
                    v[it.row()] = it.value();
                    // colNorm += abs(it.value());
                }
            }
        }
        // else remove column

        if (opt.verbose >= CONV) cout << "eigenlusol::repCol: mode1 " << *mode1 << " mode2 " << *mode2 << endl;
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        int lenL0   = luparm[22];
        try
        {
            clu8rpcsparse(mode1, mode2, m, n, jrep, v, w, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, inform, diag, vnorm, lmin); 
            // clu8rpc(mode1, mode2, m, n, jrep, v, w, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, inform, diag, vnorm);  // FIXME: deprecated for mode3
        }
        catch(...)
        {
            cout << "clu8rpc failed" << endl;
        }
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::repCol: clu8rpc")); }
#endif

        assignInfo();

        // cout << "0 numL0_0 " << numL00 << " lenl0_0 " << lenL00 << " lenL_0 " << lenL0 << " === 1 numL0_1 " << numL01 << " lenl0_1 " << lenL01 << " lenL_1 " << lenL1 << endl;
        // cout << "affected columns (rows?) during last clu8rpc:\n";
        int lenL1   = luparm[22];
        int l = lena[0] - lenL0;
        for (int ldummy = 0; ldummy < lenL1 - lenL0; ldummy++)
        {
            l--;
            // cout << "l " << l << ", lenL0 " << lenL0 << " ldummy " << ldummy << ", a " << a[l] << "; ";
            // the new entry is necessarily greater than
            if (vL[indr[l]-1] == -1)
                vL[indr[l]-1] = lenL0 + ldummy;
        }
        // cout << endl;
        // cout << "eigenlusol::repcol:vL\n" << vL.transpose() << endl;

        if (opt.verbose >= CONV) cout << "clu8rpc finished inform " << inform[0] << " diag " << diag[0] << " vnorm " << vnorm[0] << endl;
        if (opt.verbose >= CONV) cout << "info rank " << info->nrank << " info->lenU " << info->lenU << " info->nsing " << info->nsing << endl;
        if (opt.verbose >= VAR && inform[0] == -1) cout << "eigenlusol::repCol: clu8rpc: inform = -1  if the rank of U decreased by 1." << endl;
        if (opt.verbose >= VAR && inform[0] == 0) cout << "eigenlusol::repCol: clu8rpc: inform =  0  if the rank of U stayed the same." << endl;
        if (opt.verbose >= VAR && inform[0] == 1) cout << "eigenlusol::repCol: clu8rpc: inform =  1  if the rank of U increased by 1." << endl;
        if (opt.verbose >= VAR && inform[0] == 2) cout << "eigenlusol::repCol: clu8rpc: inform =  2  if the update seemed to be unstable (diag much bigger than vnorm). " << endl;
        if (opt.verbose >= NONE && inform[0] == 7) cout << "eigenlusol::repCol: clu8rpc: inform =  7  if the update was not completed (lack of storage)." << endl;
        if (opt.verbose >= NONE && inform[0] == 8) cout << "eigenlusol::repCol: clu8rpc: inform =  8  if jrep is not between 1 and n." << endl;
        if (opt.verbose >= NONE && inform[0] == 9) cout << "eigenlusol::repCol: clu8rpc: inform =  9  fatal error in lu7rnk" << endl;

        return inform[0];
    }
} // namespace cpplusol
#endif // _EIGENLUSOL_
