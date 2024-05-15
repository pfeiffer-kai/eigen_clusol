// This file is part of EIGENLUSOL.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "eigenlusol.h"

namespace cpplusol {

    void eigenlusol::reset()
    {
#if TIMEMEASUREMENTS
        cpplusol::timer t0;
#endif
        info = NULL;

        _P = NULL;
        _L = NULL;
        _U = NULL;
        _Q = NULL;
        _Z = NULL;

        _invLP = NULL;
        _QinvUinvLP = NULL;

        // this is slow
        // probably not necessary (?)
        // probably not everything has to be set to zero
        // this fixed rank problems in updates (repCol)
        // memset(a, 0, *lena*sizeof(double));
        // memset(indc, 0, *lena*sizeof(int64_t));
        // memset(indr, 0, *lena*sizeof(int64_t));
        // memset(p, 0, maxmn*sizeof(int64_t));
        // memset(lenr, 0, maxmn*sizeof(int64_t));
        // memset(locr, 0, maxmn*sizeof(int64_t));
        // memset(iqloc, 0, maxmn*sizeof(int64_t));
        // memset(ipinv, 0, maxmn*sizeof(int64_t));
        // // // vectors of length n
        // memset(w, 0, sizeof(double) * maxmn);
        // memset(q, 0, sizeof(int64_t) * maxmn);
        // memset(lenc, 0, sizeof(int64_t) * maxmn);
        // memset(locc, 0, sizeof(int64_t) * maxmn);
        // memset(iploc, 0, sizeof(int64_t) * maxmn);
        // memset(iqinv, 0, sizeof(int64_t) * maxmn);

        // // FIXME: is this necessary?
        // memset(v, 0, sizeof(double) * maxmn);

        // memset(vidx, 0, sizeof(double) * maxmn);
#if TIMEMEASUREMENTS
        { t0.stopTime(); times.push_back(cpplusol::time(t0.time, "eigenlusol::reset")); }
#endif
    }

    void eigenlusol::factorize(const mat& A, bool construct)
    {
        if (A.rows() * A.cols() > 0.1 * opt.maxmn)
            cout << "eigenlusol::factorize: WARNING: matrix size " << A.rows() << " x " << A.cols() << " (" << A.rows() * A.cols() << ") may be too large for current memory allocation of " << opt.maxmn << endl;
        // initialize solver data
#if TIMEMEASUREMENTS
        cpplusol::timer t0;
#endif
        reset();
        init(A);
#if TIMEMEASUREMENTS
        { t0.stopTime(); times.push_back(cpplusol::time(t0.time, "eigenlusol::factorize: init(A)")); }
#endif
        vL.head(*m).setConstant(-1);

        if (opt.verbose >= MAT) printPb();

        // call libclusol and do factorization
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        try {
        int64_t* n1 = new int64_t[1]; n1[0] = -1;
        int64_t* n2 = new int64_t[1]; n2[0] = -1;
        int64_t* np = new int64_t[1]; np[0] = 0;
            clu1fac( m, n, nelem, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, iploc, iqloc, ipinv, iqinv, w, inform, n1,n2,np); 
        }
        catch (...) {
            cout << "clu1fac failed" << endl;
        }
#if TIMEMEASUREMENTS
        t1.stopTime("eigenlusol::factorize");
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: clu1fac")); }
#endif
        int l      = lena[0];
        // cout << "affected columns during clu1fac, L0:\n";
        for (int ldummy = 0; ldummy < luparm[20]; ldummy++)
        {
            l--;
            // cout << "[" << l << ", " << indr[l] << ", " << a[l] << "]; " << endl;;
            vL[indr[l]-1] = 0;
        }
        // cout << "eigenlusol::factorize: initial vL " << vL.transpose() << endl;


#if TIMEMEASUREMENTS
        cpplusol::timer t2;
#endif
        assignInfo();
        // FIXME: why is the rank not propoerly assigned in clu1fac?
        info->nrank = *n - info->nsing;
#if TIMEMEASUREMENTS
        // t2.stopTime("assignInfo");
#endif

        if (opt.verbose >= MAT) { printPb(); print("EIGENLUSOL::clu1fac status: ",inform,1); }
        if (opt.verbose >= CONV && inform[0] == 0)  cout << "eigenlusol::factorize: clu1fac: the LU factors were obtained successfully." << endl;
        if (opt.verbose >= CONV && inform[0] == 1)  cout << "eigenlusol::factorize: clu1fac: U appears to be singular (nsing: " << info->nsing << ", as judged by lu6chk)." << endl;
        if (opt.verbose >= NONE && inform[0] == 3)  cout << "eigenlusol::factorize: clu1fac: some index pair indc(l), indr(l) lies outside the matrix dimensions 1:m , 1:n." << endl;
        if (opt.verbose >= NONE && inform[0] == 4)  cout << "eigenlusol::factorize: clu1fac: some index pair indc(l), indr(l) duplicates another such pair." << endl;
        if (opt.verbose >= NONE && inform[0] == 7)  cout << "eigenlusol::factorize: clu1fac: the arrays a, indc, indr were not large enough.  Their length lena should be increase to at least the value minlen given in luparm(13) (" << luparm[12] << ")." << endl;
        if (opt.verbose >= NONE && inform[0] == 8)  cout << "eigenlusol::factorize: clu1fac: there was some other fatal error.  (Shouldn't happen!)" << endl;
        if (opt.verbose >= NONE && inform[0] == 9)  cout << "eigenlusol::factorize: clu1fac: no diagonal pivot could be found with TSP or TDP.  The matrix must not be sufficiently definite or quasi-definite." << endl;
        if (opt.verbose >= NONE && inform[0] == 10) cout << "eigenlusol::factorize: clu1fac: there was some other fatal error." << endl;


#if TIMEMEASUREMENTS
        cpplusol::timer t3;
#endif
        if (construct)
        {
            constructL0(true);
            constructU(true);
        }
        _P0 = P();
        _L0 = L();
        _U0 = U();
        _Q0 = Q();
        _rank0 = info->nrank;
#if TIMEMEASUREMENTS
        // t3.stopTime("construct");
#endif

    }

    void eigenlusol::factorize_partial(const mat& A, const int j, const int rg, bool construct)
    {
        // initialize solver data
#if TIMEMEASUREMENTS
        cpplusol::timer t0;
#endif
        *m = A.rows();
        *n = A.cols();
        int64_t* n1 = new int64_t[1]; n1[0] = -1;
        int64_t* n2 = new int64_t[1]; n2[0] = -1;
        int64_t* np = new int64_t[1]; np[0] = *n;
        vL.head(*m).setConstant(-1);
        int ctrnnz = 0;
        for (int c = j; c < j + rg; c++)
        {
            for (mat::InnerIterator it(A,c); it; ++it)
            {
                a[ctrnnz] = it.value();
                indc[ctrnnz] = it.row() + 1;   // row index
                indr[ctrnnz] = it.col() + 1;   // col index (here it is equal to k)
                ctrnnz++;
            }
        }
        *nelem = ctrnnz;
#if TIMEMEASUREMENTS
        { t0.stopTime(); times.push_back(cpplusol::time(t0.time, "eigenlusol::factorize_partial: init(A)")); }
#endif

        if (opt.verbose >= MAT) printPb();

        // call libclusol and do factorization
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        try {
            clu1fac( m, n, nelem, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, iploc, iqloc, ipinv, iqinv, w, inform, n1,n2,np); 
        }
        catch (...) {
            cout << "clu1fac failed" << endl;
        }
#if TIMEMEASUREMENTS
        t1.stopTime("eigenlusol::factorize_partial");
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize_partial: clu1fac")); }
#endif

        if (opt.verbose >= MAT) { printPb(); print("EIGENLUSOL::clu1fac status: ",inform,1); }

#if TIMEMEASUREMENTS
        cpplusol::timer t2;
#endif
        assignInfo();
        // FIXME: why is the rank not propoerly assigned in clu1fac?
        info->nrank = *n - info->nsing;
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize_partial: assignInfo")); }
#endif

        int l      = lena[0];
        // cout << "affected columns during clu1fac, L0:\n";
        for (int ldummy = 0; ldummy < luparm[20]; ldummy++)
        {
            l = l - 1;
            // cout << "[" << l << ", " << i << ", " << j << ", " << a[l] << "]; ";
            vL[indr[l]-1] = 0;
        }
        // cout << endl;

#if TIMEMEASUREMENTS
        cpplusol::timer t3;
#endif
        if (construct)
        {
            constructL0(true);
            constructU(true);
        }
#if TIMEMEASUREMENTS
        // t3.stopTime("construct");
#endif

    }

    int eigenlusol::factorize_bin(const mat& A, const veci& cols, int& nrRows, bool construct, int n1_, int n2_, int np_)
    {
        // initialize solver data
#if TIMEMEASUREMENTS
        cpplusol::timer t0;
#endif
        *m = A.rows();
        *n = A.cols();
        int64_t* n1 = new int64_t[1]; n1[0] = 0; if (n1_ > -1) n1[0] = n1_;
        int64_t* n2 = new int64_t[1]; n2[0] = *n; if (n2_ > -1) n2[0] = n2_;
        int64_t* np = new int64_t[1]; np[0] = *n; if (np_ > -1) np[0] = np_;
        vL.head(*m).setConstant(-1);
        // mat A_bin = mat(*m,*n);
        int nrCols = 0;
        int ctrnnz = 0;
        int row0 = 1e153;
        int rowend = -1;
        for (int c = 0; c < A.cols(); c++)
        {
            if (cols[c] == 0)
            {
                for (mat::InnerIterator it(A,c); it; ++it)
                {
                    if (it.row() > rowend) rowend = it.row();
                    if (it.row() < row0) row0 = it.row();
                    a[ctrnnz] = it.value();
                    indc[ctrnnz] = it.row() + 1;   // row index
                    indr[ctrnnz] = it.col() + 1;   // col index (here it is equal to k)
                    // A_bin.coeffRef(it.row(), it.col()) = it.value();
                    ctrnnz++;
                }
                nrCols++;
            }
        }
        *nelem = ctrnnz;
        nrRows = rowend - row0 + 1;

#if TIMEMEASUREMENTS
        { t0.stopTime(); times.push_back(cpplusol::time(t0.time, "eigenlusol::factorize: init(A)")); }
#endif

        if (opt.verbose >= MAT) printPb();

        // call libclusol and do factorization
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        clu1fac( m, n, nelem, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, iploc, iqloc, ipinv, iqinv, w, inform, n1,n2,np); 
#if TIMEMEASUREMENTS
        t1.stopTime("eigenlusol::factorize_bin");
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: clu1fac")); }
#endif

        if (opt.verbose >= MAT) { printPb(); print("EIGENLUSOL::clu1fac status: ",inform,1); }

#if TIMEMEASUREMENTS
        cpplusol::timer t2;
#endif
        assignInfo();
        // FIXME: why is the rank not propoerly assigned in clu1fac?
        info->nrank = *n - info->nsing;
#if TIMEMEASUREMENTS
        // t2.stopTime("assignInfo");
#endif

        int l      = lena[0];
        // cout << "affected columns during clu1fac, L0:\n";
        for (int ldummy = 0; ldummy < luparm[20]; ldummy++)
        {
            l = l - 1;
            // cout << "[" << l << ", " << i << ", " << j << ", " << a[l] << "]; ";
            vL[indr[l]-1] = 0;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t3;
#endif
        if (construct)
        {
            constructL0(true);
            constructU(true);
        }
        // cout << "A_bin\n" << (matd)A_bin << endl;
        // cout << "U\n" << (matd)U() << endl;
        // cout << "A - lu " << (A_bin - P().transpose() * L() * U() * Q().transpose()).norm() << endl;
#if TIMEMEASUREMENTS
        // t3.stopTime("construct");
#endif

        return nrCols;
    }

    void eigenlusol::computeNS(const mat& A, shared_ptr<mat> Zio, bool construct)
    {
        reset();
        if (info)
        {
            cout << "EIGENLUSOL::COMPUTENS::WARNING: this instance of eigenlusol was already used to compute a factorization; consider using a new instance of eigenlusol in order to prevent overwriting" << endl;
        }

        // initialize solver data
        // factorize(A.transpose(), false);
        // constructL0(true);
        factorize(A, false);
        constructU(true);

        if (construct)
        {
            if (Zio) constructZ(Zio);
            else constructZ();
        }
    }

    mat eigenlusol::applyNSOnTheRight(const mat& B)
    {
        if (!info)
        {
            cout << "Factorization not existing, call computeNS() first" << endl;
            throw;
        }
        if (opt.nstype == 0)
        {
            // applies the NS of A (computed in computeNS) on the right of a given matrix B
            // B * Z = B * P' * inv(L') * [0; I]
            // note that (B * P' * inv(L'))' = inv(L) * P * B'
            // _P->transpose().applyThisOnTheRight(B);
            if (opt.computeOnce)
            {
                if (!_invLP)
                {
                    computeInvLP();
                }
    
                // apply inv(L') on the right of B
                // cout << "L\n" << *_L << endl;
                // cout << "BT\n" << BT << endl;
    #if TIMEMEASUREMENTS
                cpplusol::timer t1 = cpplusol::timer();
    #endif
                // _invLP->transpose().applyThisOnTheRight(B);
                mat out;
                // cout << "B\n" << (MatrixXd)B << endl;
                // cout << "_invLP->bottomRows(*m - info->nrank).transpose()\n" << (MatrixXd)_invLP->bottomRows(*m - info->nrank).transpose() << endl;
                if (*m - info->nrank > 0)
                    out = B * _invLP->bottomRows(*m - info->nrank).transpose();
                else
                {
                    out = mat(B.rows(), 0);
                }
    #if TIMEMEASUREMENTS
                { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "lusol:applyNSontheRight:invLBT")); }
    #endif
                return out;
            }
            else
            {
                mat PBT = *_P * B.transpose();
                _L->triangularView<Eigen::Lower>().solveInPlace(PBT);
                return PBT.transpose().rightCols(*m - info->nrank);
            }
        }
        else if (opt.nstype == 1)
        {
            // // applies the NS of A (computed in computeNS) on the right of a given matrix B
            // // B * Z = B * Q [-U1^-1 U2 \\ I]
            // // cout << "U\n" << (matd)lu->U() << endl;
            // mat U1 = U().block(0,0,info->nrank,info->nrank);
            // // cout << "U1\n" << (matd)U1 << endl;
            // mat U2 = U().block(0,info->nrank,info->nrank,*n-info->nrank);
            // // cout << "U2\n" << (matd)U2 << endl;
            // mat I(*n-info->nrank, *n-info->nrank); I.setIdentity();
            // mat Zhard = mat(*n, *n-info->nrank);
            // // th.getTriplets(U2);
            // // th.getTriplets(I, lu->rank(), 0);
            // // th.setFromTriplets(Zhard);
            // th.reset();
            // U1.triangularView<Eigen::Upper>().solveInPlace(U2);
            // // cout << "invU1U2\n" << (matd)U2 << endl;
            // th.getTriplets(-U2);
            // th.getTriplets(I, info->nrank, 0);
            // Zhard.setZero();
            // th.setFromTriplets(Zhard);
            // Q().applyThisOnTheLeft(Zhard);
            // // cout << "Zhard\n" << (matd)Zhard << endl;
            // cout << "NS test1 " << (U()*Q().transpose() * Zhard).norm() << endl;
            // if (opt.verbose >= MAT)
            // {
            //     cout << "Q\n" << (MatrixXd)Q() << endl;
            //     cout << "Z: Q  [U2//I]\n" << matd(Q() * Zhard) << endl;
            //     cout << "U1\n" << (matd)U1 << endl;
            //     cout << "invU1 U2\n" << (matd)U2 << endl;
            //     th.reset();
            //     cout << "Z: Q  invU1 [U2//I]\n" << matd(Q() * Zhard) << endl;
            // }
            // th.reset();
             
            // return B * Zhard;
            return B * Z();
        }
    }

    void eigenlusol::applyNSOnTheRight(mat& B, vector<mat>& _storage)
    {
        if (!info)
        {
            cout << "Factorization not existing, call computeNS() first" << endl;
            throw;
        }
        // applies the NS of A (computed in computeNS) on the right of a given matrix B
        // B * Z = B * P' * inv(L') * [0; I]
        // note that (B * P' * inv(L'))' = inv(L) * P * B'
        if (opt.computeOnce)
        {
            if (!_invLP)
            {
                computeInvLP();
            }

#if TIMEMEASUREMENTS
            cpplusol::timer t1 = cpplusol::timer();
#endif
            // _invLP->transpose().applyThisOnTheRight(B);
            _storage.push_back(B * Z());
#if TIMEMEASUREMENTS
            { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "lusol:applyNSontheRight:invLBT")); }
#endif
        }
        else
        {
            // Eigen::SparseMatrix<double,Eigen::RowMajor>::Map PT(_P->cols(), _P->rows(), _P->nonZeros(), _P->outerIndexPtr(), _P->innerIndexPtr(), _P->valuePtr());
            // PT.applyThisOnTheRight(B);
            // Eigen::SparseMatrix<double,Eigen::RowMajor>::Map BT(B.cols(), B.rows(), B.nonZeros(), B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr());
            // *_P->applyThisOnTheLeft(BT);

#if TIMEMEASUREMENTS
            cpplusol::timer t1 = cpplusol::timer();
#endif
            mat PBT = *_P * B.transpose();
#if TIMEMEASUREMENTS
            { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "lusol:applyNSontheRight:PBT")); }
            t1 = cpplusol::timer();
#endif
            _L->triangularView<Eigen::Lower>().solveInPlace(PBT);
#if TIMEMEASUREMENTS
            { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "lusol:applyNSontheRight:solve")); }
            t1 = cpplusol::timer();
#endif
            _storage.push_back(PBT.transpose().rightCols(*m - info->nrank));
#if TIMEMEASUREMENTS
            { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "lusol:applyNSontheRight:pushback")); }
            t1 = cpplusol::timer();
#endif

        }
    }

    mat eigenlusol::applyNSTOnTheLeft(mat& B)
    {
        if (!info)
        {
            cout << "Factorization not existing, call computeNS() first" << endl;
            throw;
        }

        // applies the NS of A (computed in computeNS) on the left of a given vector b
        // ZT * B = inv(L) * P * B
        // note that (B * P' * inv(L'))' = inv(L) * P * B'

        if (opt.computeOnce)
        {
            if (!_invLP)
            {
                computeInvLP();
            }

            _invLP->applyThisOnTheLeft(B);
        }
        else
        {
            // Eigen::SparseMatrix<double,Eigen::RowMajor>::Map LT(_L->cols(), _L->rows(), _L->nonZeros(), _L->outerIndexPtr(), _L->innerIndexPtr(), _L->valuePtr());
            // LT.triangularView<Eigen::Upper>().solveInPlace(B);
            // _P->transpose().applyThisOnTheLeft(B);
            _P->applyThisOnTheLeft(B);
            L().triangularView<Eigen::Lower>().solveInPlace(B);
            return B.bottomRows(*m - info->nrank);
        }

        // // lusol does not provide a sparse solve
        // // the returned solution v will be dense
        // for i < B.rows()
        //     v = B.row(i)
        //     clu6sol( m, n, v, w, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, inform); 
    }

    vec eigenlusol::applyNSOnTheLeft(const vec& b)
    {
        if (!info)
        {
            cout << "Factorization not existing, call computeNS() first" << endl;
            throw;
        }

        if (opt.nstype == 0)
        {
            // applies the NS of A (computed in computeNS) on the left of a given vector b
            // Z * b = P' * inv(L') * [0; I] * b
            // note that (B * P' * inv(L'))' = inv(L) * P * B'

            if (opt.computeOnce)
            {
                if (!_invLP)
                {
                    computeInvLP();
                }

                return _invLP->transpose() * b;
            }
            else
            {
                Eigen::SparseMatrix<double,Eigen::RowMajor>::Map LT(_L->cols(), _L->rows(), _L->nonZeros(), _L->outerIndexPtr(), _L->innerIndexPtr(), _L->valuePtr());
                vec tmp = LT.triangularView<Eigen::Upper>().solve(b);
                return _P->transpose() * tmp;
            }

            // // lusol does not provide a sparse solve
            // // the returned solution v will be dense
            // for i < B.rows()
            //     v = B.row(i)
            //     clu6sol( m, n, v, w, lena, luparm, parmlu, a, indc, indr, p, q, lenc, lenr, locc, locr, inform); 
        }
        else if (opt.nstype == 1)
        {
            cout << "NIY" << endl;
            throw;
            // Z * b = Q [-U1^-1U2 \\ I] * b
            // cout << "U\n" << (matd)lu->U() << endl;
            mat U1 = U().block(0,0,info->nrank,info->nrank);
            // cout << "U1\n" << (matd)U1 << endl;
            mat U2 = U().block(0,info->nrank,info->nrank,*n-info->nrank);
            // cout << "U2\n" << (matd)U2 << endl;
            mat I(*n-info->nrank, *n-info->nrank); I.setIdentity();
            mat Zhard = mat(*n, *n-info->nrank);
            // th.getTriplets(U2);
            // th.getTriplets(I, lu->rank(), 0);
            // th.setFromTriplets(Zhard);
            th.reset();
            U1.triangularView<Eigen::Upper>().solveInPlace(U2);
            // cout << "invU1U2\n" << (matd)U2 << endl;
            th.getTriplets(-U2);
            th.getTriplets(I, info->nrank, 0);
            Zhard.setZero();
            th.setFromTriplets(Zhard);
            Q().applyThisOnTheLeft(Zhard);
            // cout << "Zhard\n" << (matd)Zhard << endl;
            cout << "NS test1 " << (U()*Q().transpose() * Zhard).norm() << endl;
            if (opt.verbose >= MAT)
            {
                cout << "Q\n" << (MatrixXd)Q() << endl;
                cout << "Z: Q  [U2//I]\n" << matd(Q() * Zhard) << endl;
                cout << "U1\n" << (matd)U1 << endl;
                cout << "invU1 U2\n" << (matd)U2 << endl;
                th.reset();
                cout << "Z: Q  invU1 [U2//I]\n" << matd(Q() * Zhard) << endl;
            }
            th.reset();
             
            // return Zhard * b;
            cout << "left z\n" << (matd)Z() << endl;
            return Zhard * b;
        }
    }

    void eigenlusol::solveInPlace(mat& B)
    {
        // solve A x = B
        // do Q invU invL P B
        cout << "eigenlusol::solveInPlace B NIY" << endl; 
        throw;
        _P->applyThisOnTheLeft(B); 
        _L->triangularView<Eigen::Lower>().solveInPlace(B);
        _U->triangularView<Eigen::Upper>().solveInPlace(B);
        _Q->applyThisOnTheLeft(B);
    }

    void eigenlusol::solveInPlace(vec& b)
    {
        // // solve A x = P' L U Q' x = b
        // if (!_QinvUinvLP)
        // {
        //     _QinvUinvLP = new mat;
        //     _QinvUinvLP = _P;
        //     _L->triangularView<Eigen::Lower>().solveInPlace(*_P);
        //     mat tmp = _P->topRows(info->nrank);
        //     U().topLeftCorner(info->nrank,info->nrank).triangularView<Eigen::Upper>().solveInPlace(tmp); // cant pass segment here, eigen complains, need tmp's
        //     mat tmp2 = mat(*n, tmp.cols());
        //     tmp2.topRows(info->nrank) = tmp;
        //     Q().applyThisOnTheLeft(tmp2);
        // }

        P().applyThisOnTheLeft(b); 
        L().triangularView<Eigen::Lower>().solveInPlace(b);
        vec tmp = b.head(info->nrank);
        U().topLeftCorner(info->nrank,info->nrank).triangularView<Eigen::Upper>().solveInPlace(tmp); // cant pass segment here, eigen complains, need tmp's
        vec tmp2 = vec::Zero(*n);
        tmp2.head(info->nrank) = tmp;
        Q().applyThisOnTheLeft(tmp2);
        b = tmp2;
    }

    vec eigenlusol::solveInPlaceT(const vec& b)
    {
        // solve least squares
        // solve ATA x = ATb
        // we have the decomposition of AT from computeNS
        // ATb is on input if normalF is chosen
        // we then have
        // x = (ATA)^-1 ATb
        // x = (PT L U QT Q UT LT P)^-1 ATb
        // x = PT invLT invUT invU invL P ATb 

        // cout << "_A\n" << (MatrixXd)_A << endl;
        // cout << "PTLUQT\n" << (MatrixXd)P().transpose() * L() * U() * Q().transpose() << endl;
        // cout << "PTLUQTT\n" << (MatrixXd)(P().transpose() * L() * U() * Q().transpose()).transpose() << endl;
        // cout << "P().transpose()\n" << (MatrixXd)P().transpose() << endl;
        // cout << "L\n" << (MatrixXd)L() << endl;
        // cout << "U\n" << (MatrixXd)U() << endl;
        // cout << "Q().transpose()\n" << (MatrixXd)Q().transpose() << endl;
        // cout << "b " << b.transpose() << endl;
        vec tmp0 = P() * b;
        // cout << "tmp0 " << tmp0.transpose() << endl;
        L().triangularView<Eigen::Lower>().solveInPlace(tmp0);
        // cout << "tmp0 " << tmp0.transpose() << endl;
        vec tmp = tmp0.head(info->nrank);
        // cout << "tmp " << tmp.transpose() << endl;
        // cout << "info->nrank " << info->nrank << endl;
        // cout << "U().topLeftCorner(info->nrank,info->nrank)\n" << (MatrixXd)U().topLeftCorner(info->nrank,info->nrank) << endl;
        U().topLeftCorner(info->nrank,info->nrank).triangularView<Eigen::Upper>().solveInPlace(tmp); // cant pass segment here, eigen complains, need tmp's
        // cout << "tmp " << tmp.transpose() << endl;
        U().topLeftCorner(info->nrank,info->nrank).triangularView<Eigen::Lower>().transpose().solveInPlace(tmp);
        // cout << "tmp " << tmp.transpose() << endl;
        tmp0.head(info->nrank) = tmp.head(info->nrank);
        // cout << "tmp0 " << tmp0.transpose() << endl;
        L().triangularView<Eigen::Lower>().transpose().solveInPlace(tmp0); // cant pass segment here, eigen complains, need tmp's
        // cout << "tmp0 " << tmp0.transpose() << endl;
        P().transpose().applyThisOnTheLeft(tmp0);
        // cout << "tmp0 " << tmp0.transpose() << endl;
        return tmp0;
    }

    vec eigenlusol::solveInPlaceTT(const vec& b)
    {
        // solves with the transpose of the existing decomposition
        // A = P^T L [U1 U2 \\ 0 0] Q^T
        // with A^T = Q [U1^T 0 \\ U2^T 0] L^T P
        // such that
        // A^-T = P^T L^-T [U1^-T b_1 \\ 0] Q^T
        // This only U2 is linear dependent of U1 so can be neglected for basic solution?
        cout << "A\n" << (matd)_A << endl;
        cout << "b " << b.transpose() << endl;
        cout << "Q0 " << Q0().indices().transpose() << endl;
        vec tmp0 = Q0().transpose() * b;
        cout << "tmp0 " << tmp0.transpose() << endl;
        cout << "rank0 " << rank0() << endl;
        vec tmp = -tmp0.head(rank0());
        cout << "tmp " << tmp.transpose() << endl;
        U0().topLeftCorner(rank0(),rank0()).triangularView<Eigen::Upper>().solveInPlace(tmp); // cant pass segment here, eigen complains, need tmp's
        cout << "tmp " << tmp.transpose() << endl;
        cout << "L0()\n" << (matd)L0() << endl;
        vec tmp1 = vec::Zero(L0().rows());
        tmp1.head(rank0()) = tmp;
        cout << "tmp1 " << tmp1.transpose() << endl;
        L0().triangularView<Eigen::Lower>().transpose().solveInPlace(tmp1); // cant pass segment here, eigen complains, need tmp's
        cout << "tmp1 " << tmp1.transpose() << endl;
        P0().transpose().applyThisOnTheLeft(tmp1);
        cout << "tmp1 " << tmp1.transpose() << endl;

        return tmp1;
    }

    void eigenlusol::constructL0(bool permute)
    {
        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        _L = make_unique<mat>(*m, *m);

        for (int i = 0; i < info->lenL0; i++)
        {
            th.tripletL()[i] = triplet(indc[*lena-info->lenL0+i] - 1, indr[*lena-info->lenL0+i] - 1, -a[*lena-info->lenL0+i]);
        }
        // add 1's along the diagonal
        for (int i = 0; i<*m; i++)
        {
            th.tripletL()[info->lenL0 + i] = triplet(i, i, 1.);
        }
        _L->setFromTriplets(th.tripletL().begin(), th.tripletL().begin() + info->lenL0 + *m);

        _L->makeCompressed();


        // for triangular factors
        if (permute)
        {
            P().applyThisOnTheLeft(*_L);
            _P->transpose().applyThisOnTheRight(*_L);
        }

#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructL0")); }
#endif

    }

    bool eigenlusol::constructU(bool permute)
    {
        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        // construct U
        // initialize arrays for U triplets
        // read the triplet form from LUSOL data
        // print("ui",ui,info->lenU); //-info->nsing);
        // print("uj",uj,info->lenU); //-info->nsing);
        // print("ua",ua,info->lenU); //-info->nsing);
        //
        // the behaviour of lenU (nnz in U) can be erratic at times (for example < 0), especially in the update case, but we don't seem to need it (?)

        _U = make_unique<mat>(*m, *n);

        th.reset();
        for (int i = 0; i<info->nrank; i++)
        {
            // get row index
            int piv = (int)p[i];
            // get length of row
            int len = int(lenr[(int)piv-1]);
            // get location of row
            int loc = int(locr[(int)piv-1]);
            // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
            // load data into triplet arrays
            for (int j = 0; j < len; j++)
            {
                th.setTriplet(piv-1, indr[loc-1+j]-1, a[loc-1+j]);
            }
            // increment row start pointer
        }
        th.setFromTriplets(*_U);

        // for triangular factors
        if (permute)
        {
            P().applyThisOnTheLeft(*_U);
            // cout << "Q\n" << (matd)Q() << endl;
            // cout << "U\n" << (matd)*_U << endl;
            Q().applyThisOnTheRight(*_U);
        }

        _U->makeCompressed();
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructU")); }
#endif

        // cout << "U\n" << *_U << endl;
        return true;
    }
    
    void eigenlusol::updateInvpq()
    {
        for (int i = 0; i < *m; i++) ipinv[p[i]-1] = i + 1;
        // cout << "j q qinv: \n";
        for (int j = 0; j < *n; j++)
        {
            iqinv[q[j]-1] = j + 1;
            // cout << j << " " << q[j] << " " << iqinv[q[j]-1] << endl;;
        }
        // cout << endl;
    }

    void eigenlusol::getv(vec& u)
    {
        // assumes mode2=3 and v is linearly dependent
        for (int j = 0; j < min(*m,*n); j++)
        {
            u(j) = -v[p[j]-1];
        }
    }

    void eigenlusol::constructu_partial(vec& u)
    {
        // assumes mode2=3 and v is linearly dependent
        for (int j = 0; j < rank(); j++)
        {
            u(j) = -v[p[j]-1];
        }
    }

    bool eigenlusol::constructU_partial(bool permute, int uidx, vec* u)
    {

        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        // construct U
        // initialize arrays for U triplets
        // the behaviour of lenU (nnz in U) can be erratic at times (for example < 0), especially in the update case, but we don't seem to need it (?)
        // cout << "Upartial uidx " << uidx << " q[uidx] - 1 " << q[uidx] - 1 << endl;

        // first, update ipinv and iqinv in order to avoid repeated loops over p and q
        // we do this here in order to not have to touch the lusol fortran code
        updateInvpq();

        if (!_U) _U = make_unique<mat>(rank(), rank());
        else
        {
            _U->conservativeResize(rank(), rank());
            _U->setZero();
        }

        th.reset();
        int piv, len, loc;
        // go through the first currank rows
        if (opt.verbose >= CONV) cout << "eigenlusol::constructU_partial: Udiag\n";
        for (int i = 0; i<info->nrank; i++)
        {
            // get row index
            piv = (int)p[i];
            // get length of row
            len = int(lenr[(int)piv-1]);
            // get location of row
            loc = int(locr[(int)piv-1]);
            // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
            // load data into triplet arrays
            // permutation needs to be more efficient
            for (int j = 0; j < len; j++)
            {
                // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                if (abs(indr[loc-1+j]-1) > *n) 
                {
                    cout << "eigenlusol::constructU_partial: there is some error in indr, most likely because of clusol:lu8rpc:mode3; return false" << endl;
                    // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                    return false;
                }
                // cout << "ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1, a[loc-1+j] " << ipinv[piv-1]-1 << ", " << iqinv[indr[loc-1+j]-1]-1 << ", " << a[loc-1+j] << endl;
                if (uidx > -1 && iqinv[indr[loc-1+j]-1]-1 == uidx)
                {
                    // cout << "indr[loc-1+j] == q[uidx] " << indr[loc-1+j] << " " << q[uidx] << " iqinv[indr[loc-1+j]-1]-1,  uidx " << iqinv[indr[loc-1+j]-1]-1 << endl;
                    if (ipinv[piv-1]-1 < rank())
                    {
                        (*u)[ipinv[piv-1]-1] = -a[loc-1+j];
                    }
                    else
                    {
                        cout << "eigenlusol::constructU_partial: there is some error in ipinv, most likely because of clusol:lu8rpc:mode3; return false" << endl;
                        return false;
                    }
                }
                else
                {
                    // we directly construct PUQ
                    if (iqinv[indr[loc-1+j]-1]-1 < rank())
                    {
                        if (iqinv[indr[loc-1+j]-1]-1 == i)
                        {
                        //     // if (abs(a[loc-1+j]) < 1e-7)
                        //     //     a[loc-1+j] = 1e-7; // can also be achieved by Utol (?)
                            if (opt.verbose >= CONV) cout << abs(a[loc-1+j]) << " ";
                            if (ipinv[piv-1]-1 != iqinv[indr[loc-1+j]-1]-1)
                            {
                                cout << "eigenlusol::constructU_partial: diagonal element but not reflected in q and p indices" << endl;
                                cout << "ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1 " << ipinv[piv-1]-1 << " " << iqinv[indr[loc-1+j]-1]-1 << endl;
                                return false;
                            }
                        }
                        else
                        {
                            if (ipinv[piv-1]-1 == iqinv[indr[loc-1+j]-1]-1)
                            {
                                cout << "eigenlusol::constructU_partial: wrong diagonal element" << endl;
                                return false;
                            }
                        }
                        th.setTriplet(ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1, a[loc-1+j]);
                    }

                }
            }
        }
        if (opt.verbose >= CONV) cout << endl;

        th.setFromTriplets(*_U);

        _U->makeCompressed();

        // for triangular factors
        if (permute)
        {
            constructQ();
        }
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructU")); }
#endif

        // cout << "U\n" << *_U << endl;
        return true;
    }

    bool eigenlusol::constructU_partial(mat& U2, int n1, int n2, int np, veci& elimCols, veci& chosenColsP, veci& chosenColsPinv, int tid)
    {

        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        // construct U
        // initialize arrays for U triplets
        // the behaviour of lenU (nnz in U) can be erratic at times (for example < 0), especially in the update case, but we don't seem to need it (?)
        // cout << "Upartial uidx " << uidx << " q[uidx] - 1 " << q[uidx] - 1 << endl;

        // first, update ipinv and iqinv in order to avoid repeated loops over p and q
        // we do this here in order to not have to touch the lusol fortran code
        updateInvpq();

        if (!_U) _U = make_unique<mat>(rank(), rank());
        else
        {
            _U->conservativeResize(rank(), rank());
            _U->setZero();
        }

        th.reset();
        th2.reset();
        int piv, len, loc;
        // go through the first currank rows
        if (opt.verbose >= CONV) cout << "eigenlusol::constructU_partial: Udiag\n";
        chosenColsP.head(*n).setConstant(-1);
        chosenColsPinv.head(*n).setConstant(-1);
        int npctr = 0;
        for (int i = 0; i<info->nrank; i++)
        {
            // get row index
            int piv = (int)p[i];
            // get length of row
            int len = int(lenr[(int)piv-1]);
            // get location of row
            int loc = int(locr[(int)piv-1]);
            // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
            // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
            // load data into triplet arrays
            // permutation needs to be more efficient
            for (int j = 0; j < len; j++)
            {
                // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                if (abs(indr[loc-1+j]-1) > *n) 
                {
                    cout << "eigenlusol::constructU_partial: there is some error in indr, most likely because of clusol:lu8rpc:mode3; return false" << endl;
                    // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                    return false;
                }
                // cout << "ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1, a[loc-1+j] " << ipinv[piv-1]-1 << ", " << iqinv[indr[loc-1+j]-1]-1 << ", " << a[loc-1+j] << endl;
                // we directly construct PUQ
                if (iqinv[indr[loc-1+j]-1]-1 < rank())
                {
                    if (iqinv[indr[loc-1+j]-1]-1 == i)
                    {
                        //     // if (abs(a[loc-1+j]) < 1e-7)
                        //     //     a[loc-1+j] = 1e-7; // can also be achieved by Utol (?)
                        if (opt.verbose >= CONV) cout << abs(a[loc-1+j]) << " ";
                        if (ipinv[piv-1]-1 != iqinv[indr[loc-1+j]-1]-1)
                        {
                            cout << "eigenlusol::constructU_partial: diagonal element but not reflected in q and p indices" << endl;
                            cout << "ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1 " << ipinv[piv-1]-1 << " " << iqinv[indr[loc-1+j]-1]-1 << endl;
                            return false;
                        }
                    }
                    else
                    {
                        if (ipinv[piv-1]-1 == iqinv[indr[loc-1+j]-1]-1)
                        {
                            cout << "eigenlusol::constructU_partial: wrong diagonal element" << endl;
                            return false;
                        }
                    }
                    th.setTriplet(ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1, a[loc-1+j]);
                }
                else
                {
                    // assemble U2
                    // cout << " iqinv[indr[loc-1+j]-1]-1 > n1 && elimCols(iqinv[indr[loc-1+j]-1]-1) " << indr[loc-1+j]-1 << " " << n1 << " " << elimCols(indr[loc-1+j]-1) << endl;
                    // if (tid == 0) cout << "tid " << tid << " n1 <= indr[loc-1+j]-1 && indr[loc-1+j]-1 <= n2 " << n1 << " <= " << indr[loc-1+j]-1 << " <= " << n2 << " iqinv[indr[loc-1+j]-1]-1 " << iqinv[indr[loc-1+j]-1]-1 << " >= rank " << rank() << " a " << a[loc-1+j] << endl;
                    if (iqinv[indr[loc-1+j]-1]-1 >= rank() && n1 <= indr[loc-1+j]-1 && indr[loc-1+j]-1 <= n2 && elimCols(indr[loc-1+j]-1) == 0)
                    {
                        int colIdx;
                        if (chosenColsP[iqinv[indr[loc-1+j]-1]-1] == -1) npctr++;
                        // chosenCols[indr[loc-1+j]-1] = 1;
                        if (chosenColsP[iqinv[indr[loc-1+j]-1]-1] > -1)
                            colIdx = chosenColsP[iqinv[indr[loc-1+j]-1]-1];
                        else
                        {
                            colIdx = 0;
                            while (chosenColsPinv[colIdx] > -1)
                                colIdx++;
                            chosenColsPinv[colIdx] = iqinv[indr[loc-1+j]-1]-1;
                            chosenColsP[iqinv[indr[loc-1+j]-1]-1] = colIdx;
                        }
                        th2.setTriplet(ipinv[piv-1]-1, colIdx, a[loc-1+j]);
                    }
                }
            }
        }
        if (opt.verbose >= CONV) cout << endl;
        if (opt.verbose >= CONV && tid == 0 && np != npctr)
        {
            cout << "tid " << tid << " eigenlusol::constructU_partial U2: Error, detected " << npctr << " U2 vectors as compared to " << np << " desired ones" << endl;
            // th2.print();
            // throw;
        }

        th.setFromTriplets(*_U);
        th2.setFromTriplets(U2);

        _U->makeCompressed();
        U2.makeCompressed();

        // for triangular factors
        constructQ();
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructU")); }
#endif

        // cout << "U\n" << *_U << endl;
        return true;
    }

    void eigenlusol::constructP()
    {
        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        _P = make_unique<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> >(*m);
        for (int i = 0; i < *m; i++) 
        {
            _P->indices()[p[i] - 1] = i;
        }
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructP")); }
#endif
    }

    void eigenlusol::constructQ()
    {
        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        _Q = make_unique<Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> >(*n);
        for (int i = 0; i < *n; i++) _Q->indices()[i] = q[i] - 1;
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructQ")); }
#endif
    }

    double eigenlusol::getUcond()
    {

        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        // construct U
        // initialize arrays for U triplets
        // the behaviour of lenU (nnz in U) can be erratic at times (for example < 0), especially in the update case, but we don't seem to need it (?)
        // cout << "Upartial uidx " << uidx << " q[uidx] - 1 " << q[uidx] - 1 << endl;

        // first, update ipinv and iqinv in order to avoid repeated loops over p and q
        // we do this here in order to not have to touch the lusol fortran code
        updateInvpq();

        double maxVal = 0;
        double minVal = 1e153;
        int piv, len, loc;
        // go through the first currank rows
        if (opt.verbose >= CONV) cout << "eigenlusol::getUcond: UDiag with rank " << rank() << ":\n";
        int rankCheck = 0;
        for (int i = 0; i<info->nrank; i++)
        {
            // get row index
            piv = (int)p[i];
            // get length of row
            len = int(lenr[(int)piv-1]);
            // get location of row
            loc = int(locr[(int)piv-1]);
            // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
            // load data into triplet arrays
            // permutation needs to be more efficient
            for (int j = 0; j < len; j++)
            {
                // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                if (abs(indr[loc-1+j]-1) > *n) 
                {
                    if (opt.verbose >= CONV) cout << "eigenlusol::getUcond: there is some error in indr, most likely because of clusol:lu8rpc:mode3; ignore this entry" << endl;
                    // cout << "i rank, piv-1, indr[loc-1+j]-1, a[loc-1+j] " << i << " " << info->nrank << " " << piv-1 << " " << indr[loc-1+j]-1 << " " << a[loc-1+j] << endl;
                    continue;
                }
                else
                {
                    // cout << "ipinv[piv-1]-1, iqinv[indr[loc-1+j]-1]-1, a[loc-1+j] " << ipinv[piv-1]-1 << ", " << iqinv[indr[loc-1+j]-1]-1 << ", " << a[loc-1+j] << endl;
                    if (iqinv[indr[loc-1+j]-1]-1 == i)
                    {
                        if (abs(a[loc-1+j]) > maxVal) maxVal = abs(a[loc-1+j]);
                        if (abs(a[loc-1+j]) < minVal) minVal = abs(a[loc-1+j]);
                        if (abs(a[loc-1+j]) > opt.Utol1 * 1e-2) rankCheck++;
                        if (opt.verbose >= CONV) cout << a[loc-1+j] << " ";
                    }
                }
            }
        }
        if (opt.verbose >= CONV)
        {
            cout << endl;
            cout << "eigenlusol::getUcond: rankCheck " << rankCheck << endl;
        }
        if (rankCheck != info->nrank)
        {
            if (opt.verbose >= NONE) cout << "eigenlusol::getUcond: rank check failed with " << rankCheck << " (detected) and " << info->nrank << " (lusol)" << endl;
            return 1e153;
        }
#if TIMEMEASUREMENTS
        t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: getUcond"));
#endif

        // cout << "U\n" << *_U << endl;
        return maxVal / max(1e-100, minVal);
        // return 0;
    }

    int eigenlusol::lenRow(int idx)
    {

        if (!info)
        {
            cout << "Factorization not existing, call factorize() first" << endl;
            throw;
        }

#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif

        // construct U
        // initialize arrays for U triplets
        // the behaviour of lenU (nnz in U) can be erratic at times (for example < 0), especially in the update case, but we don't seem to need it (?)
        // cout << "Upartial uidx " << uidx << " q[uidx] - 1 " << q[uidx] - 1 << endl;

        // get row index
        int piv = (int)p[idx];
        // get length of row
        int len = int(lenr[(int)piv-1]);
        // get location of row
        int loc = int(locr[(int)piv-1]);
        // cout << "piv len loc " << piv << " " << len << " " << loc << endl;
        // load data into triplet arrays
        // permutation needs to be more efficient
        int lennnz = 0;
        for (int j = 0; j < len; j++)
        {
            if (abs(a[loc-1+j]) > 1e-31) lennnz++;
        }
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: getUcond")); }
#endif

        return lennnz;
    }

    void eigenlusol::constructZ(shared_ptr<mat> Zio)
    {
        if (!info)
        {
            cout << "Factorization not existing, call computeNS() first" << endl;
            throw;
        }
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        // FIXME: there is a bug here (ns test fails for large problems for either type 1 and 2)
        
        if (opt.nstype == 0)
        {
            // type 2
            // constructs the nullspace of input matrix A using the LU decomposition of A^T = P' L U Q
            // Z = P' * inv(L') * [0; I]
            // note that (P' * inv(L'))' = inv(L) * P

            // mat invLP(*m, *m);
            // invLP.setIdentity();
            // invLP = P() * invLP;

            // // apply inv(L) on the left of P
            // _L->triangularView<Eigen::Lower>().solveInPlace(invLP);
            computeInvLP();

            if (Zio)
            {
                Zio->conservativeResize(*n, *m - info->nrank);
                *Zio = _invLP->transpose().rightCols(*m - info->nrank);
            }
            else
            {
                _Z = make_unique<mat>(*n, *m - info->nrank);
                *_Z = _invLP->transpose().rightCols(*m - info->nrank);
            }
        }
        else if (opt.nstype == 1)
        {
            // type 1
            // constructs the nullspace of input matrix A using the LU decomposition of A = P' L U Q
            // Z = Q * [-invU1U2; I]
            // cout << "U\n" << (matd)U() << endl;
            mat U1 = U().block(0,0,info->nrank,info->nrank);
            // cout << "U1\n" << (matd)U1 << endl;
            mat U2 = U().block(0,info->nrank,info->nrank,*n-info->nrank);
            // cout << "U2\n" << (matd)U2 << endl;
            mat I(*n-info->nrank, *n-info->nrank); I.setIdentity();
            th.reset();
            U1.triangularView<Eigen::Upper>().solveInPlace(U2);
            // cout << "invU1U2\n" << (matd)U2 << endl;
            th.getTriplets(-U2);
            th.getTriplets(I, info->nrank, 0);
            if (Zio)
            {
                Zio->resize(*n, *n - info->nrank);
                th.setFromTriplets(*Zio);
                Q().applyThisOnTheLeft(*Zio);
                // cout << "NS test1 " << (U() * Q().transpose() * Z()).norm() << endl;
            }
            else
            {
                _Z = make_unique<mat>(*n, *n - info->nrank);
                th.setFromTriplets(*_Z);
                Q().applyThisOnTheLeft(*_Z);
                // cout << "NS test1 " << (U() * Q().transpose() * Z()).norm() << endl;
            }
        }

#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::factorize: constructZ")); }
#endif
    }

    void eigenlusol::computeInvLP()
    {
        // if (opt.verbose > NONE) cout << "eigenlusol::computeInvLP on level " << hp->l << endl;
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        // does this make sense? Corresponds to calculating inverse of L explicitly
        // on the other hand eigen::sparse doesn't allow solves on the right so quite a few temp's would be necessary every time this function is called
#if TIMEMEASUREMENTS
        cpplusol::timer t2;
#endif
        _invLP = make_unique<mat>(*m, *m);
        _invLP->setIdentity();
#if TIMEMEASUREMENTS
        { t2.stopTime(); times.push_back(cpplusol::time(t2.time, "eigenlusol::computeInvLP resize")); }
        cpplusol::timer t3;
#endif
        *_invLP = *_P * *_invLP;
#if TIMEMEASUREMENTS
        { t2.stopTime(); times.push_back(cpplusol::time(t2.time, "eigenlusol::computeInvLP copy")); }
#endif
        _L->triangularView<Eigen::Lower>().solveInPlace(*_invLP);
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::computeInvLP")); }
#endif
    }

    void eigenlusol::assignInfo()
    {
        if (!info)
            info = new lusolinfo;
        info->inform = luparm[9];
        info->nsing = luparm[10];
        info->jsing = luparm[11];
        info->minlen = luparm[12];
        info->maxlen = luparm[13];
        info->nupdat = luparm[14];
        info->nrank = luparm[15];
        info->ndens1 = luparm[16];
        info->ndens2 = luparm[17];
        info->jumin = luparm[18];
        info->numL0 = luparm[19];
        info->lenL0 = luparm[20];
        info->lenU0 = luparm[21];
        info->lenL = luparm[22];
        info->lenU = luparm[23];
        info->lrow = luparm[24];
        info->ncp = luparm[25];
        info->mersum = luparm[26];
        info->nUtri = luparm[27];
        info->nLtri = luparm[28];
        info->Amax = parmlu[9];
        info->Lmax = parmlu[10];
        info->Umax = parmlu[11];
        info->DUmax = parmlu[12];
        info->DUmin = parmlu[13];
        info->Akmax = parmlu[14];
        info->growth = parmlu[15];
        info->resid = parmlu[19];
    }

    void eigenlusol::init(const mat& A)
    {
#if SAFEGUARD
        _A = A;
        cout << "_A\n" << (matd)_A << endl;
#endif

        allocate(A);

        // extract data from A for use in LUSOL
        // FIXME: Can we just redirect the pointers?
#if TIMEMEASUREMENTS
        cpplusol::timer t1;
#endif
        int ctrnnz = 0;
        for (int k=0; k<A.outerSize(); ++k)
        {
            for (mat::InnerIterator it(A,k); it; ++it)
            {
                a[ctrnnz] = it.value();
                indc[ctrnnz] = it.row() + 1;   // row index
                indr[ctrnnz] = it.col() + 1;   // col index (here it is equal to k)
                ctrnnz++;
            }
        }
#if TIMEMEASUREMENTS
        { t1.stopTime(); times.push_back(cpplusol::time(t1.time, "eigenlusol::init: assign A")); }
#endif
    }

    void eigenlusol::allocate()
    {
        // basic allocations
        m = new int64_t[1];
        n = new int64_t[1];
        nelem = new int64_t[1];
        lena = new int64_t[1];
        luparm = new int64_t[30]; memset(luparm, 0, 30*sizeof(int64_t));
        parmlu = new double[30]; memset(parmlu, 0, 30*sizeof(double));
        inform = new int64_t[1];

        // allocate sufficient memory
        *lena = opt.asize;
        maxmn = opt.maxmn;

        if (opt.verbose >= CONV)
        {
            cout << "eigenlusol::allocate: lena " << *lena << ", maxmn " << maxmn;
            cout << endl;
        }

        a = new double[*lena]; memset(a, 0, *lena*sizeof(double));
        indc = new int64_t[*lena]; memset(indc, 0, *lena*sizeof(int64_t));
        indr = new int64_t[*lena]; memset(indr, 0, *lena*sizeof(int64_t));

        luparm[1] = -1;
        luparm[2] = opt.maxcol;
        luparm[5] = opt.pivot;
        luparm[7] = opt.keepLU;
        parmlu[0] = opt.Ltol1;
        parmlu[1] = opt.Ltol2;
        parmlu[2] = opt.small;
        parmlu[3] = opt.Utol1;
        parmlu[4] = opt.Utol2;
        parmlu[5] = opt.Uspace;
        parmlu[6] = opt.dens1;
        parmlu[7] = opt.dens2;

        // more allocations
        // vectors of length m
        p = new int64_t[maxmn]; memset(p, 0, maxmn*sizeof(int64_t));
        lenr = new int64_t[maxmn]; memset(lenr, 0, maxmn*sizeof(int64_t));
        locr = new int64_t[maxmn]; memset(locr, 0, maxmn*sizeof(int64_t));
        iqloc = new int64_t[maxmn]; memset(iqloc, 0, maxmn*sizeof(int64_t));
        ipinv = new int64_t[maxmn]; memset(ipinv, 0, maxmn*sizeof(int64_t));
        // vectors of length n
        w = new double[maxmn]; memset(w, 0, sizeof(double) * maxmn);
        q = new int64_t[maxmn]; memset(q, 0, sizeof(int64_t) * maxmn);
        lenc = new int64_t[maxmn]; memset(lenc, 0, sizeof(int64_t) * maxmn);
        locc = new int64_t[maxmn]; memset(locc, 0, sizeof(int64_t) * maxmn);
        iploc = new int64_t[maxmn]; memset(iploc, 0, sizeof(int64_t) * maxmn);
        iqinv = new int64_t[maxmn]; memset(iqinv, 0, sizeof(int64_t) * maxmn);

        v = new double[maxmn]; memset(v, 0, sizeof(double) * maxmn);
        vidx = new int64_t[maxmn]; memset(vidx, 0, sizeof(double) * maxmn);
        vL = veci::Constant(maxmn, -1);

        th = cpplusol::tripletHandler(opt.maxmn);
        th2 = cpplusol::tripletHandler(opt.maxmn);
    }

    void eigenlusol::allocate(const mat& A)
    {
        // get problem dimensions
        *m = A.rows();
        *n = A.cols();
        *nelem = A.nonZeros();
    }

    void eigenlusol::printPb()
    {
        cout << "======================= PRINT PROBLEM =========================" << endl;
#if SAFEGUARD
        cout << "Original decomposed matrix\n" << _A << endl;
#endif
        print("m",m,1);
        print("n",n,1);
        print("nelem",nelem,1);
        print("lena",lena,1);
        print("luparm",luparm, 30);
        print("parmlu",parmlu, 30);
        print("a",a, *lena);
        print("indr",indr, *lena);
        print("indc",indc, *lena);
        print("p",p, *m);
        print("q",q, *n);
        print("lenc",lenc, *n);
        print("lenr",lenr, *m);
        print("locc",locc, *n);
        print("locr",locr, *m);
        print("iploc",iploc, *n);
        print("iqloc",iqloc, *m);
        print("ipinv",ipinv, *m);
        print("iqinv",iqinv, *n);
        print("w",w, *n);
        cout << "==============================================================" << endl;
    }
} // namespace cpplusol
