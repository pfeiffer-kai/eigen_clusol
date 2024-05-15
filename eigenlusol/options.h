#ifndef _OPTIONS_EIGENLUSOL_
#define _OPTIONS_EIGENLUSOL_

#pragma once

// FIXME: everything constant?

// this file contains the option settings for 
// cpplusol

namespace cpplusol
{
    struct options
    {
        int verbose = NONE;
        int computeOnce = true;
        
        // 0: L^-1, based on LU decomposition of A^T
        // 1: Q [-invU1 U2 \\ I], based on LU decomposition of A
        int nstype = 1;

        int asize = 1e7; // 10000
        int maxm = 1e7;
        int maxn = 1e7;
        int maxmn = 1e7;
        int sizefac = 10; // variables that need certain buffer in case of column additions
        double eps = 1e-12; // from matlab
        // storage parameters
        int nzinit = 0;
        // lusol integer parameters
        int maxcol = 1e6;
        // 'TPP' (partial pivoting)
        // 'TRP' (Threshold Rook Pivoting
        // 'TCP' (complete pivoting)
        // 'TSP' (Threshold Symmetric Pivoting), // not usable
        // 'TBP': threshold block pivoting, // not usable
        // 'TCPNM': complete pivoting without Markovitz}: more stable option, but not dense and not as expensive as dens2=0; preferably, the input data should be scaled; otherwise weaker sparse elements will be ignored, resulting in a denser, more expensive decomposition
        int pivot = TCP;
        int keepLU = 1;

        // double param
        double Ltol1 = 1; // tolerance factorization, min 1
        double Ltol2 = 1; // tolerance during updates, min 1
        // double small = std::pow(eps,.8);
        // double Utol1 = std::pow(eps,.67);
        // double Utol2 = std::pow(eps,.67);
        double small = 1e-12;
        double Utol1 = 1e-12;
        double Utol2 = 1e-12;
        double Uspace = 3.0;
        double dens1 = 0.3;
        // BEWARE: setting this to zero does not induce a dense LU from the start (only after some columns); so we made some changes in lusol to enforce this
        // from experience, setting this to zero results in the most accurate factorizations; but very expensive since everything is treated dense. Use TCPNM instaed (slightly less stable, FIXME)
        double dens2 = 0.5; // 0.5
    };
} // namespace cpplusol
#endif
