#ifndef _TYPEDEFS_EIGENLUSOL_
#define _TYPEDEFS_EIGENLUSOL_

#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Jacobi>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <memory>


using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;
using std::make_shared;
using std::make_unique;
using std::cout;
using std::endl;
using std::string;
using std::min;
using std::max;
using std::abs;
using std::sort;
using Eigen::MatrixXd;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> mat;
typedef Eigen::Triplet<double> triplet;
typedef vector<triplet> tripletList;
typedef Eigen::VectorXd vec;
typedef Eigen::MatrixXd matd;

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mati;
typedef Eigen::VectorXi veci;

namespace cpplusol
{
    enum verboseLevel {NONE = 0, CONV, VAR, MAT};
    enum TP {TPP = 0, TRP, TCP, TSP, TCPNM};

    struct tripletHandler
    {
        public:
            tripletHandler() { }
            tripletHandler(int d) { tl.resize(d); }
            void reset() { tlctr = 0; }
            int& tripletCtr() { return tlctr; }
            inline tripletList& tripletL() { return tl; }

            void setTriplet(const int r, const int c, const double val)
            {
                tl[tlctr] = triplet(r, c, val);
                tlctr++;
            }

            void getTriplets(const mat& A, int rowOffset = 0, int colOffset = 0)
            {
                for (int k=0; k < A.outerSize(); ++k)
                {
                    for (mat::InnerIterator it(A,k); it; ++it)
                    {
                        tl[tlctr] = triplet(it.row() + rowOffset, it.col() + colOffset, it.value());
                        tlctr++;
                    }
                }
            }

            void getTripletsRows(const mat& A, vector<int> rowidx, int rowOffset = 0, int colOffset = 0)
            {
                // getting triplets from rows is inefficient due to column major ordering (row major column ordering would lead to deprecation of other functions, for example solve in place)
                // binary idx search over rowidx is cheaper than all matrix entries
                for (int k=0; k < A.outerSize(); ++k)
                {
                    for (mat::InnerIterator it(A,k); it; ++it)
                    {
                        std::vector<int>::iterator itr = std::find(rowidx.begin(), rowidx.end(), it.row());
                        if (itr != rowidx.cend())
                        {
                            tl[tlctr] = triplet(rowOffset + std::distance(rowidx.begin(), itr), it.col() + colOffset, it.value());
                            tlctr++;
                        }
                    }
                }
            }

            void getLC(const mat& A, veci& lc)
            {
                for (int k=0; k < A.outerSize(); ++k)
                {
                    for (mat::InnerIterator it(A,k); it; ++it)
                    {
                        if (lc(it.row()) < it.col()) lc(it.row()) = it.col();
                    }
                }
            }

            void getTripletsVec(vec& a, int rg, int rowOffset = 0, int colOffset = 0, string type = "col")
            {
                for (int k=0; k < rg; ++k)
                {
                    if (abs(a[k]) > 1e-31) 
                    {
                        if (type == "col")
                        {
                            tl[tlctr] = triplet(rowOffset + k, colOffset, a[k]);
                        }
                        else
                        {
                            tl[tlctr] = triplet(rowOffset, colOffset + k, a[k]);
                        }
                        tlctr++;
                    }
                }
            }

            void setFromTriplets(mat& A)
            {
                // cout << "Asize " << A.rows() << " x " << A.cols() << endl;
                // print();
                A.setFromTriplets(tripletL().begin(), tripletL().begin() + tripletCtr());
            }

            void print()
            {
                cout << "DATA::PRINTTRIPLETL: TripletList with counter " << tlctr << " and size " << tl.size() << endl;
                for (int i = 0; i < tlctr; i++)
                {
                    triplet& t = tl[i];
                    cout << "row " << t.row() << " col " << t.col() << " val " << t.value() << endl;
                }
            }

        private:
            int tlctr = 0;
            tripletList tl;
    };

    class timer
    {
        public:
            timer() { start = std::chrono::steady_clock::now(); }
            void stopTime(string mes="") { end = std::chrono::steady_clock::now(); std::chrono::duration<double> difference = end - start; time = difference.count(); if (mes != "") cout << "Timer::stopTime: measured time of " << mes << " is " << time << endl; }
            double time;
        private:
            std::chrono::steady_clock::time_point start;
            std::chrono::steady_clock::time_point end;
    };

    struct time
    {
        time(double _t, string _tag) : t(_t), tag(_tag) {}
        double t;
        string tag;
    };
}
#endif
