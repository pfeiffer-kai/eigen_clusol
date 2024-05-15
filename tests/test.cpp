// This file is part of EIGENLUSOL.
//
// Copyright (c) 2024 Kai Pfeiffer
//
// This source code is licensed under the BSD 3-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <chrono>
#include <random>
#include <set>

#include <eigenlusol/eigenlusol.h>
#include <eigenlusol/typedefs.h>

using Eigen::SparseMatrix;

using namespace std;
using namespace cpplusol;

typedef Eigen::Triplet<double> triplet;

template <typename T> void print(string name, T* a, int len)
{
    cout << name << ":\n";
    for (int i=0;i<len;i++)
    {
        cout << (a)[i] << " ";
    }
    cout << endl;
}

Eigen::SparseMatrix<double> getRandomSpMat(size_t rows, size_t cols, double p) {
    typedef Eigen::Triplet<double> T;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    double range=1e2;
    std::uniform_real_distribution<> valdis(0, range);
    std::uniform_int_distribution<> rowdis(0, rows-1);
    std::uniform_int_distribution<> coldis(0, cols-1);

    std::vector<Eigen::Triplet<double> > tripletList;
    size_t nnz = (size_t) max(1.0, ((double)rows * ((double)cols * p)));
    std::set<size_t> nnz_pos;
    for (size_t i = 0; i < nnz; ++i) {
        auto r = rowdis(gen);
        auto c = coldis(gen);
        size_t pos = r * cols + c;
        while (nnz_pos.find(pos) != nnz_pos.end()) {
            r = rowdis(gen);
            c = coldis(gen);
            pos = r * cols + c;
        }

        nnz_pos.insert(pos);
        tripletList.push_back(T(r, c, valdis(gen) - range/2.));
    }

    Eigen::SparseMatrix<double> out(rows,cols);
    out = matd::Random(rows,cols).sparseView(); // this produces the same random matrix everytime
    out.setFromTriplets(tripletList.begin(), tripletList.end());   //create the matrix
    out.makeCompressed();
    return out;
}


int main()
{
    eigenlusol * lusol;
    std::vector<triplet> tripletList;

    // tripletList.clear();
    // tripletList.reserve(3);
    // tripletList.push_back(triplet(0,0,0));
    // tripletList.push_back(triplet(0,1,0));
    // tripletList.push_back(triplet(0,2,0));
    // SparseMatrix<double> A(1, 3);
    // A.setFromTriplets(tripletList.begin(), tripletList.end());

    // tripletList.clear();
    // tripletList.reserve(6);
    // tripletList.push_back(triplet(0,0,0));
    // tripletList.push_back(triplet(0,1,0));
    // tripletList.push_back(triplet(0,2,0));
    // tripletList.push_back(triplet(1,0,1));
    // tripletList.push_back(triplet(1,1,1));
    // tripletList.push_back(triplet(1,2,1));
    // SparseMatrix<double> A(2, 3);
    // A.setFromTriplets(tripletList.begin(), tripletList.end());

    tripletList.clear();
    tripletList.reserve(6);
    tripletList.push_back(triplet(0,0,1));
    tripletList.push_back(triplet(0,1,2));
    tripletList.push_back(triplet(0,2,3));
    tripletList.push_back(triplet(0,3,4));
    tripletList.push_back(triplet(1,0,5));
    tripletList.push_back(triplet(1,1,6));
    tripletList.push_back(triplet(1,2,7));
    tripletList.push_back(triplet(1,3,8));
    tripletList.push_back(triplet(2,0,9));
    tripletList.push_back(triplet(2,1,10));
    tripletList.push_back(triplet(2,2,11));
    tripletList.push_back(triplet(2,3,12));
    SparseMatrix<double> A(3, 4);
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    mat C = getRandomSpMat(5,4,1);
    mat D = getRandomSpMat(4,8,1);
    A = C * D;

    matd AA = matd::Zero(5,8);
    // AA << 137.306, -47.5911,  468.669, -469.605,
// -245.455,  455.445,  260.019,  460.845,
//  494.078,  337.077, -426.914,  427.808;
//     A = AA.sparseView();
    AA << 
        -168.662,  129.557, -68.6728,  229.939,  514.979,  562.632,  -270.79,  326.136,
-712.521,  2272.59,  723.527, -378.392,  959.206, -185.129,  307.475, -1012.64,
 1378.42, -1974.15,  554.125, -70.7709, -3201.81, -2019.27, -380.868,  757.338,
 2212.37,  233.508,  327.608,  1846.22, -3104.65, -2462.02,  2033.22, -667.653,
-786.561,  1117.12, -736.524,  454.136,  2338.01,  1705.41,  486.229, -418.256;
    A = AA.sparseView();
    cout << "A\n" << (matd)A << endl;

    // tripletList.clear();
    // tripletList.reserve(6);
    // tripletList.push_back(triplet(0,0,10));
    // tripletList.push_back(triplet(0,1,20));
    // tripletList.push_back(triplet(1,0,30));
    // tripletList.push_back(triplet(1,1,40));
    // tripletList.push_back(triplet(0,2,50));
    // tripletList.push_back(triplet(1,2,60));
    // tripletList.push_back(triplet(2,0,80));
    // tripletList.push_back(triplet(2,1,50));
    // tripletList.push_back(triplet(2,2,90));
    // SparseMatrix<double> C(3, 3);
    // C.setFromTriplets(tripletList.begin(), tripletList.end());
    // cout << "C\n" << C << endl;
    // cout << "C * C\n" << C * C << endl;
    // mat D = C;
    // C.applyThisOnTheRight(D);
    // cout << "C * C in place\n" << D << endl;

    // throw;
    // lusol = new eigenlusol;

    // lusol->factorize(A);

    // SparseMatrix<double> out = (lusol->P().transpose() * lusol->L() * lusol->U() * lusol->Q().transpose());

    // cout << "P\n" << lusol->P() << endl;
    // cout << "L\n" << lusol->L() << endl;
    // cout << "U\n" << lusol->U() << endl;
    // cout << "Q\n" << lusol->Q() << endl;
    // cout << "A - L*U\n" << A - out << endl;

    // throw;

    lusol = new eigenlusol();

    lusol->computeNS(A, NULL, false);
    cout << "L\n" << (matd)lusol->L() << endl;
    cout << "U\n" << (matd)lusol->U() << endl;

    SparseMatrix<double> outAt = lusol->P().transpose() * lusol->L() * lusol->U() * lusol->Q().transpose();
    cout << "outAt\n" << (matd)outAt << endl;
    mat AT = A;
    cout << "A - PLUQ " << matd(A - outAt).norm() << endl;
    mat A_ = A;
    cout << "Z\n" << (matd)lusol->Z() << endl;
    cout << "AZ\n" << (matd)lusol->applyNSOnTheRight(A_) << endl;
    cout << "A * Z\n" << (matd)(A * lusol->Z()) << endl;;
    cout << "A * Z " << (A * lusol->Z()).norm() << endl;;

    // tripletList.clear();
    // tripletList.reserve(6);
    // tripletList.push_back(triplet(0,0,5));
    // tripletList.push_back(triplet(0,1,7));
    // tripletList.push_back(triplet(1,0,8));
    // tripletList.push_back(triplet(1,1,3));
    // tripletList.push_back(triplet(0,2,2));
    // tripletList.push_back(triplet(1,2,1));
    // SparseMatrix<double> B(3, 4);
    // B.setFromTriplets(tripletList.begin(), tripletList.end());

    // cout << "BZ\n" << (matd)lusol->applyNSOnTheRight(B) << endl;
}

