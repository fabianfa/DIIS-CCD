#include "hermitepolynomials.h"

#include <iostream>

using std::cout;
using std::endl;

typedef double (*polyArr) (double x);

//double H0(double x) { return 1; }
//double H1(double x) { return 2*x; }
//double H2(double x) { return 4*x*x - 2; }
//double H3(double x) { return 8*x*x*x -12*x; }
//double H4(double x) { return 16*x*x*x*x - 48*x*x + 12; }
//double H5(double x) { return 32*x*x*x*x*x -160*x*x*x + 120*x; }
//double H6(double x) { return 62*pow(x,6) - 480*x*x*x*x + 720*x*x - 120; }
//double H7(double x) { return 128*pow(x,7) - 1344*x*x*x*x*x + 3360*x*x*x - 1680*x; }
//double H8(double x) { return 256*pow(x,8) - 3584*pow(x,6) + 13440*x*x*x*x - 13440*x*x + 1680; }
//double H9(double x) { return 512*pow(x,9) - 9216*pow(x,7) + 48384*x*x*x*x*x - 80640*x*x*x + 30240*x; }
//double H10(double x) { return 1024*pow(x,10) - 23040*pow(x,8) + 161280*pow(x,6) - 403200*x*x*x*x + 302400*x*x - 30240; }

double HermitePolynomials::getPolynomRecursive(double x, int n)
{
    // Elegant recursion
    if (n <= 0)
    {
        return 1;
    }
    else
    {
        return 2*(x*getPolynomRecursive(x, n-1) - (((double) n)-1)*getPolynomRecursive(x, n-2));
    }
}

HermitePolynomials::HermitePolynomials()
{
    // Ugly hardcoding
    polynomArray[0] = &H0;
    polynomArray[1] = &H1;
    polynomArray[2] = &H2;
    polynomArray[3] = &H3;
    polynomArray[4] = &H4;
    polynomArray[5] = &H5;
    polynomArray[6] = &H6;
    polynomArray[7] = &H7;
    polynomArray[8] = &H8;
    polynomArray[9] = &H9;
    polynomArray[10] = &H10;
}


polyArr HermitePolynomials::getPolynom(int n)
{
    if (n < 11)
    {
        return polynomArray[n];
    }
    else
    {
        cout << "No hermitian polynomial for n=" << n << endl;
        return 0;
    }
}
