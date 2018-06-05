#ifndef HERMITEPOLYNOMIALS_H
#define HERMITEPOLYNOMIALS_H

#include <cmath>

typedef double (*polyArr) (double x);

class HermitePolynomials
{
private:
    static double H0(double x) { return 1; }
    static double H1(double x) { return 2*x; }
    static double H2(double x) { return 4*x*x - 2; }
    static double H3(double x) { return 8*x*x*x -12*x; }
    static double H4(double x) { return 16*x*x*x*x - 48*x*x + 12; }
    static double H5(double x) { return 32*x*x*x*x*x -160*x*x*x + 120*x; }
    static double H6(double x) { return 62*pow(x,6) - 480*x*x*x*x + 720*x*x - 120; }
    static double H7(double x) { return 128*pow(x,7) - 1344*x*x*x*x*x + 3360*x*x*x - 1680*x; }
    static double H8(double x) { return 256*pow(x,8) - 3584*pow(x,6) + 13440*x*x*x*x - 13440*x*x + 1680; }
    static double H9(double x) { return 512*pow(x,9) - 9216*pow(x,7) + 48384*x*x*x*x*x - 80640*x*x*x + 30240*x; }
    static double H10(double x) { return 1024*pow(x,10) - 23040*pow(x,8) + 161280*pow(x,6) - 403200*x*x*x*x + 302400*x*x - 30240; }

    polyArr * polynomArray = new polyArr[11];
public:
    HermitePolynomials();

    polyArr getPolynom(int n);
    double getPolynomRecursive(double x, int n);
};

#endif // HERMITEPOLYNOMIALS_H
