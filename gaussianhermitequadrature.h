#ifndef GAUSSIANHERMITEQUADRATURE_H
#define GAUSSIANHERMITEQUADRATURE_H


class GaussianHermiteQuadrature
{
private:
//    double *Integrand = nullptr;
    double (*Integrand)(double x1, double x2, double x3, double x4);
    void getWeights(double * x, double * w, int n);
public:
    GaussianHermiteQuadrature();
    void setIntegrand(double (*I)(double x1, double x2, double x3, double x4));
    double solve(int n);

};


#endif // GAUSSIANHERMITEQUADRATURE_H
