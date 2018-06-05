#ifndef BASIS_H
#define BASIS_H
#include <vector>
#include <singlestate.h>

using std::vector;

class Basis
{
private:
    vector<SingleState *> statesArray;
    int maxShell;
public:
    Basis();
    ~Basis();

    double omega;

    void initializeBasis(int maxShell, double initOmega);

    // Getters
    SingleState * getState(int state);
    int getMaxShell() { return maxShell; }
    int getShellDegeneracy(int shellNumber);
    int getTotalDegeneracy();
    int getTotalParticleNumber();
    int getShellParticles(int shellNumber);
    double getOmega() { return omega; }
    double getTotalEnergy();
    double getShellEnergy(int shellNumber);

    // Printers
    void printBasisSize();
    void printBasis(bool polar=false);

    // Setters
    void setOmega(double newOmega);
};

#endif // BASIS_H
