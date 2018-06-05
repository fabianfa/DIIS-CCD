#ifndef HARTREEFOCK_H
#define HARTREEFOCK_H

#include <cmath>
#include <armadillo>
#include "basis.h"
#include "hermitepolynomials.h"
#include "Coulomb_Functions.h"

class HartreeFock
{
private:
    int N_SPS;
    int N_Electrons;
    int HFCounter = 0;
    double HFEnergy;
    double TimeForHFBasisTrafo;
    double TimeForHFComp;
    double lambda = 1e-5;
    double sqrtOmega = 1;
    double * interactionMatrix;
    arma::vec SPS_Energies;
    arma::mat densityMatrix;
    arma::mat C;
    bool orthogonalityTest = false;     // Bool to set if orthogonality test are to be performed
    bool orthogonalityResults = true;   // Bool to store test results in

    Basis *basis = nullptr;

    void initializeCMatrix();
    void updateDensityMatrix();
public:
    HartreeFock();
    HartreeFock(int NElectrons, int singleParticleStates);
    ~HartreeFock() { }
    void initializeHF(int NElectrons, int singleParticleStates, Basis *newBasis);
    int runHF(int maxHFIterations);
    void updateHFMatrix(arma::mat &HFMatrix);
    double calculateInnerHFMatrixElement(int alpha, int alpha_ml, int alphaSpin, int beta, int beta_ml, int betaSpin);
    void writeToFile();
    void writeTBME();
    void ComHFEnergy(double &HFEnergyResults, int &HFIterationsResults, arma::mat &HFReturnMatrix, arma::vec &HFSPSEnergiesReturnVector);
    void testCOrthogonality();
    double ** HFBasis();

    // Printers
    void printHFMatrix(arma::mat HFMatrix);
    void printSPEnergies();

    // Getters
    bool getOrthonormalityResults() { return orthogonalityResults; }
    arma::vec getSPS_Energies();
    double getHFEnergy();
    double getBaisTrafoTime(){return TimeForHFBasisTrafo; }
    double getHFCompTime(){return TimeForHFComp; }

    // Setters
    void setConvergence(double newLambda) { lambda = newLambda; }
    void setInteractionMatrix(double * newInteractionMatrix);
    void setOmega(double newOmega);
    void setTestOrthogonality(bool boolTest) { orthogonalityTest = boolTest; }
};

#endif // HARTREEFOCK_H
