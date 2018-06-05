#ifndef CCD_BRUTE_FORCE_H
#define CCD_BRUTE_FORCE_H
#include <iostream>

using namespace std;

class CCD_Brute_Force
{
private:
    //System variables:
    int number_of_electrons;
    int number_of_shells;
    int number_of_states;
    int number_of_virt_orb;
    int number_of_ccd_ampl;

    //HF variables:
    double * ABIJ_Block;
    double * ABCD_Block;
    double * KLIJ_Block;
    double * KBCJ_Block;
    double * KLCD_Block;
    double * fock_energies;
    double * TBME;

    //CCD variables:
    double * ccd_amplitudes;
    double * ccd_root;
    double ccd_root_norm;
    double correlation_energy;
    int quasiNewIterations;
    double compTime;
    double MaxEvalTime;

public:
    CCD_Brute_Force(int number_of_electrons_, int number_of_shells_,
                    double **HFBlocks, double * fock_energies_);

    void ccd_function();
    void quasi_newton(double eps, int max_it);
    void corr_energy();

    //Getters:
    double getCorrEnergy ();
    double getCompTime (){ return compTime; }
    int getIterationNumb (){ return quasiNewIterations; }
    double getEvalTime (){ return MaxEvalTime; }

    //Setters:
};

#endif // CCD_BRUTE_FORCE_H
