#ifndef CCD_INTERMEDIATES_OMP_H
#define CCD_INTERMEDIATES_OMP_H
#include <iostream>

using namespace std;

class CCD_Intermediates_OMP
{
private:
    //System variables:
    int number_of_electrons;
    int number_of_shells;
    int number_of_states;
    int number_of_virt_orb;
    int number_of_ccd_ampl;
    int length_of_intermediates;

    //HF variables:
    double * ABIJ_Block;
    double * ABCD_Block;
    double * KLIJ_Block;
    double * KBCJ_Block;
    double * KLCD_Block;
    double * fock_energies;
    double * TBME;

    //CCD variables:
    double * ccd_intermediates;
    double * ccd_intermediates1;
    double * ccd_intermediates2;
    double * ccd_intermediates3;
    double * ccd_intermediates4;
    double * ccd_amplitudes;
    double * ccd_root;
    double ccd_root_norm;
    double correlation_energy;
    int quasiNewIterations;
    double compTime;
    double MaxEvalTime;

public:
    CCD_Intermediates_OMP(int number_of_electrons_, int number_of_shells_,
                    double **HFBlocks, double * fock_energies_);

    void ccd_function();
    void quasi_newton(double eps, int max_it);
    void DIIS(double eps, int max_it);
    void corr_energy();

    //Getters:
    double getCorrEnergy();
    double getCompTime (){ return compTime; }
    int getIterationNumb (){ return quasiNewIterations; }
    double getEvalTime (){ return MaxEvalTime; }

    //Setters:
};

#endif // CCD_INTERMEDIATES_OMP_H
