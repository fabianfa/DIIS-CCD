#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <armadillo>
#include "unittests.h"
#include "singlestate.h"
#include "basis.h"
#include "hermitepolynomials.h"
#include "functions.h"
#include "quantumdot.h"
#include "hartreefock.h"
#include "ccd_brute_force.h"
#include "ccd_intermediates.h"
#include "ccd_intermediates_omp.h"
#include <mpi.h> // For mac
//#include "mpi/mpi.h" // For ubuntu
#include <omp.h>
//#include <libiomp/omp.h>

using namespace std;

int main(int numberOfArguments, char *cmdLineArguments[])
{
omp_set_num_threads(4);
/*
#pragma omp parallel sections
{
    #pragma omp section
    {
        printf ("section 1 id = %d, \n", omp_get_thread_num());
    }
    #pragma omp section
    {
        printf ("section 2 id = %d, \n", omp_get_thread_num());
    }
    #pragma omp section
    {
        printf ("section 3 id = %d, \n", omp_get_thread_num());
    }
        //printf ("section 3 id = %d, \n", omp_get_thread_num());
}
*/



    int NElectronArrElems   = 4;
    int NElectronsArray[NElectronArrElems];
    NElectronsArray[0]      = 2;
    NElectronsArray[1]      = 6;
    NElectronsArray[2]      = 12;
    NElectronsArray[3]      = 20;

    int NShellArrElems      = 5;
    int shells[NShellArrElems];
    shells[0]               = 3;
    shells[1]               = 4;
    shells[2]               = 5;
    shells[3]               = 6;
    shells[4]               = 7;

    int maxHFIterations     = 1000;
    int noOmegas            = 3;
    double omega[noOmegas];
    omega[0]                = 1;
    omega[1]                = 0.5;
    omega[2]                = 0.28;
    double epsilon          = 1e-10;

    double ** HFBlocks;
    double HF_Energy = 0;
    arma::vec HSPSEnergies;
    int number_of_states = 0;
    int number_of_virt_ampl = 0;

    ofstream results;
    results.open ("results.txt");
    int fileWidth = 12;

    clock_t setupStart, setupFinish;
    setupStart = clock();

    int numprocs, processRank;
    MPI_Init (&numberOfArguments, &cmdLineArguments);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &processRank);

    if (processRank == 0) { cout << "Starting up..." << endl; }

//    for (int i = magicNumberIndex; i < magicNumberIndex+1; i++)
    for (int i = 0; i < NElectronArrElems; i++)
    {
        for (int s = 0; s < NShellArrElems; s++)
        {
            quantumDot QMDot(NElectronsArray[i], shells[s], omega[0]);
            if (false == checkElectronShellNumber(QMDot.getN_SPS(), QMDot.getN_Electrons())) { continue; }
            QMDot.initializeHF();
            QMDot.setupInteractionMatrixPolarParalell(numprocs, processRank);
            if (processRank == 0)
            {
                for (int j = 0; j < noOmegas; j++)
                {
                    QMDot.setOmega(omega[j]);
                    QMDot.setHFLambda(epsilon);
                    HFBlocks = QMDot.runHartreeFock(maxHFIterations);
                    HF_Energy = QMDot.getHFEnergy();
                    //QMDot.storeResults(filename);
                    results << " NElectrons: "   << setw(2) << NElectronsArray[i]
                            << " Shells: "       << setw(2) << shells[s]
                            << " Omega: "        << setw(3) << omega[j]
                            << " HFIterations: " << setw(3) << QMDot.getHFIterations()
                            << " HFCompTime: "   << setw(fileWidth) << QMDot.getHFCompTime()
                            << " HFEnergy: "     << setw(8) << HF_Energy
                            << " HFBasisTrafo: " << setw(fileWidth) << QMDot.getBaisTrafoTime();


                    HSPSEnergies = QMDot.getHSPSEnergies();
                    number_of_states = shells[s]*shells[s]+shells[s];
                    number_of_virt_ampl = number_of_states-NElectronsArray[i];
                    int Index = 0;
                    double * fock_energies = new double[ (int) pow(number_of_states,4)];

                    //cout << "Fock Energies:" << endl;
                    for(int k = 0; k < NElectronsArray[i]; k++){
                        for(int l = 0; l < NElectronsArray[i]; l++){
                            for(int b = 0; b <number_of_virt_ampl; b++){
                                for(int a =0; a < number_of_virt_ampl; a++){
                                    Index = number_of_virt_ampl*(number_of_virt_ampl*(NElectronsArray[i]*k+l)+b)+a;
                                    fock_energies[Index]
                                        =HSPSEnergies(a+NElectronsArray[i])+HSPSEnergies(b+NElectronsArray[i])
                                        -HSPSEnergies(l)-HSPSEnergies(k);
                                    //cout << fock_energies[Index] << endl;
                                    if (fock_energies[Index] < 1e-15)
                                        fock_energies[Index] = 0;
                                }
                            }
                        }
                    }

                    CCD_Intermediates_OMP CCD1 = CCD_Intermediates_OMP(NElectronsArray[i], shells[s], HFBlocks, fock_energies);
                    CCD1.DIIS(epsilon, 600);

                    results << " CCDIterations: " << setw(3) << CCD1.getIterationNumb()
                            << " CCDComptTime: "  << setw(fileWidth) << CCD1.getCompTime()
                            << " CCDEvalTime: "   << setw(fileWidth) << CCD1.getEvalTime()
                            << " CCDCorrEnergy: " << setw(fileWidth) << CCD1.getCorrEnergy()
                            << " TotalEnergy:  "  << setw(8) << CCD1.getCorrEnergy()+HF_Energy
                            << endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    setupFinish = clock();
    if (processRank == 0)
    {
        cout << "Program complete. Time used: " << ((setupFinish - setupStart)/((double)CLOCKS_PER_SEC)) << endl;
    }
    /*
     * TODO:
     * [ ] Increase comment density
     * [ ] Add write-to-file capability(easy). For future project
     */
    results.close();
    return 0;














    /*
    //    testOrthogonality(numberOfArguments, cmdLineArguments);
    //    testDegeneracy(numberOfArguments, cmdLineArguments);
    //    testUnperturbedHF(numberOfArguments, cmdLineArguments);
    //    exit(1);

    int NElectronArrElems   = 2;
    int NElectronsArray[NElectronArrElems]; // Ugly setup
    NElectronsArray[0]      = 6;
    NElectronsArray[1]      = 12;

    int electrons = 6;
    int shells = 4;
    double omega = 1;
    double epsilon = 1e-10;
    int maxHFIterations = 500;
    std::string filename    = "../output3/HF_results";
    double ** HFBlocks;
    double HF_Energy = 0;

    int numprocs, processRank;
    MPI_Init (&numberOfArguments, &cmdLineArguments);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &processRank);

    for (int i = 0; i < NElectronArrElems; i++)
    {
        quantumDot QMDot(NElectronsArray[i], shells, omega);
        QMDot.initializeHF();
        QMDot.setupInteractionMatrixPolarParalell(numprocs, processRank);
        if (processRank == 0)
        {
            QMDot.setOmega(omega);
            QMDot.setHFLambda(epsilon);
            cout << "Length Interaction Matrix: "<< QMDot.getInteractionMatrixLength() << ", expected value : " <<pow(pow(shells,2)+shells,4)<< endl;
            HFBlocks = QMDot.runHartreeFock(maxHFIterations);
            HF_Energy = QMDot.getHFEnergy();
            QMDot.storeResults(filename);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();

        arma::vec HSPSEnergies = QMDot.getHSPSEnergies();
        int number_of_states = shells*shells+shells;
        int number_of_virt_ampl = number_of_states-electrons;
        int Index = 0;
        double * fock_energies = new double[ (int) pow(number_of_states,4)];

        for(int j = 0; j < electrons; j++){
            for(int i = 0; i < electrons; i++){
                for(int b = 0; b <number_of_virt_ampl; b++){
                    for(int a =0; a < number_of_virt_ampl; a++){
                        Index = number_of_virt_ampl*(number_of_virt_ampl*(electrons*j+i)+b)+a;
                        fock_energies[Index]
                            =HSPSEnergies(a+electrons)+HSPSEnergies(b+electrons)
                            -HSPSEnergies(i)-HSPSEnergies(j);
                        if (fock_energies[Index] < 1e-15)
                            fock_energies[Index] = 0;
                    }
                }
            }
        }

        CCD_Brute_Force test = CCD_Brute_Force(electrons, shells, HFBlocks, fock_energies);
        test.quasi_newton(1e-10, 200);
        cout << "Total Energy: " << HF_Energy + test.getCorrEnergy() << endl;


        //CCD_Intermediates testInter = CCD_Intermediates(electrons, shells, HFBlocks, fock_energies);
        //testInter.quasi_newton(1e-10, 200);
        //cout << "Total Energy: " << HF_Energy + testInter.getCorrEnergy() << endl;
    }

    return 0;
     */
}
