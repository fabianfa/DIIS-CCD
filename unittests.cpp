#include <iostream>
#include <iomanip>
#include <time.h>
#include <cmath>
#include <vector>
#include <math.h>
#include <mpi.h>
#include "quantumdot.h"
#include "hermitepolynomials.h"
#include "gaussianhermitequadrature.h"
#include "functions.h"

using std::cout;
using std::endl;
using std::setw;
using std::setprecision;
using std::vector;

int runHermiteTimer(int N, double X)
{
    /*
     * Benchmarking for the Hermitian polynom retriever.
     * N: number of iterations
     * X: value to insert into the Hermitian polynoms
     */
    HermitePolynomials herm;

    clock_t startHardCoded, finishHardCoded;
    startHardCoded = clock();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            herm.getPolynom(j)(X);
        }
    }
    finishHardCoded = clock();

    clock_t startRecursive, finishRecursive;
    startRecursive = clock();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            herm.getPolynomRecursive(X,j);
        }
    }
    finishRecursive = clock();

    cout << "Hardcoded results: " << setw(10) << setprecision(10) << (finishHardCoded - startHardCoded)/((double)CLOCKS_PER_SEC/1000) << endl;
    cout << "Recursive results: " << setw(10) << setprecision(10) << (finishRecursive - startRecursive)/((double)CLOCKS_PER_SEC/1000) << endl;

    return 0;
}

int checkCorrectHermitePolynoms(int maxHermite)
{
    /*
     * Checks that the different methods are equal
     */
    if (maxHermite > 10)
    {
        maxHermite = 10;
    }

    HermitePolynomials Hermite;

    double overflowX = 0;
    int overflowN = 0;

    int correct = 0;
    double eps = 1e-15;
    for (int i = 0; i < maxHermite + 1; i++)
    {
        for (double x = 0; x < 10; x = x + 0.1)
        {
            if (std::abs(Hermite.getPolynomRecursive(x,i) - Hermite.getPolynom(i)(x)) < eps)
            {
                correct++;
            }
            else
            {
                overflowX = x;
                overflowN = i;
                break;
            }
        }
    }
    cout << "Correct Hermite polynoms: " << correct << endl;
    if (overflowN > 0)
    {
        cout << "Most likely overflow encountered at Hermite polynomial N=" << overflowN << " and value x=" << overflowX << endl;
    }

    return 0;
}

double testFunction(double x)
{
    /*
     * Function used to test Gaussian quadrature
     */
    return 4*x*x;
}

double emptyFunction(double x)
{
    /*
     * Empty function used to test Gaussian quadrature
     */
    return 1;
}

double emptyPotentialFunction(double x1, double x2, double x3, double x4)
{
    /*
     * Empty function used to test Gaussian quadrature
     */
    return 1;
}

double testIntegrand(double x1, double x2, double x3, double x4)
{
    /*
     * Function used to test the integrator class.
     */
    return testFunction(x1);
}

void testIntegratorClass(int NIntPoints, double eps=1e-10)
{
    /*
     * Small function for testing my integration class.
     */
    double analyticalSolution = 2*M_PI*M_PI;

    GaussianHermiteQuadrature integrator;
    integrator.setIntegrand(testIntegrand);
    double numericalSolution = integrator.solve(NIntPoints);

    printf("  Integration points: %d \n", NIntPoints);
    printf("  Numerical:  %.10f \n", numericalSolution);
    printf("  Analytical: %.10f \n", analyticalSolution);

    if (fabs(analyticalSolution - numericalSolution) < eps)
    {
        printf("  TEST PASSED \n");
    }
    else
    {
        printf("  TEST FAILED \n");
    }
}

void testIntegratorFunction(int NIntPoints, double eps=1e-10)
{
    /*
     * Small function for testing my integration function
     */
    double analyticalSolution = 2*M_PI*M_PI;

    // Not a pretty method
    double numericalSolution = solveGaussianHermiteQuadrature(NIntPoints, testFunction, emptyFunction, emptyFunction, emptyFunction, emptyFunction, emptyFunction, emptyFunction, emptyFunction, emptyPotentialFunction);

    printf("  Integration points: %d \n", NIntPoints);
    printf("  Numerical:  %.10f \n", numericalSolution);
    printf("  Analytical: %.10f \n", analyticalSolution);

    if (fabs(analyticalSolution - numericalSolution) < eps)
    {
        printf("  TEST PASSED \n");
    }
    else
    {
        printf("  TEST FAILED \n");
    }
}

void testOrthogonality(int numberOfArguments, char *cmdLineArguments[])
{
    /*
     * Function for testing the orthogonality in the Hartree-Fock method
     */
    int numprocs, processRank;
    MPI_Init (&numberOfArguments, &cmdLineArguments);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &processRank);

    if (processRank == 0) { cout << "UNIT TEST: Running orthogonality unit test for C matrix" << endl; }

    int passed = 0; // Counts number of anomalies
    int NElectronArrElems   = 4;
    int NElectronsArray[NElectronArrElems];
    NElectronsArray[0]      = 2;
    NElectronsArray[1]      = 6;
    NElectronsArray[2]      = 12;
    NElectronsArray[3]      = 20;
    int startShell          = 3;
    int maxShell            = 8;
    int maxHFIterations     = 500;
    double omega            = 1.0;
    double epsilon          = 1e-10;


    for (int i = 0; i < NElectronArrElems; i++)
    {
        for (int shells = startShell; shells < maxShell; shells++)
        {
            quantumDot QMDot(NElectronsArray[i], shells, omega);
            if (false == checkElectronShellNumber(QMDot.getN_SPS(), QMDot.getN_Electrons())) { continue; }
            QMDot.initializeHF();
            QMDot.setupInteractionMatrixPolarParalell(numprocs, processRank);
            if (processRank == 0)
            {
                QMDot.setHFLambda(epsilon);
                QMDot.setTestOrthogonoality(true); // Orthogonality test
                QMDot.runHartreeFock(maxHFIterations);
                passed = QMDot.getOrthonormalityResults();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    if (processRank == 0)
    {
        if (passed) { cout << "TEST PASSED: Orthogonality preserved." << endl; }
        else { cout << "TEST FAILED: Orthogonality not preserved." << endl; }
    }
}

void testDegeneracy(int numberOfArguments, char *cmdLineArguments[])
{
    /*
     * Function for testing if the degeneracy is preserved in the Hartree-Fock method.
     */
    int numprocs, processRank;
    MPI_Init (&numberOfArguments, &cmdLineArguments);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &processRank);

    if (processRank == 0) { cout << "UNIT TEST: Comparing degeneracies before and after HF-algorithm." << endl; }

    int NElectronArrElems   = 4;
    int NElectronsArray[NElectronArrElems];
    NElectronsArray[0]      = 2;
    NElectronsArray[1]      = 6;
    NElectronsArray[2]      = 12;
    NElectronsArray[3]      = 20;

    int startShell          = 2;
    int maxShell            = 6;
    int maxHFIterations     = 500;
    double omega            = 1.0;
    double epsilon          = 1e-10;

    for (int i = 0; i < NElectronArrElems; i++)
    {
        for (int shells = startShell; shells < maxShell; shells++)
        {
            quantumDot QMDot(NElectronsArray[i], shells, omega);
            if (false == checkElectronShellNumber(QMDot.getN_SPS(), QMDot.getN_Electrons())) { continue; }
            QMDot.initializeHF();
            QMDot.setupInteractionMatrixPolarParalell(numprocs, processRank);
            QMDot.setOmega(omega);
            if (processRank == 0)
            {
                QMDot.setHFLambda(epsilon);
                QMDot.runHartreeFock(maxHFIterations);
                QMDot.compareDegenerateStates();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

void testUnperturbedHF(int numberOfArguments, char *cmdLineArguments[])
{
    /*
     * Function for testing that the Hartree-Fock method converges after one iterations if the Hamiltonian is unperturbed.
     */
    int numprocs, processRank;
    MPI_Init (&numberOfArguments, &cmdLineArguments);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &processRank);

    if (processRank == 0) { cout << "UNIT TEST: Ensuring unperturbed Hartree-Fock potential converges after 1 HF-iteration." << endl; }

    bool passed = true; // Counts number of anomalies
    int NElectronArrElems   = 4;
    int NElectronsArray[NElectronArrElems];
    NElectronsArray[0]      = 2;
    NElectronsArray[1]      = 6;
    NElectronsArray[2]      = 12;
    NElectronsArray[3]      = 20;
    int startShell          = 3;
    int maxShell            = 9;
    int maxHFIterations     = 500;
    double omega            = 1.0;
    double epsilon          = 1e-10;

    for (int i = 0; i < NElectronArrElems; i++)
    {
        for (int shells = startShell; shells < maxShell; shells++)
        {
            quantumDot QMDot(NElectronsArray[i], shells, omega);
            if (checkElectronShellNumber(QMDot.getN_SPS(), QMDot.getN_Electrons()) == false) { continue; }
            QMDot.initializeHF();
            QMDot.setupEmptyInteractionMatrix();
            if (processRank == 0)
            {
                QMDot.setHFLambda(epsilon);
                QMDot.runHartreeFock(maxHFIterations);
                if (QMDot.getHFIterations() != 1)
                {
                    passed = false;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();

    if (processRank == 0)
    {
        if (passed) { cout << "TEST PASSED: Correct number of iterations for the unperturbed HF potential." << endl; }
        else { cout << "TEST FAILED: inncorrect result for unperturbed HF potential." << endl; }
    }
}
