#include "hartreefock.h"
#include <armadillo>
#include <iomanip>
#include <ctime>
#include "functions.h"
#include "Coulomb_Functions.h"

using std::cout;
using std::endl;
using std::setprecision;
using std::setw;

HartreeFock::HartreeFock()
{

}

void HartreeFock::initializeHF(int NElectrons, int singleParticleStates, Basis *newBasis)
{
    N_Electrons = NElectrons;
    N_SPS = singleParticleStates;
    basis = newBasis;
    densityMatrix = arma::zeros<arma::mat>(N_SPS,N_SPS);
    C = arma::zeros<arma::mat>(N_SPS,N_SPS);
    // Initializing C as a diagonal matrix(other initializations exists)
    initializeCMatrix();
    // Setting up the density matrix
    updateDensityMatrix();
}

void HartreeFock::initializeCMatrix()
{
    /*
     * Initializing the C-matrix.
     */
    for (int i = 0; i < N_SPS; i++)
    {
        C(i,i) = 1;
    }
}

void HartreeFock::updateDensityMatrix()
{
    /*
     * Function for updating the density matrix
     */
    for (int gamma = 0; gamma < N_SPS; gamma++)
    {
        for (int delta = 0; delta < N_SPS; delta++)
        {
            double sum = 0;
            for (int i = 0; i < N_Electrons; i++)
            {
                sum += C(gamma,i)*C(delta,i);
            }
            densityMatrix(gamma,delta) = sum;
        }
    }
}

void HartreeFock::setInteractionMatrix(double * newInteractionMatrix)
{
    /*
     * Function for setting the interaction matrix to be used in the algorithm.
     */
    interactionMatrix = newInteractionMatrix;
}

void HartreeFock::updateHFMatrix(arma::mat &HFMatrix)
{
    /*
     * Function for updating the Hartree-Fock matrix.
     */
    HFMatrix.zeros();
    for (int alpha = 0; alpha < N_SPS; alpha++)
    {
        int alpha_ml = basis->getState(alpha)->getM();
        int alphaSpin = basis->getState(alpha)->getSpin();
        for (int beta = 0; beta < N_SPS; beta++)
        {
            int beta_ml = basis->getState(beta)->getM();
            int betaSpin = basis->getState(beta)->getSpin();

            // Spin and M conservation test
            if ((alpha_ml != beta_ml) || (alphaSpin != betaSpin)) { continue; }

            HFMatrix(alpha,beta) = calculateInnerHFMatrixElement(alpha, alpha_ml, alphaSpin, beta, beta_ml, betaSpin);
        }
        // Instead of an if-test inside the nested for-loop
        HFMatrix(alpha,alpha) += basis->getState(alpha)->getEnergyPolar();
    }
}

double HartreeFock::calculateInnerHFMatrixElement(int alpha, int alpha_ml, int alphaSpin, int beta, int beta_ml, int betaSpin)
{
    /*
     * Funciton for calculating the inner-most HF matrix elements.
     */
    double HFElement = 0;
    for (int gamma = 0; gamma < N_SPS; gamma++)
    {
        int gamma_ml = basis->getState(gamma)->getM();
        double gammaSpin = basis->getState(gamma)->getSpin();
        for (int delta = 0; delta < N_SPS; delta++)
        {
            int delta_ml = basis->getState(delta)->getM();
            double deltaSpin = basis->getState(delta)->getSpin();

            // Total spin and angular momentum conservation test
            if ((alpha_ml + gamma_ml == beta_ml + delta_ml) && (alphaSpin + gammaSpin == betaSpin + deltaSpin))
            {
                // Brute-forcey method, not removed the extra deltaFunction for spin
//                HFElement += densityMatrix(gamma,delta) * interactionMatrix[index(alpha, gamma, beta, delta, N_SPS)];
                HFElement += densityMatrix(gamma,delta) * sqrtOmega * interactionMatrix[index(alpha, gamma, beta, delta, N_SPS)];
            }
        }
    }
    return HFElement;
}

int HartreeFock::runHF(int maxHFIterations) // testOrthogonality if optional
{
    /*
     * Hartree-Fock algorithm. For loop will run till max HF iteration is reached, or we get a convergence in energies.
     */
    arma::vec oldEnergies = arma::zeros<arma::vec>(N_SPS);
    arma::vec singleParticleEnergies = arma::zeros<arma::vec>(N_SPS);
    arma::mat HFMatrix = arma::zeros<arma::mat>(N_SPS,N_SPS);

    // For timing functions
    double mainLoopTime = 0;
    double eigTime = 0;
    double minimaTime = 0;
    clock_t loopStart, loopFinish;
    clock_t eigStart, eigFinish;
    clock_t minimaStart, minimaFinish;

    for (int HFIteration = 0; HFIteration < maxHFIterations; HFIteration++)
    {
        // Setting up HFMatrix
        loopStart = clock();
        updateHFMatrix(HFMatrix);
        loopFinish = clock();

        // Finding eigenvalues & eigenvectors
        eigStart = clock();
        singleParticleEnergies.zeros();
        arma::eig_sym(singleParticleEnergies, C, HFMatrix); // dc faster for larger matrices, std default
        densityMatrix.zeros(); // Setting densityMatrix back to zeros only
        eigFinish = clock();

        // Updating the density matrix
        updateDensityMatrix();

        if (orthogonalityTest == true)
        {
            testCOrthogonality();
        }

        // When if we have convergence, rather than checking everyone, find the max element
        minimaStart = clock();
        if ((arma::sum(arma::abs(singleParticleEnergies - oldEnergies)))/N_SPS < lambda)
        {
            HFCounter = HFIteration;
            break;
        }
        else
        {
            oldEnergies = singleParticleEnergies;
        }
        minimaFinish = clock();

        // Summing up time spent
        mainLoopTime += ((loopFinish - loopStart)/((double)CLOCKS_PER_SEC));
        eigTime += ((eigFinish - eigStart)/((double)CLOCKS_PER_SEC));
        minimaTime += ((minimaFinish - minimaStart)/((double)CLOCKS_PER_SEC));
    }

    TimeForHFComp = (mainLoopTime + eigTime  +minimaTime) / (double) HFCounter;

    // Checking if maximum HF iterations have been reached
    if (HFCounter == 0)
    {
        HFCounter = maxHFIterations;
        cout << "    Max HF iterations reached." << endl;
    }

    // Printing out average time per loop element
//    cout << "Average time per main loop:                 " << setprecision(8) << mainLoopTime / (double) HFCounter   << " seconds" << endl;
//    cout << "Average time for solving eigenvalueproblem: " << setprecision(8) << eigTime / (double) HFCounter        << " seconds" << endl;
//    cout << "Average time per finding minima:            " << setprecision(8) << minimaTime / (double) HFCounter     << " seconds" << endl;

    SPS_Energies = singleParticleEnergies;
    //cout << "   HF computations terminated:\n";
    //cout << "   single Particle Energies:\n";
    //cout << singleParticleEnergies << endl;
    //cout << "   C Matrix (Matrix of Eigenvectors):\n";
    //cout << C << endl;
    return 0;
}

void HartreeFock::printHFMatrix(arma::mat HFMatrix)
{
    /*
     *  Small function for printing out the Hartree-Fock matrix
     */
    for (int alpha = 0; alpha < N_SPS; alpha++)
    {
        for (int beta = 0; beta < N_SPS; beta++)
        {
            double val = HFMatrix(alpha,beta);
            if (fabs(val) < (1e-15))
            {
                cout << setw(12) << 0;
            }
            else
            {
                cout << setw(12) << HFMatrix(alpha,beta);
            }
        }
    cout << endl;
    }
}

void HartreeFock::printSPEnergies()
{
    /*
     * Small function for printing out the single particle energies with their belonging quantum numbers
     */
    for (int alpha = 0; alpha < N_SPS; alpha++)
    {
        cout << "SP-Energy = " << SPS_Energies(alpha) << " N = " << basis->getState(alpha)->getN() << " M = " << basis->getState(alpha)->getM() << endl;
    }
    cout << endl;
}

void HartreeFock::writeToFile()
{
    /*
     * Function for writing eigenvectors and the egeinvalue to file
     * Setup:
     * eigenValue # eigenvector-elements
     */
    std::ofstream file;
    std::string filename = "output_NSPS" + std::to_string(N_SPS) + "_Electrons" + std::to_string(N_Electrons) + ".dat";
    file.open(filename);
    for (int i = 0; i < N_SPS; i++)
    {
        file << setprecision(8) << SPS_Energies(i) << " # ";
        for (int j = 0; j < N_SPS; j++)
        {
            file << setprecision(8) << C(i,j); // AM I FILLING IN EIGENVECTORS HERE?
        }
        file << endl;
    }
    file.close();
    cout << filename << " written" << endl;
}

void HartreeFock::ComHFEnergy(double &HFEnergyResults, int &HFIterationsResults, arma::mat &HFReturnMatrix, arma::vec &HFSPSEnergiesReturnVector)
{
    /*
     * Function for retrieving the Hartree-Fock ground state energy
     */
    double energy = 0;

    for (int i = 0; i < N_Electrons; i++)
    {
        energy += SPS_Energies(i);
        for (int j = 0; j<N_Electrons; j++)
        {
            for (int alpha = 0; alpha < N_SPS; alpha++)
            {
                for (int beta = 0; beta < N_SPS; beta++)
                {
                    for (int gamma = 0; gamma < N_SPS; gamma++)
                    {
                        for (int delta= 0; delta < N_SPS; delta++)
                        {
//                            energy += - 0.5 * C(alpha,i) * C(beta,i) * C(gamma,j) * C(delta,j) * interactionMatrix[index(alpha, gamma, beta, delta, N_SPS)];
                            energy += - 0.5 * C(alpha,i) * C(beta,i) * C(gamma,j) * C(delta,j) * sqrtOmega * interactionMatrix[index(alpha, gamma, beta, delta, N_SPS)];
                        }
                    }
                }
            }
        }
    }
    printf("HFIterations = %4d Electrons = %2d Shells = %2d Omega = %3.2f Energy = %3.6f \n", HFCounter, N_Electrons, basis->getMaxShell(), sqrtOmega*sqrtOmega, energy);
    HFEnergy = energy;
    HFIterationsResults = HFCounter;
    HFReturnMatrix = C;
    HFSPSEnergiesReturnVector = SPS_Energies;
}

void HartreeFock::testCOrthogonality()
{
    /*
     * Function for testing the orthogonality of the C-matrix. Sets orthogonality results to false if triggered.
     */
    if ((arma::sum(arma::sum(C.t() * C)) - N_SPS) > lambda)
    {
        orthogonalityResults = false;
        cout << "ERROR: Orthogonality not preserved." << endl;
    }
}

void HartreeFock::setOmega(double newOmega)
{
    basis->setOmega(newOmega);
    sqrtOmega = sqrt(newOmega);
}

double ** HartreeFock::HFBasis(){
    /*
     * Setting up the interaction matrix in HF basis
     */
    clock_t setupStart, setupFinish;
    setupStart = clock();

    //int Index;
    int N_virt = N_SPS-N_Electrons;
    double * Zero_Block = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
    double * First_Block = new double[(int) pow(N_virt,4)];
    double * Second_Block = new double[(int) pow(N_Electrons,4)];
    double * Third_Block = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
    double * FourthBlock = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
    double ** HFBlocks = new double*[5];

#pragma omp parallel sections
{
    #pragma omp section
    {
    int Index;
    //double * Zero_Block = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
    //Zeroth "Block" <ab|v|ij>, where a,b is virt. and i,j is occ.
    for(int a = 0; a < N_virt; a++){
        for(int b =0; b < a; b++){
            for(int i = 0; i < N_Electrons; i++){
                for(int j = 0; j < i; j++){
                    //Entering inner loops
                    double MatrixElement0 = 0.0;
                    for( int alpha = 0; alpha < N_SPS; alpha++){
                        for( int beta = 0; beta < N_SPS; beta++){
                            for (int gamma = 0; gamma < N_SPS; gamma++){
                                for( int delta = 0; delta < N_SPS; delta++){
                                    Index = index(alpha, beta, gamma, delta, N_SPS);
                                    if(interactionMatrix[Index] != 0){
                                        MatrixElement0 += C(alpha, a+N_Electrons)*C(beta, b+N_Electrons)
                                                *C(gamma, i)*C(delta, j)
                                                *interactionMatrix[Index];
                                    }
                                }
                            }
                        }
                    }
                    Zero_Block[N_virt*(N_virt*(N_Electrons*j+i)+b)+a] = MatrixElement0;
                    Zero_Block[N_virt*(N_virt*(N_Electrons*j+i)+a)+b] = -MatrixElement0;
                    Zero_Block[N_virt*(N_virt*(N_Electrons*i+j)+b)+a] = -MatrixElement0;
                    Zero_Block[N_virt*(N_virt*(N_Electrons*i+j)+a)+b] = MatrixElement0;
                }
            }
        }
    }

}
#pragma omp section
{
    int Index;
    //int N_virt = N_SPS-N_Electrons;
    //double * First_Block = new double[(int) pow(N_virt,4)];
    //First "Block" <ab|v|cd>, where a,b,c,d is virt.
    for(int a = 0; a < N_virt; a++){
        for(int b =0; b < a; b++){
            for(int c = 0; c < N_virt; c++){
                for(int d = 0; d < c; d++){
                    //Entering inner loops
                    double MatrixElement1 = 0.0;
                    for( int alpha = 0; alpha < N_SPS; alpha++){
                        for( int beta = 0; beta < N_SPS; beta++){
                            for (int gamma = 0; gamma < N_SPS; gamma++){
                                for( int delta = 0; delta < N_SPS; delta++){
                                    Index = index(alpha, beta, gamma, delta, N_SPS);
                                    if(interactionMatrix[Index] != 0){
                                        MatrixElement1 += C(alpha, a+N_Electrons)*C(beta, b+N_Electrons)
                                                *C(gamma, c+N_Electrons)*C(delta, d+N_Electrons)
                                                *interactionMatrix[Index];
                                    }
                                }
                            }
                        }
                    }
                    First_Block[N_virt*(N_virt*(N_virt*d+c)+b)+a] = MatrixElement1;
                    First_Block[N_virt*(N_virt*(N_virt*d+c)+a)+b] = -MatrixElement1;
                    First_Block[N_virt*(N_virt*(N_virt*c+d)+b)+a] = -MatrixElement1;
                    First_Block[N_virt*(N_virt*(N_virt*c+d)+a)+b] = MatrixElement1;
                }
           }
        }
    }

}
#pragma omp section
{
            int Index;
            //int N_virt = N_SPS-N_Electrons;
            //double * FourthBlock = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
           //Fourth "Block" <kl|v|cd>, where c,d are virt. and k,l are occ.
            for(int k = 0; k < N_Electrons; k++){
                for(int l =0; l < k; l++){
                    for(int c = 0; c < N_virt; c++){
                        for(int d = 0; d < c; d++){
                            //Entering inner loops
                            double MatrixElement4 = 0.0;
                            for( int alpha = 0; alpha < N_SPS; alpha++){
                                for( int beta = 0; beta < N_SPS; beta++){
                                    for (int gamma = 0; gamma < N_SPS; gamma++){
                                        for( int delta = 0; delta < N_SPS; delta++){
                                            Index = index(alpha, beta, gamma, delta, N_SPS);
                                            if(interactionMatrix[Index] != 0){
                                                MatrixElement4 += C(alpha, k)*C(beta, l)*C(gamma, c+N_Electrons)*C(delta, d+N_Electrons)
                                                        *interactionMatrix[Index];
                                            }
                                        }
                                    }
                                }
                            }
                            FourthBlock[N_Electrons*(N_Electrons*(N_virt*d+c)+l)+k] = MatrixElement4;
                            FourthBlock[N_Electrons*(N_Electrons*(N_virt*d+c)+k)+l] = -MatrixElement4;
                            FourthBlock[N_Electrons*(N_Electrons*(N_virt*c+d)+l)+k] = -MatrixElement4;
                            FourthBlock[N_Electrons*(N_Electrons*(N_virt*c+d)+k)+l] = MatrixElement4;
                        }
                    }
                }
            }

}
#pragma omp section
{
   int Index;
   //int N_virt = N_SPS-N_Electrons;
   //double * Third_Block = new double[(int) (pow(N_virt,2)*pow(N_Electrons,2))];
   //Third "Block" <kb|v|cj>, where b,c are virt. and k,j are occ
    for(int k = 0; k < N_Electrons; k++){
        for(int b =0; b < N_virt; b++){
            for(int c = 0; c < N_virt; c++){
                for(int j = 0; j < N_Electrons; j++){
                    //Entering inner loops
                    double MatrixElement3 = 0.0;
                    for( int alpha = 0; alpha < N_SPS; alpha++){
                        for( int beta = 0; beta < N_SPS; beta++){
                            for (int gamma = 0; gamma < N_SPS; gamma++){
                                for( int delta = 0; delta < N_SPS; delta++){
                                    Index = index(alpha, beta, gamma, delta, N_SPS);
                                    if(interactionMatrix[Index] != 0){
                                        MatrixElement3 += C(alpha, k)*C(beta, b+N_Electrons)*C(gamma, c+N_Electrons)*C(delta, j)
                                                *interactionMatrix[Index];
                                    }
                                }
                            }
                        }
                    }
                    Third_Block[N_Electrons*(N_virt*(N_virt*j+c)+b)+k] = MatrixElement3;
                }
            }
        }
    }

}
}

    int Index;
    //int N_virt = N_SPS-N_Electrons;
    //double * Second_Block = new double[(int) pow(N_Electrons,4)];
    //Second "Block" <kl|v|ij>, where k,l,i,j are occ.
    for(int k = 0; k < N_Electrons; k++){
        for(int l =0; l < k; l++){
            for(int i = 0; i < N_Electrons; i++){
                for(int j = 0; j < i; j++){
                    //Entering inner loops
                    double MatrixElement2 = 0.0;
                    for( int alpha = 0; alpha < N_SPS; alpha++){
                        for( int beta = 0; beta < N_SPS; beta++){
                            for (int gamma = 0; gamma < N_SPS; gamma++){
                                for( int delta = 0; delta < N_SPS; delta++){
                                    Index = index(alpha, beta, gamma, delta, N_SPS);
                                    if(interactionMatrix[Index] != 0){
                                        MatrixElement2 += C(alpha, k)*C(beta, l)*C(gamma, i)*C(delta, j)
                                                *interactionMatrix[Index];
                                    }
                                }
                            }
                        }
                    }
                    Second_Block[N_Electrons*(N_Electrons*(N_Electrons*j+i)+l)+k] = MatrixElement2;
                    Second_Block[N_Electrons*(N_Electrons*(N_Electrons*j+i)+k)+l] = -MatrixElement2;
                    Second_Block[N_Electrons*(N_Electrons*(N_Electrons*i+j)+l)+k] = -MatrixElement2;
                    Second_Block[N_Electrons*(N_Electrons*(N_Electrons*i+j)+k)+l] = MatrixElement2;
                }
            }
        }
    }

    HFBlocks[0] = Zero_Block;
    HFBlocks[1] = First_Block;
    HFBlocks[2] = Second_Block;
    HFBlocks[3] = Third_Block;
    HFBlocks[4] = FourthBlock;
    setupFinish = clock();
    TimeForHFBasisTrafo = ((setupFinish - setupStart)/((double)CLOCKS_PER_SEC));

    //cout << "Matrix Elements in HF basis complete. Computational time: " << TimeForHFBasisTrafo << endl;

    return HFBlocks;

    /*
    for(int p = 0; p < N_SPS; p++){
        //cout << "Iteration: " << p << endl;
        for(int q = 0; q < p; q++){
            //#pragma omp parallel for
            for(int r = 0; r < N_SPS; r++){
                for(int s = 0; s < r; s++){
                    double MatrixElement = 0.0;
                    for( int alpha = 0; alpha < N_SPS; alpha++){
                        for( int beta = 0; beta < N_SPS; beta++){
                            for (int gamma = 0; gamma < N_SPS; gamma++){
                                for( int delta = 0; delta < N_SPS; delta++){
                                    //CHECK the index in the interaction matrix
                                    Index = index(alpha, beta, gamma, delta, N_SPS);
                                    if(interactionMatrix[Index] != 0){
                                        MatrixElement += C(alpha, p)*C(beta, q)*C(gamma, r)*C(delta, s)
                                                *interactionMatrix[Index];
                                    }
                                }
                            }
                        }
                    }
                    HFInteractionMatrix[index(p,q,r,s,N_SPS)] = MatrixElement;
                    HFInteractionMatrix[index(q,p,r,s,N_SPS)] = -1*MatrixElement;
                    HFInteractionMatrix[index(p,q,s,r,N_SPS)] = -1*MatrixElement;
                    HFInteractionMatrix[index(q,p,s,r,N_SPS)] = MatrixElement;

                }
            }
        }
    }

    return HFInteractionMatrix;
    */
}

arma::vec HartreeFock::getSPS_Energies(){
    return SPS_Energies;
}

double HartreeFock::getHFEnergy(){
    return HFEnergy;
}


