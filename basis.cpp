#include "basis.h"
#include <vector>

using std::vector;
using std::cout;
using std::endl;

Basis::Basis()
{

}

Basis::~Basis()
{
    // Destructor
    for (SingleState * state : statesArray)
    {
        delete state;
    }
    statesArray.clear();
}

void Basis::initializeBasis(int initMaxShell, double initOmega)
{
    // Retrieving all possible states up until a cutoff-shell
    omega = initOmega;
    maxShell = initMaxShell;

    for (int shellInt = 0; shellInt < maxShell; shellInt++) // Ensuring we are adding the right shell
    {
        for (int i = 0; i < maxShell; i++) // Running over variations of nx
        {
            for (int j = 0; j < maxShell; j++) // Running over variations of ny
            {
                for (double k = -1; k < 2; k += 2) // Running over the two possible spin configurations
                {
                    if (i+j == shellInt) // If-test to prevent counting outside of the shell
                    {
                        SingleState * state = new SingleState();
                        state->set(i, j, k, omega);
                        statesArray.push_back(state);
                    }
                }
            }
        }
    }
}

SingleState *Basis::getState(int state)
{
//    cout << statesArray << endl;
    return statesArray[state];
}

int Basis::getShellDegeneracy(int shellNumber)
{
    int degeneracy = 0;
    for (SingleState * state : statesArray)
    {
        if ((state->getN_x() + state->getN_y()) == (shellNumber - 1))
        {
            degeneracy++;
        }
    }
    return degeneracy;
}

int Basis::getTotalDegeneracy()
{
    return statesArray.size();
}

double Basis::getShellEnergy(int shellNumber)
{
    int energy = 0;
    for (SingleState * state : statesArray)
    {
        if ((state->getN_x() + state->getN_y()) == (shellNumber - 1))
        {
            energy += state->getEnergy();
        }
    }
    return energy;
}

double Basis::getTotalEnergy()
{
    double energy = 0;

    for (SingleState * state : statesArray)
    {
        energy += state->getEnergy();
    }

    return energy;
}

int Basis::getTotalParticleNumber()
{
    return statesArray.size();
}

int Basis::getShellParticles(int shellNumber)
{
    int particleNumber = 0;

    for (SingleState * state : statesArray)
    {
        if ((state->getN_x() + state->getN_y()) == (shellNumber - 1))
        {
            particleNumber++;
        }
    }
    return particleNumber;
}

void Basis::printBasisSize()
{
    cout << statesArray.size() << endl;
}

void Basis::printBasis(bool polar)
{
    for (SingleState * state : statesArray)
    {
        state->printSystem(polar);
    }
}

void Basis::setOmega(double newOmega)
{
    omega = newOmega;
    for (SingleState * state : statesArray)
    {
        state->setOmega(omega);
    }
}


