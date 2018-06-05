#include "singlestate.h"

using std::cout;
using std::endl;

SingleState::SingleState()
{
    n_x = 0;
    n_y = 0;
    spin = 0;
}


SingleState::SingleState(int initN_x, int initN_y, double initSpin, double initOmega)
{
    n_x = initN_x;
    n_y = initN_y;
    spin = initSpin;
    omega = initOmega;
}

void SingleState::printSystem(bool polar=false)
{
    if (polar)
    {
        cout << "n   = " << n   << ", m   = " << m   << ", spin = " << spin << ", energy = " << getEnergy() << endl;
    }
    else
    {
        cout << "n_x = " << n_x << ", n_y = " << n_y << ", spin = " << spin << ", energy = " << getEnergy() << endl;
    }
}

void SingleState::set(int newN_x, int newN_y, double newSpin, double omega)
{
    setN_x(newN_x);
    setN_y(newN_y);
    setSpin(newSpin);
    setOmega(omega);
    setPolar(newN_x, newN_y);
}

void SingleState::setPolar(int newN_x, int newN_y)
{
    if (newN_x <= newN_y)
    {
        n = newN_x;
    }
    else
    {
        n = newN_y;
    }
    m = newN_x - newN_y;
//    if (newN_x == newN_y)
//    {
//        n = newN_x;
//        m = newN_x - newN_y;
//    }
//    else if (newN_x > newN_y)
//    {
//        n = newN_y;
//        m = newN_x - newN_y;

//    }
//    else
//    {
//        n = newN_x;
//        m = newN_x - newN_y;
//    }
}

std::ostream &operator<<(std::ostream &os, const SingleState &state) // Allows cout << myVector << endl;
{
    os << "n = " << state.getN() << ", m  = " << state.getM() << ", spin = " << state.getSpin() << ", energy = " << state.getEnergy();
    return os;
}

//std::ostream &operator<<(std::ostream &os, const vec3 &myVector) // Allows cout << myVector << endl;
//{
//    os << "[" << myVector.x() << ", " << myVector.y() << ", " << myVector.z() << "];";
//    return os;
//}
