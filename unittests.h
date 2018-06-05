#ifndef UNITTESTS_H
#define UNITTESTS_H

int runHermiteTimer(int N, double X);
int checkCorrectHermitePolynoms(int maxHermite);
void testIntegratorClass(int NIntPoints, double eps=1e-10);
void testIntegratorFunction(int NIntPoints, double eps=1e-10);
void testOrthogonality(int numberOfArguments, char *cmdLineArguments[]);
void testUnperturbedHF(int numberOfArguments, char *cmdLineArguments[]);
void testDegeneracy(int numberOfArguments, char *cmdLineArguments[]);

#endif // UNITTESTS_H
