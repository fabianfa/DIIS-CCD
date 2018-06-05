#include "ccd_brute_force.h"
#include <iostream>
#include <cstdio>
#include <ctime>
#include <math.h>

CCD_Brute_Force::CCD_Brute_Force(int number_of_electrons_, int number_of_shells_,
                                 double ** HFBlocks, double * fock_energies_)
{
    number_of_electrons = number_of_electrons_;
    number_of_shells = number_of_shells_;
    number_of_states = number_of_shells_ * number_of_shells_ + number_of_shells_;
    number_of_virt_orb = number_of_states - number_of_electrons;
    number_of_ccd_ampl = (int) (pow(number_of_virt_orb,2)*pow(number_of_electrons,2));

    ABIJ_Block = HFBlocks[0];
    ABCD_Block = HFBlocks[1];
    KLIJ_Block = HFBlocks[2];
    KBCJ_Block = HFBlocks[3];
    KLCD_Block = HFBlocks[4];
    TBME = HFBlocks[0];
    fock_energies = fock_energies_;

    ccd_amplitudes = new double[number_of_ccd_ampl];
    ccd_root = new double[number_of_ccd_ampl];

    for(int i = 0; i < number_of_ccd_ampl; i++){
        ccd_amplitudes[i] = 0;
        ccd_root[i] = 0;
    }

    ccd_root_norm = 1;
    correlation_energy = 0;
    MaxEvalTime = 0;
}

void CCD_Brute_Force::quasi_newton(double eps, int max_it){
    //start timing:
    std::clock_t start_ccd_function;
    start_ccd_function = std::clock();

    int iteration = 0;
    int Index = 0;


    while(ccd_root_norm>eps && iteration < max_it){
        //cout << "################################################################" << endl;
        //cout << "   Iteration: "<< iteration << endl;
        ccd_function();
        //printf("%-15s%-15s%-15s%-15s%-15s%-15s\n", "ccd ampl.", "<ab|v|ij>"
        //       ,"ccd ampl.old", "ccd root", "fock_energies", "Index");
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        iteration = iteration +1;
        //cout << "Norm of approximate solution: "<< ccd_root_norm << endl;
    }
    quasiNewIterations = iteration;
    corr_energy();
    //cout<< "Correlation energy="<< correlation_energy << endl;
    compTime = ( std::clock() - start_ccd_function ) / (double) CLOCKS_PER_SEC;

}

void CCD_Brute_Force::ccd_function(){
    //start timing:
    std::clock_t start_ccd_function;
    double duration_total;
    start_ccd_function = std::clock();

    int Index = 0;
    int aux_index1 = 0;
    int aux_index2 = 0;
    int aux_index3 = 0;
    double norm = 0;
    double aux_var = 0;
    double TBME_value;
    double first_sum;
    double second_sum;
    double third_sum;
    double fourth_sum;

    //CCD amplitude function:
    for(int a = 0; a < number_of_virt_orb; a++){
        for(int b = 0; b < number_of_virt_orb; b++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int j = 0; j < number_of_electrons; j++){
                   if (i > j && a > b){
                        Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;

                        //Two particle integrals:
                        ccd_root[Index] = ABIJ_Block[Index];

                        //Fock-Energies:
                        ccd_root[Index] += fock_energies[Index]*ccd_amplitudes[Index];

                        //virt. contributions for fixed i,j:
                        aux_var = 0;
                        for(int d = 0; d < number_of_virt_orb; d++){
                            for(int c = 0; c < number_of_virt_orb; c++){
                                aux_index1 = number_of_virt_orb*(number_of_virt_orb*(number_of_virt_orb*d+c)+b)+a;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+d)+c;
                                aux_var += ABCD_Block[aux_index1]
                                       *ccd_amplitudes[aux_index2];
                            }
                        }
                        ccd_root[Index] += 0.5*aux_var;

                        //occ. contributions for fixed a,b:
                        aux_var = 0;
                        for(int l = 0; l < number_of_electrons; l++){
                            for(int k = 0; k < number_of_electrons; k++){
                                aux_index1 = number_of_electrons*(number_of_electrons*(number_of_electrons*j+i)+l)+k;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+b)+a;
                                aux_var += KLIJ_Block[aux_index1]*ccd_amplitudes[aux_index2];
                            }
                        }
                        ccd_root[Index] += 0.5*aux_var;

                        //Mixed contribution single-exc. for fied b,j:
                        aux_var = 0;
                        for(int c = 0; c < number_of_virt_orb; c++){
                            for(int k = 0; k < number_of_electrons; k++){
                                //applying P(ij|ab)= Id - P_{ab}- P_{ij}+ P_{ij}P_{ab}
                                //Identity contribution:
                                aux_index1 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*j+c)+b)+k;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+a;
                                ccd_root[Index] += KBCJ_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                //P_{ab} contribution:
                                aux_index1 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*j+c)+a)+k;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+b;
                                ccd_root[Index] -= KBCJ_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                //P_{ij} contribution:
                                aux_index1 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*i+c)+b)+k;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+a;
                                ccd_root[Index] -= KBCJ_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                //P_{ij}P_{ab} contribution:
                                aux_index1 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*i+c)+a)+k;
                                aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+b;
                                ccd_root[Index] += KBCJ_Block[aux_index1]*ccd_amplitudes[aux_index2];
                            }
                        }
                        //ccd_root[Index] += aux_var;

                        //Double exc. contributions:
                        TBME_value = 0;
                        first_sum = 0;
                        second_sum = 0;
                        third_sum = 0;
                        fourth_sum = 0;

                        for(int k = 0; k < number_of_electrons; k++){
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){

                                        //First double exc. contribution:

                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+d)+c;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+b)+a;
                                        TBME_value = KLCD_Block[aux_index1];
                                        first_sum += TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                        //Second double exc. contribution:
                                        //Applying P(ij) = 1 - P_{ij}
                                        //Identity contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+j)+d)+b;
                                        second_sum += TBME_value *ccd_amplitudes[aux_index2] *ccd_amplitudes[aux_index3];

                                        //P_{ij} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+i)+d)+b;
                                        second_sum -= TBME_value *ccd_amplitudes[aux_index2] *ccd_amplitudes[aux_index3];

                                        //Third double exc. contribution:
                                        //Applying P(ij) = 1 - P_{ij}
                                        //Identity contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+d;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+l)+b)+a;
                                        third_sum += TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                        //P_{ij} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+d;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+l)+b)+a;
                                        third_sum -= TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                        //Fourth double exc. contribution:
                                        //Applying P(ab) = 1 - P_{ab}
                                        //Identity contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+l)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+d;
                                        fourth_sum +=TBME_value
                                                 *ccd_amplitudes[aux_index2]
                                                 *ccd_amplitudes[aux_index3];

                                        //P_{ab} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+l)+c)+b;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+a)+d;
                                        fourth_sum -= TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                    }
                                }
                            }
                        }
                        ccd_root[Index] += 0.25*first_sum + second_sum - 0.5*third_sum - 0.5*fourth_sum;
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+a)+b] = -ccd_root[Index];
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+b)+a] = -ccd_root[Index];
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+a)+b] = ccd_root[Index];
                        norm += 4*ccd_root[Index]*ccd_root[Index];

/*
                        //FOR OPENMP:
                        //First double exc. contribution:
                        for(int k = 0; k < number_of_electrons; k++){
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){

                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+d)+c;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+b)+a;
                                        TBME_value = KLCD_Block[aux_index1];
                                        first_sum += TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];
                                    }
                                }
                            }
                        }

                        //Second double exc. contribution:
                        for(int k = 0; k < number_of_electrons; k++){
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){

                                        //Applying P(ij) = 1 - P_{ij}
                                        //Identity contribution:

                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+j)+d)+b;
                                        TBME_value = KLCD_Block[aux_index1];
                                        second_sum += TBME_value *ccd_amplitudes[aux_index2] *ccd_amplitudes[aux_index3];

                                        //P_{ij} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+i)+d)+b;
                                        second_sum -= TBME_value *ccd_amplitudes[aux_index2] *ccd_amplitudes[aux_index3];
                                    }
                                }
                            }
                        }

                        //Third double exc. contribution:
                        for(int k = 0; k < number_of_electrons; k++){
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){

                                        //Applying P(ij) = 1 - P_{ij}
                                        //Identity contribution:

                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+d;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+l)+b)+a;
                                        TBME_value = KLCD_Block[aux_index1];
                                        third_sum += TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                        //P_{ij} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+d;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+l)+b)+a;
                                        third_sum -= TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];
                                    }
                                }
                            }
                        }

                        //Fourth double exc. contribution:
                        for(int k = 0; k < number_of_electrons; k++){
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){

                                        //Applying P(ab) = 1 - P_{ab}
                                        //Identity contribution:

                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+l)+c)+a;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+d;
                                        TBME_value = KLCD_Block[aux_index1];
                                        fourth_sum += TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                        //P_{ab} contribution:

                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+l)+c)+b;
                                        aux_index3 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+a)+d;
                                        fourth_sum -= TBME_value*ccd_amplitudes[aux_index2]*ccd_amplitudes[aux_index3];

                                    }
                                }
                            }
                        }
                        ccd_root[Index] += 0.25*first_sum + second_sum - 0.5*third_sum - 0.5*fourth_sum;
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+a)+b] = -ccd_root[Index];
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+b)+a] = -ccd_root[Index];
                        ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+a)+b] = ccd_root[Index];
                        norm += 4*ccd_root[Index]*ccd_root[Index];
*/

                    }
                }
            }
        }
    }

    ccd_root_norm = sqrt (norm);
    duration_total = ( std::clock() - start_ccd_function ) / (double) CLOCKS_PER_SEC;
    if (duration_total > MaxEvalTime)
        MaxEvalTime = duration_total;
    //std::cout<<"Time for ccd function evaluation: "<< duration_total <<'\n';

}

void CCD_Brute_Force::corr_energy(){
    double energy = 0;
    int Index1 = 0;
    int Index2 = 0;

    for(int j = 0; j < number_of_electrons; j++){
        for(int i = 0; i < number_of_electrons; i++){
            for(int b = 0; b < number_of_virt_orb; b++){
                for(int a = 0; a < number_of_virt_orb; a++){
                    Index1 =  number_of_electrons*(number_of_electrons*(number_of_virt_orb*b+a)+j)+i;
                    Index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                    energy += KLCD_Block[Index1]*ccd_amplitudes[Index2];
                    }
                }
            }
        }
    correlation_energy = 0.25*energy;
}

double CCD_Brute_Force::getCorrEnergy(){
    return correlation_energy;
}

