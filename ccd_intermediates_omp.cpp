#include "ccd_intermediates_omp.h"
#include <iostream>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <armadillo>

using namespace arma;

CCD_Intermediates_OMP::CCD_Intermediates_OMP(int number_of_electrons_, int number_of_shells_,
                                     double **HFBlocks, double * fock_energies_)
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

    if (number_of_electrons >number_of_virt_orb)
        length_of_intermediates = pow(number_of_electrons,2);
    else
        length_of_intermediates = number_of_electrons * number_of_virt_orb;

    ccd_intermediates = new double[length_of_intermediates];
    ccd_intermediates1 = new double[length_of_intermediates];
    ccd_intermediates2 = new double[length_of_intermediates];
    ccd_intermediates3 = new double[length_of_intermediates];
    ccd_intermediates4 = new double[length_of_intermediates];

    for (int i = 0; i < length_of_intermediates; i++){
        ccd_intermediates[i] = 0;
        ccd_intermediates1[i] = 0;
        ccd_intermediates2[i] = 0;
        ccd_intermediates3[i] = 0;
        ccd_intermediates4[i] = 0;
    }

}


void CCD_Intermediates_OMP::DIIS(double eps, int max_it){
    //start timing:
    std::clock_t start_ccd_function;
    start_ccd_function = std::clock();

    int iteration = 0;
    int Index = 0;

    double scalar_product = 0;

    bool is_zero = false;

    mat B(6,6);
    mat pseudoInv;
    vec rhs = zeros<vec>(6);
    rhs(5) = 1;
    vec weights = zeros<vec>(6);
    vec precon;

    double * first_QN_it = new double[number_of_ccd_ampl];
    double * second_QN_it = new double[number_of_ccd_ampl];
    double * third_QN_it = new double[number_of_ccd_ampl];
    double * fourth_QN_it = new double[number_of_ccd_ampl];
    double * fifth_QN_it = new double[number_of_ccd_ampl];

    double * first_QN_step = new double[number_of_ccd_ampl];
    double * second_QN_step = new double[number_of_ccd_ampl];
    double * third_QN_step = new double[number_of_ccd_ampl];
    double * fourth_QN_step = new double[number_of_ccd_ampl];
    double * fifth_QN_step = new double[number_of_ccd_ampl];

    double ** QN_steps = new double*[5];

    while(ccd_root_norm>eps && iteration < max_it){
        cout << "################################################################" << endl;
        cout << "   DIIS-Iteration: "<< iteration << endl;

        //printf("%-15s%-15s%-15s%-15s%-15s%-15s\n", "ccd ampl.", "<ab|v|ij>"
        //      ,"ccd ampl.old", "ccd root", "fock_energies", "Index");

        //First quasi-Newton steps:
        //cout << "       First QN-Iteration"<< endl;
        ccd_function();
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            first_QN_it[Index] = ccd_amplitudes[Index];
                            first_QN_step[Index] = -ccd_root[Index]/fock_energies[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        QN_steps[0] = first_QN_step;

        //Second quasi-Newton steps:
        //cout << "       Second QN-Iteration"<< endl;
        ccd_function();
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            second_QN_it[Index] = ccd_amplitudes[Index];
                            second_QN_step[Index] = -ccd_root[Index]/fock_energies[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        QN_steps[1] = second_QN_step;

        //Third quasi-Newton steps:
        //cout << "       Third QN-Iteration"<< endl;
        ccd_function();
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            third_QN_it[Index] = ccd_amplitudes[Index];
                            third_QN_step[Index] = -ccd_root[Index]/fock_energies[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        QN_steps[2] = third_QN_step;

        //Fourth quasi-Newton steps:
        //cout << "       Fourth QN-Iteration"<< endl;
        ccd_function();
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            fourth_QN_it[Index] = ccd_amplitudes[Index];
                            fourth_QN_step[Index] = -ccd_root[Index]/fock_energies[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        QN_steps[3] = fourth_QN_step;

        //Fifth quasi-Newton steps:
        //cout << "       Fifth QN-Iteration"<< endl;
        ccd_function();
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index]= ccd_amplitudes[Index]-ccd_root[Index]/fock_energies[Index];
                            fifth_QN_it[Index] = ccd_amplitudes[Index];
                            fifth_QN_step[Index] = -ccd_root[Index]/fock_energies[Index];
                            //cout << fifth_QN_step[Index] << endl;
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }
        QN_steps[4] = fifth_QN_step;

        //Setting up the B matrix
        for(int i = 0 ; i < 6; i++ ){
            for(int j = i; j < 6; j++ ){
                if (j == 5){
                    B(i,j) = 1;
                } else {
                    scalar_product = 0;
                    for(int k = 0; k < number_of_ccd_ampl; k++){
                        scalar_product += QN_steps[i][k]*QN_steps[j][k];
                    }
                    B(i,j) = scalar_product;
                }
                B(j,i) = B(i,j);
            }

        }
        B(5,5) = 0;

        //Precondition the B matrix
        precon = zeros<vec>(6);

        is_zero = false;
        for (int i = 0; i < 5 ; i++){
            if (B(i,i) <= 0){
                is_zero = true;
            }
        }

        if (is_zero){
            for(int i = 0; i < 5; i++)
                precon(i) = 1;
        } else {
            for(int i = 0; i < 5; i++)
                precon(i) = pow(B(i,i),-0.5);
        }

        precon(5) = 1.0;
        for(int i = 0; i < 6; i++){
            for(int j = 0; j < 6; j++){
                B(i,j) = B(i,j)*precon(i)*precon(j);
            }
        }

        //Solving the linear system:
        pseudoInv = pinv(B);

        for(int i = 0; i < 5; i++){
            weights(i)= pseudoInv(5,i)*precon(i);
        }

        //DIIS step
        for(int j = 0; j < number_of_electrons; j++){
            for(int i = 0; i < number_of_electrons; i++){
                for(int b = 0; b < number_of_virt_orb; b++){
                    for(int a = 0; a < number_of_virt_orb; a++){
                            Index = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+b)+a;
                            //double old = ccd_amplitudes[Index];
                            ccd_amplitudes[Index] =
                                        weights(0)*first_QN_it[Index]
                                        + weights(1)*second_QN_it[Index]
                                        + weights(2)*third_QN_it[Index]
                                        + weights(3)*fourth_QN_it[Index]
                                        + weights(4)*fifth_QN_it[Index];
                            //printf("%-15.5e%-15.5e%-15.5e%-15.5e%-15.5e%-15d\n",ccd_amplitudes[Index]
                            //       , ABIJ_Block[Index],old, ccd_root[Index]
                            //        , fock_energies[Index], Index);
                    }
                }
            }
        }


        iteration = iteration +1;
        cout << "Norm of approximate solution: "<< ccd_root_norm << endl;
    }
    quasiNewIterations = iteration;
    corr_energy();
    //cout<< "Correlation energy="<< correlation_energy << endl;
    compTime = ( std::clock() - start_ccd_function ) / (double) CLOCKS_PER_SEC;

}


void CCD_Intermediates_OMP::quasi_newton(double eps, int max_it){
    //start timing:
    std::clock_t start_ccd_function;
    start_ccd_function = std::clock();

    int iteration = 0;
    int Index = 0;

    while(ccd_root_norm>eps && iteration < max_it){
        cout << "################################################################" << endl;
        cout << "   Iteration: "<< iteration << endl;
        ccd_function();
        //printf("%-15s%-15s%-15s%-15s%-15s%-15s\n", "ccd ampl.", "<ab|v|ij>"
        //      ,"ccd ampl.old", "ccd root", "fock_energies", "Index");
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
        cout << "Norm of approximate solution: "<< ccd_root_norm << endl;
    }
    quasiNewIterations = iteration;
    corr_energy();
    cout<< "Correlation energy="<< correlation_energy << endl;
    compTime = ( std::clock() - start_ccd_function ) / (double) CLOCKS_PER_SEC;
}

void CCD_Intermediates_OMP::ccd_function(){
    //start timing:
    std::clock_t start_ccd_function;
    double duration_total;
    start_ccd_function = std::clock();

    int Index = 0;
    int aux_index1 = 0;
    int aux_index2 = 0;
    double norm = 0;
    double aux_var = 0;
    double sum = 0;

    double id_cont = 0;
    double P_ab_cont = 0;
    double P_ij_cont = 0;
    double P_ij_P_ab_cont = 0;

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

                       //***************************************************//
                       //****************First intermediate:****************//
                       //***************************************************//

                       //Writing first intermediate
                       for(int l = 0; l < number_of_electrons; l++){
                           for(int k = 0; k < number_of_electrons; k++){
                                //Inner contraction:
                                sum = 0;
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+d)+c;
                                        sum += KLCD_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                    }
                                }
                                aux_index1 = number_of_electrons*(number_of_electrons*(number_of_electrons*j+i)+l)+k;
                                ccd_intermediates[number_of_electrons*l+k] = 0.5*sum + KLIJ_Block[aux_index1];
                           }
                       }
                       //Contraction:
                       aux_var = 0;
                       for(int l = 0; l < number_of_electrons; l++){
                           for(int k = 0; k < number_of_electrons; k++){
                               aux_index1 = number_of_electrons*l+k;
                               aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+b)+a;
                               aux_var += ccd_intermediates[aux_index1]*ccd_amplitudes[aux_index2];
                           }
                       }

                       ccd_root[Index] += 0.5*aux_var;

                       //***************************************************//
                       //****************Second intermediate:***************//
                       //***************************************************//

                       //applying P(ij|ab)= Id - P_{ab}- P_{ij}+ P_{ij}P_{ab}
                       //Identity contribution:

                       #pragma omp parallel sections
                       {
                       #pragma omp section
                       {
                       double sum1=0;
                       int aux_index3 = 0;
                       int aux_index4 = 0;
                       double aux_var1 = 0;

                       //Writing second intermediate
                       for(int c = 0; c < number_of_virt_orb; c++){
                            for(int k = 0; k < number_of_electrons; k++){
                            //Inner contraction:
                                sum1 = 0;
                                for(int l = 0; l < number_of_electrons; l++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index3 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+l)+b)+d;
                                        sum1 += KLCD_Block[aux_index3]*ccd_amplitudes[aux_index4];
                                    }
                                }
                                aux_index3 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*j+c)+b)+k;
                                ccd_intermediates1[number_of_electrons*c+k] = 0.5*sum1+ KBCJ_Block[aux_index3];
                            }
                        }

                       //Contraction:
                       for(int k = 0; k < number_of_electrons; k++){
                            for(int c = 0; c < number_of_virt_orb; c++){
                                aux_index3 = number_of_electrons*c+k;
                                aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+a;
                                aux_var1 += ccd_intermediates1[aux_index3]*ccd_amplitudes[aux_index4];
                            }
                       }
                       id_cont = aux_var1;
                       }


                       //P_{ab} contribution:
                        #pragma omp section
                        {
                        double sum1=0;
                        int aux_index3 = 0;
                        int aux_index4 = 0;
                        double aux_var1 = 0;

                       //Writing second intermediate
                       for(int c = 0; c < number_of_virt_orb; c++){
                            for(int k = 0; k < number_of_electrons; k++){
                            //Inner contraction:
                                sum1 = 0;
                                for(int l = 0; l < number_of_electrons; l++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index3 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+l)+a)+d;
                                        sum1 += KLCD_Block[aux_index3]*ccd_amplitudes[aux_index4];
                                    }
                                }
                                aux_index3 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*j+c)+a)+k;
                                ccd_intermediates2[number_of_electrons*c+k] = 0.5*sum1+ KBCJ_Block[aux_index3];
                            }
                        }

                       //Contraction:
                       for(int k = 0; k < number_of_electrons; k++){
                            for(int c = 0; c < number_of_virt_orb; c++){
                                aux_index3 = number_of_electrons*c+k;
                                aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+c)+b;
                                aux_var1 += ccd_intermediates2[aux_index3]*ccd_amplitudes[aux_index4];
                            }
                       }
                       P_ab_cont = aux_var1;
                       }
                       //P_{ij} contribution:
                       #pragma omp section
                       {
                       double sum1=0;
                       int aux_index3 = 0;
                       int aux_index4 = 0;
                       double aux_var1 = 0;
                       //Writing second intermediate
                       for(int c = 0; c < number_of_virt_orb; c++){
                            for(int k = 0; k < number_of_electrons; k++){
                            //Inner contraction:
                                sum1 = 0;
                                for(int l = 0; l < number_of_electrons; l++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index3 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+l)+b)+d;
                                        sum1 += KLCD_Block[aux_index3]*ccd_amplitudes[aux_index4];
                                    }
                                }
                                aux_index3 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*i+c)+b)+k;
                                ccd_intermediates3[number_of_electrons*c+k] = 0.5*sum1+ KBCJ_Block[aux_index3];
                            }
                        }

                       //Contraction:
                       for(int k = 0; k < number_of_electrons; k++){
                            for(int c = 0; c < number_of_virt_orb; c++){
                                aux_index3 = number_of_electrons*c+k;
                                aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+a;
                                aux_var1 += ccd_intermediates3[aux_index3]*ccd_amplitudes[aux_index4];
                            }
                       }
                       P_ij_cont = aux_var1;
                       }

                       //P_{ij}P_{ab} contribution:
                       #pragma omp section
                       {
                       double sum1=0;
                       int aux_index3 = 0;
                       int aux_index4 = 0;
                       double aux_var1 = 0;
                       //Writing second intermediate
                       for(int c = 0; c < number_of_virt_orb; c++){
                            for(int k = 0; k < number_of_electrons; k++){
                            //Inner contraction:
                                sum1 = 0;
                                for(int l = 0; l < number_of_electrons; l++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index3 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+l)+a)+d;
                                        sum1 += KLCD_Block[aux_index3]*ccd_amplitudes[aux_index4];
                                    }
                                }
                                aux_index3 = number_of_electrons*(number_of_virt_orb*(number_of_virt_orb*i+c)+a)+k;
                                ccd_intermediates4[number_of_electrons*c+k] = 0.5*sum1+ KBCJ_Block[aux_index3];
                            }
                        }

                       //Contraction:
                       for(int k = 0; k < number_of_electrons; k++){
                            for(int c = 0; c < number_of_virt_orb; c++){
                                aux_index3 = number_of_electrons*c+k;
                                aux_index4 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+c)+b;
                                aux_var1 += ccd_intermediates4[aux_index3]*ccd_amplitudes[aux_index4];
                            }
                       }
                       P_ij_P_ab_cont = aux_var1;
                       }
                       }
                       ccd_root[Index] += id_cont -P_ab_cont -P_ij_cont+P_ij_P_ab_cont;

                       //***************************************************//
                       //****************Third intermediate:****************//
                       //***************************************************//

                       //applying P(ij)= Id - P_{ij}
                       //Identity contribution:

                        //Writing third intermediate
                        for(int k = 0; k < number_of_electrons; k++){
                            sum = 0;
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+j)+d)+c;
                                        sum += KLCD_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                    }
                                }
                            }
                            ccd_intermediates[k] = sum;
                        }

                        //Contraction:
                        aux_var = 0;
                        for(int k = 0; k < number_of_electrons; k++){
                            aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+i)+b)+a;
                            aux_var += ccd_intermediates[k]*ccd_amplitudes[aux_index2];
                        }
                        ccd_root[Index] -= 0.5*aux_var;

                        //P_{ij} contribution:

                        //Writing third intermediate
                        for(int k = 0; k < number_of_electrons; k++){
                            sum = 0;
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int c = 0; c < number_of_virt_orb; c++){
                                    for(int d = 0; d < number_of_virt_orb; d++){
                                        aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                        aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+i)+d)+c;
                                        sum += KLCD_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                    }
                                }
                            }
                            ccd_intermediates[k] = sum;
                        }

                        //Contraction:
                        aux_var = 0;
                        for(int k = 0; k < number_of_electrons; k++){
                            aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*k+j)+b)+a;
                            aux_var += ccd_intermediates[k]*ccd_amplitudes[aux_index2];
                        }
                        ccd_root[Index] += 0.5*aux_var;

                         //***************************************************//
                         //****************Fourth intermediate:***************//
                         //***************************************************//

                         //applying P(ab)= Id - P_{ab}
                         //Identity contribution:

                         //Writing fourth intermediate
                         for(int c = 0; c < number_of_virt_orb; c++){
                            sum = 0;
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int k = 0; k < number_of_electrons; k++){
                                     for(int d = 0; d < number_of_virt_orb; d++){
                                         aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                         aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+d)+b;
                                         sum += KLCD_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                     }
                                 }
                             }
                             ccd_intermediates[c] = sum;
                         }

                         //Contraction:
                         aux_var = 0;
                         for(int c = 0; c < number_of_virt_orb; c++){
                             aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+c)+a;
                             aux_var += ccd_intermediates[c]*ccd_amplitudes[aux_index2];
                         }
                         ccd_root[Index] -= 0.5*aux_var;

                         //P_{ab} contribution:

                         //Writing fourth intermediate
                         for(int c = 0; c < number_of_virt_orb; c++){
                            sum = 0;
                            for(int l = 0; l < number_of_electrons; l++){
                                for(int k = 0; k < number_of_electrons; k++){
                                     for(int d = 0; d < number_of_virt_orb; d++){
                                         aux_index1 = number_of_electrons*(number_of_electrons*(number_of_virt_orb*d+c)+l)+k;
                                         aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*l+k)+d)+a;
                                         sum += KLCD_Block[aux_index1]*ccd_amplitudes[aux_index2];
                                     }
                                 }
                             }
                             ccd_intermediates[c] = sum;
                         }

                         //Contraction:
                         aux_var = 0;
                         for(int c = 0; c < number_of_virt_orb; c++){
                             aux_index2 = number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+c)+b;
                             aux_var += ccd_intermediates[c]*ccd_amplitudes[aux_index2];
                         }
                         ccd_root[Index] += 0.5*aux_var;

                         ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*j+i)+a)+b] = -ccd_root[Index];
                         ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+b)+a] = -ccd_root[Index];
                         ccd_root[number_of_virt_orb*(number_of_virt_orb*(number_of_electrons*i+j)+a)+b] = ccd_root[Index];
                         norm += 4*ccd_root[Index]*ccd_root[Index];


                   }
                }
            }
        }
    }

    ccd_root_norm = sqrt (norm);
    duration_total = ( std::clock() - start_ccd_function ) / (double) CLOCKS_PER_SEC;
    //std::cout<<"Time for ccd function evaluation: "<< duration_total <<'\n';
    if (duration_total > MaxEvalTime)
        MaxEvalTime = duration_total;
}


void CCD_Intermediates_OMP::corr_energy(){
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

double CCD_Intermediates_OMP::getCorrEnergy(){
    return correlation_energy;
}






















