///*
// *  wavefunction_hamiltonian.cpp
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/9/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

#include <boost/timer.hpp>

#include <libqt/qt.h>

#include "wavefunction.h"

namespace psi{ namespace libadaptive{

/**
 * Apply the Hamiltonian to the wave function
 * @param result Wave function object which stores the resulting vector
 */
void FCIWfn::Hamiltonian(FCIWfn& result,RequiredLists required_lists)
{
//    check_temp_space();
    result.zero();

    // H0
    {
        H0(result);
    }
    // H1_aa
    { boost::timer t;
        H1(result,true);
        h1_aa_timer += t.elapsed();
    }
    // H1_bb
    { boost::timer t;
        H1(result,false);
        h1_bb_timer += t.elapsed();
    }
    // H2_aabb
    { boost::timer t;
        H2_aabb(result);
        h2_aabb_timer += t.elapsed();
    }
    // H2_aaaa
    { boost::timer t;
        H2_aaaa2(result,true);
        h2_aaaa_timer += t.elapsed();
    }
    // H2_bbbb
    { boost::timer t;
        H2_aaaa2(result,false);
        h2_bbbb_timer += t.elapsed();
    }
}


/**
 * Apply the scalar part of the Hamiltonian to the wave function
 */
void FCIWfn::H0(FCIWfn& result)
{
    double core_energy = scalar_energy_ + ints_->frozen_core_energy();
    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        result.C_[alfa_sym]->copy(C_[alfa_sym]);
        result.C_[alfa_sym]->scale(core_energy);
    }
}

/**
 * Apply the one-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::H1(FCIWfn& result, bool alfa)
{
    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        int beta_sym = alfa_sym ^ symmetry_;
        if(detpi_[alfa_sym] > 0){
            SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            SharedMatrix Y = alfa ? result.C_[alfa_sym] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

            if(!alfa){
                C->zero();
                Y->zero();
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** C0h = C_[alfa_sym]->pointer();

                // Copy C0 transposed in C1
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(beta_sym) : alfa_graph_->strpi(alfa_sym);

            for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
                int q_sym = p_sym;  // Select the totat symmetric irrep
                for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                    for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                        int p_abs = p_rel + cmopi_offset_[p_sym];
                        int q_abs = q_rel + cmopi_offset_[q_sym];

                        double Hpq = alfa ? oei_aa(p_abs,q_abs) : oei_bb(p_abs,q_abs); // Grab the integral
                        std::vector<StringSubstitution>& vo = alfa ? lists_->get_alfa_vo_list(p_abs,q_abs,alfa_sym)
                                                                   : lists_->get_beta_vo_list(p_abs,q_abs,beta_sym);
                        // TODO loop in a differen way
                        int maxss = vo.size();

                        for(int ss = 0; ss < maxss; ++ss){
#if CAPRICCIO_USE_DAXPY
                            C_DAXPY(maxL,static_cast<double>(vo[ss].sign) * Hpq, &(Ch[vo[ss].I][0]), 1, &(Yh[vo[ss].J][0]), 1);
#else
                            double H = static_cast<double>(vo[ss].sign) * Hpq;
                            double* y = &Y[vo[ss].J][0];
                            double* c = &C[vo[ss].I][0];
                            for(size_t L = 0; L < maxL; ++L)
                                y[L] += c[L] * H;
#endif
                        }
                    }
                }
            }
            if(!alfa){
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** HC = result.C_[alfa_sym]->pointer();
                // Add Y1 transposed to Y
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
            }
        }
    } // End loop over h
}



/**
 * Apply the same-spin two-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::H2_aaaa2(FCIWfn& result, bool alfa)
{
    // Notation
    // ha - symmetry of alpha strings
    // hb - symmetry of beta strings
    for(int ha = 0; ha < nirrep_; ++ha){
        int hb = ha ^ symmetry_;
        if(detpi_[ha] > 0){
            SharedMatrix C = alfa ? C_[ha] : C1;
            SharedMatrix Y = alfa ? result.C_[ha] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

            if(!alfa){
                C->zero();
                Y->zero();
                size_t maxIa = alfa_graph_->strpi(ha);
                size_t maxIb = beta_graph_->strpi(hb);

                double** C0h = C_[ha]->pointer();

                // Copy C0 transposed in C1
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(hb) : alfa_graph_->strpi(ha);
            // Loop over (p>q) == (p>q)
            for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
                size_t max_pq = lists_->pairpi(pq_sym);
                for(size_t pq = 0; pq < max_pq; ++pq){
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;

                    double integral = alfa ? tei_aaaa(p_abs,q_abs,p_abs,q_abs) : tei_bbbb(p_abs,q_abs,p_abs,q_abs);

                    std::vector<StringSubstitution>& OO = alfa ? lists_->get_alfa_oo_list(pq_sym,pq,ha)
                                                               : lists_->get_beta_oo_list(pq_sym,pq,hb);

                    size_t maxss = OO.size();
                    for(size_t ss = 0; ss < maxss; ++ss)
                        C_DAXPY(maxL,static_cast<double>(OO[ss].sign) * integral, &(C->pointer()[OO[ss].I][0]), 1, &(Y->pointer()[OO[ss].J][0]), 1);
                }
            }
            // Loop over (p>q) > (r>s)
            for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
                size_t max_pq = lists_->pairpi(pq_sym);
                for(size_t pq = 0; pq < max_pq; ++pq){
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;
                    for(size_t rs = 0; rs < pq; ++rs){
                        const Pair& rs_pair = lists_->get_nn_list_pair(pq_sym,rs);
                        int r_abs = rs_pair.first;
                        int s_abs = rs_pair.second;
                        double integral = alfa ? tei_aaaa(p_abs,q_abs,r_abs,s_abs) : tei_bbbb(p_abs,q_abs,r_abs,s_abs);

                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(p_abs,q_abs,r_abs,s_abs,ha)
                                                                         : lists_->get_beta_vvoo_list(p_abs,q_abs,r_abs,s_abs,hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(r_abs,s_abs,p_abs,q_abs,ha)
                                                                         : lists_->get_beta_vvoo_list(r_abs,s_abs,p_abs,q_abs,hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                    }
                }
            }
            if(!alfa){
                size_t maxIa = alfa_graph_->strpi(ha);
                size_t maxIb = beta_graph_->strpi(hb);

                double** HC = result.C_[ha]->pointer();

                // Add Y1 transposed to Y
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
            }
        }
    } // End loop over h
}


/**
 * Apply the different-spin component of two-particle Hamiltonian to the wave function
 */
void FCIWfn::H2_aabb(FCIWfn& result)
{
    // Loop over blocks of matrix C
    for(int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym){
        size_t maxIa = alfa_graph_->strpi(Ia_sym);
        int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for(int rs_sym = 0; rs_sym < nirrep_; ++rs_sym){
            int Ja_sym = Ia_sym ^ rs_sym;
            size_t maxJa = alfa_graph_->strpi(Ja_sym);
            double** Y = result.C_[Ja_sym]->pointer();
            for(int r_sym = 0; r_sym < nirrep_; ++r_sym){
                int s_sym = rs_sym ^ r_sym;

                for(int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel){
                    for(int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel){
                        int r_abs = r_rel + cmopi_offset_[r_sym];
                        int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,Ib_sym)
                        std::vector<StringSubstitution>& vo_beta = lists_->get_beta_vo_list(r_abs,s_abs,Ib_sym);
                        size_t maxSSb = vo_beta.size();

                        C1->zero();
                        Y1->zero();

                        // Gather cols of C into C1
                        for(size_t Ia = 0; Ia < maxIa; ++Ia){
                            if(maxSSb > 0){
                                double* c1 = &(C1->pointer()[Ia][0]);//&C1[Ia][0];
                                double* c  = &(C[Ia][0]);
                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                    c1[SSb] = c[vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
                                }
                            }
                        }


                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
                            int q_sym = pq_sym ^ p_sym;
                            for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                                    int q_abs = q_rel + cmopi_offset_[q_sym];
                                    // Grab the integral
                                    double integral = tei_aabb(p_abs,r_abs,q_abs,s_abs);

                                    std::vector<StringSubstitution>& vo_alfa = lists_->get_alfa_vo_list(p_abs,q_abs,Ia_sym);

                                    // ORIGINAL CODE
                                    size_t maxSSa = vo_alfa.size();
                                    for(size_t SSa = 0; SSa < maxSSa; ++SSa){
#if CAPRICCIO_USE_DAXPY
                                        C_DAXPY(maxSSb,integral * static_cast<double>(vo_alfa[SSa].sign),
                                                &(C1->pointer()[vo_alfa[SSa].I][0]), 1, &(Y1->pointer()[vo_alfa[SSa].J][0]), 1);
#else
                                        double V = integral * static_cast<double>(vo_alfa[SSa].sign);
                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                            Y1[vo_alfa[SSa].J][SSb] += C1[vo_alfa[SSa].I][SSb] * V;
                                        }
#endif
                                    }
                                }
                            }
                        } // End loop over p,q
                        // Scatter cols of Y1 into Y
                        for(size_t Ja = 0; Ja < maxJa; ++Ja){
                            if(maxSSb > 0){
                                double* y = &Y[Ja][0];
                                double* y1 = &(Y1->pointer()[Ja][0]);
                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                    y[vo_beta[SSb].J] += y1[SSb];
                                }
                            }
                        }

                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
}

}}