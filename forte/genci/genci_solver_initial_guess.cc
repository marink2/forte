/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_functions.hpp"
#include "sparse_ci/ci_spin_adaptation.h"

#include "genci_solver.h"
#include "genci_vector.h"
#include "genci_string_lists.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "helpers/determinant_helpers.h"
#include "genci_string_address.h"
#include <cmath>
#include "helpers/timer.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <future>

#include "sparse_ci/sparse_initial_guess.h"

namespace forte {

std::vector<Determinant> GenCISolver::initial_guess_generate_dets(std::shared_ptr<psi::Vector> diag,
                                                                  size_t num_guess_states) {
    size_t ndets = diag->dim();
    // number of guess to be used must be at most as large as the number of determinants
    size_t num_guess_dets = num_guess_states * ndets_per_guess_;

    // Get the address of the most important determinants
    // this list has size exactly num_guess_dets
    double emax = std::numeric_limits<double>::max();
    size_t added = 0;
    size_t rejected = 0;

    std::vector<std::tuple<double, size_t>> vec_e_I(num_guess_dets, std::make_tuple(emax, 0));

    for (size_t I = 0; I < ndets; ++I) {
        double e = diag->get(I);

        if (core_guess_) {
            bool core_add = false;
            int core_add_count = 0;
            for (int c=0; c < core_bits_.size(); c++) {
                int p = core_bits_[c];
                bool alfa = lists_->determinant(I).get_alfa_bit(p);
                bool beta = lists_->determinant(I).get_beta_bit(p);
                core_add = (not (alfa and beta) and not (not alfa and not beta));
                if (core_add) {
                    core_add_count++;
                }
            }
            if (not (core_add_count == 1)) {
                rejected++;
                continue;
            }
        }

        // Find where to inser this determinant
        if ((e < emax) or (added < num_guess_dets)) { //num_guess_states
            vec_e_I.pop_back();
            auto it = std::find_if(
                vec_e_I.begin(), vec_e_I.end(),
                [&e](const std::tuple<double, size_t>& t) { return e < std::get<0>(t); });
            vec_e_I.insert(it, std::make_tuple(e, I));
            emax = std::get<0>(vec_e_I.back());
            added++;
        }
    }


    int count = 0;
    std::vector<Determinant> guess_dets;
    for (const auto& [e, I] : vec_e_I) {
        if (count >= (num_guess_dets)) {
            break;
        }
        const auto& det = lists_->determinant(I);
        guess_dets.push_back(det);
        count++;
    }

    psi::outfile->Printf("\n\n  DL Initial Guess Parameters");
    psi::outfile->Printf("\n  ---------------------------------");
    psi::outfile->Printf("\n  number of det: %zu", ndets);
    psi::outfile->Printf("\n  number of det added: %zu", added);
    psi::outfile->Printf("\n  number of det rejected: %zu", rejected);
    psi::outfile->Printf("\n  number of det selected: %zu", guess_dets.size());
    psi::outfile->Printf("\n  ---------------------------------");

    if (core_print_) {
        psi::outfile->Printf("\n\n  Determinants Selected as Initial Guess");
        psi::outfile->Printf("\n  %6s %14.9s %24s", "I", "e", "det");
        psi::outfile->Printf("\n  ------------------------------------------------------------------");
        for (const auto& [e, I] : vec_e_I) {
            const auto& det = lists_->determinant(I);
            psi::outfile->Printf("\n  %6d %14.9f %24s", I, e, str(det, 18).c_str());
        }
        psi::outfile->Printf("\n  ------------------------------------------------------------------");
    }

    // Make sure that the spin space is complete
    enforce_spin_completeness(guess_dets, active_mo_.size());
    if (guess_dets.size() > num_guess_dets) {
        if (print_ >= PrintLevel::Brief) {
            psi::outfile->Printf("\n  Initial guess space is incomplete.\n  Adding "
                                 "%d determinant(s).",
                                 guess_dets.size() - num_guess_dets);
        }
    }
    return guess_dets;
}

std::pair<sparse_mat, sparse_mat>
GenCISolver::initial_guess_det(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    auto guess_dets = initial_guess_generate_dets(diag, num_guess_states);
    size_t num_guess_dets = guess_dets.size();

    std::vector<size_t> guess_dets_pos(num_guess_dets);
    for (size_t I = 0; I < num_guess_dets; ++I) {
        guess_dets_pos[I] = lists()->determinant_address(guess_dets[I]);
    }

    // here we use a standard guess procedure
    return find_initial_guess_det(guess_dets, guess_dets_pos, num_guess_states, fci_ints,
                                  state().multiplicity(), true, print_ >= PrintLevel::Default,
                                  std::vector<std::vector<std::pair<size_t, double>>>(),
                                  core_guess_);
}

sparse_mat GenCISolver::initial_guess_csf(std::shared_ptr<psi::Vector> diag,
                                          size_t num_guess_states) {
    return find_initial_guess_csf(diag, num_guess_states, state().multiplicity(),
                                  print_ >= PrintLevel::Default);
}

std::shared_ptr<psi::Vector>
GenCISolver::form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                            std::shared_ptr<SpinAdapter> spin_adapter) {
    auto Hdiag_csf = std::make_shared<psi::Vector>(spin_adapter->ncsf());
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    double E0 = fci_ints->nuclear_repulsion_energy() + fci_ints->scalar_energy();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner_) {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter_->csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * fci_ints->energy(dets_[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      fci_ints->slater_rules(dets_[det_add_I], dets_[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_csf->set(i, energy);
        }
    } else {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                energy += c_I * c_I * fci_ints->energy(dets_[det_add_I]);
            }
            Hdiag_csf->set(i, energy);
        }
    }
    return Hdiag_csf;
}

std::shared_ptr<psi::Vector>
GenCISolver::form_Hdiag_det(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    const double E0 = fci_ints->nuclear_repulsion_energy() + fci_ints->scalar_energy();
    GenCIVector Hdiag(lists_);
    Determinant I;
    Hdiag.for_each_element([&](const size_t& /*n*/, const int& class_Ia, const int& class_Ib,
                               const size_t& Ia, const size_t& Ib, double& c) {
        I.set_str(lists_->alfa_str(class_Ia, Ia), lists_->beta_str(class_Ib, Ib));
        c = E0 + fci_ints->energy(I);
    });

    size_t ndets = Hdiag.size();

    auto Hdiag_det = std::make_shared<psi::Vector>(nfci_dets_);
    Hdiag.copy_to(Hdiag_det);

    if (print_ham_od_ and core_guess_) {

        // Form vector that orders determinant's indecis (n) in accending energy (e), 
        // and segregated by core occupations (c = 2: docc, 1:socc, 0:unocc)
        std::vector<std::tuple<double, size_t, size_t>> vec_e_n_c;
        for (size_t n = 0; n < ndets; n++) {
            double e = Hdiag_det->get(n);
            size_t c = 0;
            c += lists_->determinant(n).get_alfa_bit(0) ? 1 : 0;
            c += lists_->determinant(n).get_beta_bit(0) ? 1 : 0; 

            vec_e_n_c.emplace_back(e, n, c);
        }

        std::sort(vec_e_n_c.begin(), vec_e_n_c.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return std::tie(std::get<2>(lhs), std::get<0>(rhs)) > std::tie(std::get<2>(rhs), std::get<0>(lhs));
                  });

        // Form vector of newly ordered determinants (det_space) 
        // and get (nu) indeces for inital and final core det
        size_t nu = 0;
        size_t c_nu_i = 0;
        size_t c_nu_f = 0;
        std::vector<Determinant> det_space;
        for (const auto& [e, n, c] : vec_e_n_c) {
            det_space.emplace_back(lists_->determinant(n));
            c_nu_i += (c == 2);
            c_nu_f += (c == 1 || c == 2); 
            nu++;
        }
        c_nu_f--;

        // Print Hamiltonian matrix in the new (nu) ordered determinant sapce and 
        // find off-diagonal element with largest abs value (Hod_max)
        auto Hod_max = std::numeric_limits<double>::min();
        
        local_timer ham_t1;
        auto H_mat = make_hamiltonian_matrix(det_space, fci_ints);
        psi::outfile->Printf("\n Time for building Hamiltonian: %.4fs", ham_t1.get());

        local_timer ham_t2;
        for (size_t i = 0; i < ndets; i++) {
            for (size_t j = (i+1); j < ndets; j++) {
                if (abs(H_mat->get(i,j)) > Hod_max) {
                    Hod_max = abs(H_mat->get(i,j));
                }
            }
        }
        psi::outfile->Printf("\n Time for Hod max: %.4fs", ham_t2.get());

        local_timer ham_t3;
        if (print_ham_) {
            H_mat->print("matrix.dat");
        }
        psi::outfile->Printf("\n Time for printing Hmat: %.4f", ham_t3.get());

        // Get all Hmat coloumn elements at the first core-determinant index (c_nu_i) that are above a user threshold (coup_H_thrs_)
        std::vector<std::tuple<size_t, size_t, double, int, double, std::string>> vec_Hod_info;
        for (size_t j = 0; j < ndets; j++) {
            double MPEcj = pow(H_mat->get(c_nu_i,j), 2) / (H_mat->get(c_nu_i,c_nu_i) - H_mat->get(j,j));
            if (abs(MPEcj) >= coup_H_thrs_){
                vec_Hod_info.emplace_back(std::make_tuple(j, std::get<1>(vec_e_n_c[j]), std::get<0>(vec_e_n_c[j]),
                                          std::get<2>(vec_e_n_c[j]), MPEcj, str(det_space[j], 18)));
            }
        }

        // Print off-diagonal info
        psi::outfile->Printf("\n\n    nu        n            Hjj    occ          MPEcj               |Phi>_j");
        psi::outfile->Printf("\n  -----------------------------------------------------------------------------");
        for (const auto& [nu, n, Hjj, occ, MPEcj, phi] : vec_Hod_info) {
            psi::outfile->Printf("\n  %6d %6d %14.9f %6d %14.9f %24s", nu, n, Hjj, occ, MPEcj, phi.c_str());
        }
        psi::outfile->Printf("\n  -----------------------------------------------------------------------------");
        psi::outfile->Printf("\n  first core nu: %zu", c_nu_i);
        psi::outfile->Printf("\n  last  core nu: %zu", c_nu_f);
        psi::outfile->Printf("\n  Hod_max: %.9f", Hod_max);
        psi::outfile->Printf("\n  -----------------------------------------------------------------------------\n");
    }

    // Determinant I;
    // size_t Iadd = 0;
    // // loop over all irreps of the alpha strings
    // for (int ha = 0; ha < nirrep_; ha++) {
    //     const int hb = ha ^ symmetry_;
    //     const auto& sa = lists_->alfa_strings()[ha];
    //     const auto& sb = lists_->beta_strings()[hb];
    //     for (const auto& Ia : sa) {
    //         for (const auto& Ib : sb) {
    //             I.set_str(Ia, Ib);
    //             Hdiag_det->set(Iadd, E0 + fci_ints->energy(I));
    //             Iadd += 1;
    //         }
    //     }
    // }
    return Hdiag_det;
}

} // namespace forte
