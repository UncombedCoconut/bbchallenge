// SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Please use decide_closed_tape_language_l2r.py as the reference implementation.
// This is a C++ translation with no attempt at clear presentation.

#include <bit>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

struct BinFA
{
    uint64_t n;
    std::unique_ptr<uint64_t[]> T;

    BinFA(uint64_t n): n{n}, T{std::make_unique<uint64_t[]>(2*n)} { }
};

struct binary_DFAs
{
    struct iterator {
        BinFA dfa;
        uint64_t states_used;
        std::unique_ptr<uint64_t[]> refs;
        bool dead;

        iterator(uint64_t n, bool dead=false): dfa{n}, states_used{1}, refs{std::make_unique<uint64_t[]>(n)}, dead{dead} { refs[0] = 2; }
        iterator(iterator const& ugh_cpp): dfa{ugh_cpp.dfa.n}, refs{std::make_unique<uint64_t[]>(ugh_cpp.dfa.n)}, dead{ugh_cpp.dead} { refs[0] = 2; }
        bool operator!=(iterator other) const { return dead != other.dead; }
        iterator& operator++();
        BinFA& operator*() { if (states_used < dfa.n) ++*this; return dfa; }
    };

    uint64_t n;

    binary_DFAs(uint64_t n): n{n} { }

    iterator begin() { return iterator(n); }
    iterator end() { return iterator(n, true); }
};

binary_DFAs::iterator& binary_DFAs::iterator::operator++()
{
    do {
        dead = true;
        for (uint64_t i = 2*states_used-1; i > 0; i--) {
            if (--refs[dfa.T[i]]) {
                if (dfa.T[i] < dfa.n-1) {
                    ++dfa.T[i];
                    ++refs[dfa.T[i]];
                    if (dfa.T[i] == states_used) {
                        ++states_used;
                        refs[0] += 2;
                    }
                    dead = false;
                    break;
                }
                else {
                    dfa.T[i] = 0;
                    ++refs[0];
                }
            }
            else {
                dfa.T[i] = 0;
                --refs[0];
                --states_used;
            }
        }
    } while (!dead && states_used < dfa.n);
    return *this;
}

typedef const uint8_t TM[30];

uint64_t step_NFA_mask(BinFA const& nfa, uint64_t mask, uint8_t bit)
{
    uint64_t out = 0;
    while (mask) {
        uint64_t lo_bit = mask & -mask;
        mask ^= lo_bit;
        out |= nfa.T[2*std::countr_zero(lo_bit) + bit];
    }
    return out;
}

bool test_zero_stacks(BinFA const& nfa, uint64_t initial_state=1)
{
    uint64_t old = 0, out = uint64_t{1} << initial_state;
    while (out != old) {
        old = out;
        out |= step_NFA_mask(nfa, out, 0);
    }
    return !!(out & 1);
}

BinFA right_half_tape_NFA(TM tm, BinFA const& dfa)
{
    auto nP = 5*dfa.n + 1;
    auto nfa = BinFA{nP};
    bool grew = true;
    nfa.T[0] = nfa.T[1] = 1<<0;

    // N. B.: This inlines the Python version's quotient_PDS() function.
    for (int s = 0; s < 5; s++)
        for (uint8_t b = 0; b < 2; b++) {
            uint8_t write = tm[6*s+3*b+0], move = tm[6*s+3*b+1], goto_p1 = tm[6*s+3*b+2];
            for (uint64_t q1 = 0; q1 < dfa.n; q1++) {
                if (!goto_p1) // HALT rule for s@b
                    nfa.T[2*(5*q1+s+1)+b] |= uint64_t{1} << 0;
                else if (!move) {  // RIGHT rule: [q1] s@b RHS => [δ(q1,write)] goto_p1@RHS
                    uint64_t q2 = dfa.T[2*q1+write];
                    nfa.T[2*(5*q1+s+1)+b] |= uint64_t{1} << (5*q2+goto_p1);
                }
            }
        }

    do {
        grew = false;
        for (int s = 0; s < 5; s++)
            for (uint8_t b = 0; b < 2; b++) {
                uint8_t write = tm[6*s+3*b+0], move = tm[6*s+3*b+1], goto_p1 = tm[6*s+3*b+2];
                if (goto_p1 && move)  // LEFT rule: [δ(q1,b1)] s@b RHS => [q1] goto_p1@b1 write RHS
                    for (uint64_t q1 = 0; q1 < dfa.n; q1++)
                        for (uint8_t b1 = 0; b1 < 2; b1++) {
                            uint64_t q2 = dfa.T[2*q1+b1];
                            auto new_Tjb = uint64_t{1} << (5*q1+goto_p1);
                            new_Tjb = step_NFA_mask(nfa, new_Tjb, b1);
                            new_Tjb = step_NFA_mask(nfa, new_Tjb, write);
                            new_Tjb |= nfa.T[2*(5*q2+s+1)+b];
                            if (nfa.T[2*(5*q2+s+1)+b] != new_Tjb) {
                                nfa.T[2*(5*q2+s+1)+b] = new_Tjb;
                                grew = true;
                            }
                        }
            }
    } while (grew);
    return nfa;
}

bool ctl_search(TM tm, uint64_t l_states_max)
{
    uint8_t mirror_tm[30];
    std::memcpy(mirror_tm, tm, 30);
    for (int i = 0; i < 10; i++)
        mirror_tm[3*i+1] ^= 1;

    for (uint64_t l_states = 1; l_states <= l_states_max; l_states++)
        for (bool mirrored: {false, true}) {
            for (auto& l_dfa: binary_DFAs(l_states)) {
                auto r_nfa = right_half_tape_NFA(mirrored ? mirror_tm : tm, l_dfa);
                if (!test_zero_stacks(r_nfa))
                    return true;
            }
        }
    return false;
}

int main(int argc, char **argv)
{
    uint64_t limit = 5;
    char const* db_path = "all_5_states_undecided_machines_with_global_header";
    for (int i = 1; i < argc; i++) {
        if (std::atoi(argv[i]))
            limit = std::atoi(argv[i]);
        else
            db_path = argv[i];
    }
    std::ifstream db(db_path, std::ifstream::binary);
    for (int seed = -1; db.good(); seed++) {
        uint8_t tm[30];
        db.read(reinterpret_cast<char*>(tm), 30);
        if (seed >= 0 && db.good())
            std::cout << seed << (ctl_search(tm, limit) ? ", infinite\n" : ", undecided\n");
    }
    db.close();
    return 0;
}
