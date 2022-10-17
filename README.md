# bbchallenge

A collection of tools for analyzing the Turing Machines of interest to the [Busy Beaver Challenge](https://bbchallenge.org/) project.

## Usage

The Python scripts in this repo are directly runnable (and support `--help`), if their dependencies are met.

They assume you want to use [PyPy](https://www.pypy.org); the decider scripts in particular are far too slow under the standard Python interpreter.

Once that's installed, initial setup might look like:
```
$ git submodule update --init
$ pypy3 -m ensurepip && pypy3 -m pip install -U pip wheel
$ pypy3 -m pip install automata-lib    numpy pygobject xdot
```
(This clones the standard `bbchallenge-py` project we use,
initializes PyPy's package management, and installs 3 groups of libraries -- for the deciders' regex output, and `interactive.py`'s visualization.)

### Deciders
* `decide_closed_tape_language_l2r.py` is the star of the show: it efficiently finds regular languages which can tell an eventually-halting TM configuration aprt from the initial one, by means explained by script docstrings and comments.
* `decide_closed_tape_language_native.cpp` reimplements the pure decision (`infinite`/`undecided`) in a stand-alone C++ program, which runs an order of magnitude faster. The code quality is not up to publishable standards.
* `dumb_dfa_decider.py` is an older and simpler decider, whose search is purely brute force.
* `coctl_party.py` is an experimental decider which starts by expressing the TM as a string rewriting system (PARTItioning the domain).
* `gnawndeterministic_scan.py` uses a SAT solver to construct an NFA with an even stronger non-halting condition, which is nonetheless true.

### Analyzers

I'll describe these briefly, but it's unknown whether they're worth learning about. The only machines they solve outright are simple cases like cyclers.

* `string_rewrite.py` is for exploring TMs as string rewriting systems, as in [this post](https://discuss.bbchallenge.org/t/7410754-does-not-halt/100).
* `interactive.py` furthermore diagrams the SRS with keyboard/mouse controls for operations that can split states or transitions piecewise.

## License

This work is dual-licensed under Apache 2.0 and MIT.
You can choose between one of them if you use this work.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

# Principles

## Overview
The powerful [Closed Tape Language](https://www.sligocki.com/2022/06/10/ctl.html) technique analyzes TM behavior using regular languages (those recognized by finite state machines).
Picture a Turing Machine as a two-stack machine: a fixed head pushes and pulls on two half-tapes.
If we split the left "stack" configurations into finitely many classes, and consider the transitions between (class, head, right-tape) tuples, we get a nondeterministic stack machine.
The following paper builds a finite state machine which recognizes configurations from which a stack machine can halt:

[[BEM97](https://www.irif.fr/~abou//BEM97.pdf)] Bouajjani, A., Esparza, J., & Maler, O. (1997, July). Reachability analysis of pushdown automata: Application to model-checking.
In International Conference on Concurrency Theory (pp. 135-150). Springer, Berlin, Heidelberg.

This makes the viewpoints interchangeable: regular CTLs classify left half-tapes (by the effect on its recognizer); the classification produces a stack machine; its exact solution is a CTL.
So motivated, we can study these modified TMs directly, and simplify away the pushdown-system formalism.


## Formal equivalence to CTL
Let L be a [Closed Tape Language](https://www.sligocki.com/2022/06/10/ctl.html) for some TM.

Fussy detail: formal languages have *finite* words, but TMs use infinitely zero-filled tapes.
There are many ways to reconcile this.
Let's take L to be a language on the alphabet of bits {0,1} and head-states {`A>`, ..., `E>`},
which is closed under TM transitions *and* invariant under zero-padding.

Let \~ be the [left syntactic equivalence](https://en.wikipedia.org/wiki/Syntactic_monoid#Syntactic_equivalence) relation it induces on bit-strings.

Let [u] denote the \~-equivalence class of a bit-string u, and v be another bit-string.

Define TM/\~ as a machine with configurations `[u] S> v`, and transitions `[u] S> v` -> [u'] S'> v' for each valid TM step 0̅ u S> v 0̅ -> 0̅ u S> v' 0̅.

Define halting for TM/\~ as for TM, and L(TM/\~) -- the language TM/\~ accepts -- to contain the configurations from which a halt is reachable.

When we view the "[u] S>" as states and the "v" as a stack, [BEM97] says TM/\~ is a "pushdown system" and L(TM/\~) is recognized by a certain finite automaton.

Thus, L' = { u S> v | TM/\~ won't accept [u] S> v 0^n for any n } is a regular language we can recognize.

L' is also a CTL: if it rejects u S> v after one step, then TM/\~ accepts it after one step (and zero-padding), so ditto before the one step, so L' rejects u S> v.

L' rejects halting words. If L does too, then L'⊇L, so we've recovered an equal or stronger CTL.


## Recognizer for TM/\~
[BEM97] is a nice black box, but it's simpler to recognize L(TM/\~) directly.

First, we need the "\~": fix a [DFA](https://en.wikipedia.org/wiki/Deterministic_finite_automaton) on the alphabet {0,1} for classifying left half-tapes.
(We only want its states and transition function δ. We require δ(q₀,0)=q₀; that is, it must ignore leading zeros.)
It shall classify words by which state they lead to.

(To be continued...)
