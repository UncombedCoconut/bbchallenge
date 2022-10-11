# bbchallenge

A collection of tools for analyzing the Turing Machines of interest to the [Busy Beaver Challenge](https://bbchallenge.org/) project.

## Usage

The Python scripts in this repo are directly runnable (and support `--help`), if their dependencies are met.

They assume you want to use [PyPy](https://www.pypy.org); the decider scripts in particular are far too slow under the standard Python interpreter.

Once that's installed, initial setup might look like:
```
$ git submodule update --init
$ pypy3 -m ensurepip && pypy3 -m pip install -U pip wheel
$ pypy3 -m pip install automata-lib    numpy pygobject xdot    pillow tabulate
```
(This clones the standard `bbchallenge-py` project we use,
initializes PyPy's package management,
and installs 3 groups of libraries -- to provide regex output, visualize TMs with `interactive.py`, and stop `bbchallenge-py` from crashing for no reason.)

### Deciders
* `decide_closed_tape_language_l2r.py` is the star of the show: it efficiently finds regular languages which can tell an eventually-halting TM configuration aprt from the initial one, by means explained by script docstrings and comments.
* `decide_closed_tape_language_native.cpp` reimplements the pure decision (`infinite`/`undecided`) in a stand-alone C++ program, which runs an order of magnitude faster. The code quality is not up to publishable standards.
* `dumb_dfa_decider.py` is an older and simpler decider, whose search is purely brute force.

### Analyzers

I'll describe these briefly, but it's unknown whether they're worth learning about. The only machines they solve outright are simple cases like cyclers.

* `string_rewrite.py` is for exploring TMs as string rewriting systems, as in [this post](https://discuss.bbchallenge.org/t/7410754-does-not-halt/100).
* `interactive.py` furthermore diagrams the SRS with keyboard/mouse controls for operations that can split states or transitions piecewise.

## License

This work is dual-licensed under Apache 2.0 and MIT.
You can choose between one of them if you use this work.

`SPDX-License-Identifier: Apache-2.0 OR MIT`
