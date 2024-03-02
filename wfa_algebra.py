# SPDX-FileCopyrightText: 2024 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from math import gcd, lcm
import re

class Poset:
    def __ge__(self, other): return self.__eq__(other) or not self.__le__(other)
    def __lt__(self, other): return not other.__le__(self)
    def __gt__(self, other): return not self.__le__(other)


class Semiring:
    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out = self.one()
        pi = 1
        powi = self
        while n:
            if n & pi:
                out = out * powi
                n ^= pi
                if not n: break
            pi *= 2
            powi = powi * powi
        return out


class BoolPoly(Poset, Semiring):
    def __init__(self, var_names, monos=()):
        self.var_names = tuple(var_names)
        self.monos = set(monos)
    def zero(self):
        return self.__class__(self.var_names)
    def one(self):
        return self.__class__(self.var_names, ((0,)*len(self.var_names),))

    def __repr__(self):
        return f'{self.__class__.__name__}{(self.var_names, self.monos)}'

    def __str__(self):
        return ' + '.join(map(self._mono_name, sorted(self.monos)))
    def _mono_name(self, mono):
        mono_names = (_pow_name(var_name, power) for var_name, power in zip(self.var_names, mono))
        return '*'.join(filter(None, mono_names)) or '1'

    @classmethod
    def from_text(cls, text):
        raise NotImplementedError  # Will this class even survive refactoring??

    def __bool__(self): return bool(self.monos)
    def __copy__(self): return self.__class__(self.var_names, self.monos)

    def __eq__(self, other):
        other = self._with_same_vars(other)
        return self.monos == other.monos
    def __le__(self, other):
        other = self._with_same_vars(other)
        return self.monos <= other.monos
    def __add__(self, other):
        other = self._with_same_vars(other)
        return self.__class__(self.var_names, self.monos | other.monos)
    def __iadd__(self, other):
        other = self._with_same_vars(other)
        self.monos.update(other.monos)
        return self
    def __mul__(self, other):
        other = self._with_same_vars(other)
        return self.__class__(self.var_names, (_tuple_sum(mi, mj) for mi in self.monos for mj in other.monos))
    def conjugate(self):
        return self.__class__(self.var_names, map(_tuple_neg, self.monos))

    __rmul__ = __mul__
    __radd__ = __add__

    def _with_same_vars(self, other):
        if isinstance(other, BoolPoly):
            assert other.var_names == self.var_names
            return other
        return self.one() if other else self.zero()


class Series(Poset, Semiring):
    def __init__(self, monos=()):
        self.monos = monos if isinstance(monos, WeightSet) else WeightSet(monos)
    def zero(self):
        return self.__class__()
    def one(self):
        return self.__class__((0,))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.monos})'

    def __str__(self):
        parts = []
        if self.monos.pos: parts.append(f'z^({self.monos.pos})')
        if self.monos.elements: parts.extend(_pow_name('z', power) or '1' for power in sorted(self.monos.elements))
        if self.monos.neg: parts.append(f'z^-({self.monos.neg})')
        return ' + '.join(parts)

    @classmethod
    def from_text(cls, text):
        out = cls()
        monos = set()
        # TODO: actually parse, move to "*" notation of https://en.wikipedia.org/wiki/Rational_series
        terms = []
        unclosed = 0
        for term in text.split('+'):
            if unclosed:
                terms[-1] = f'{terms[-1]}+{term}'
            else:
                terms.append(term)
            unclosed += term.count('(') - term.count(')')
        for term in map(str.strip, terms):
            if term == '1':
                monos.add(0)
            elif term == 'z':
                monos.add(1)
            elif not term.startswith('z^'):
                raise ValueError(f'Invalid series term: {term}')
            else:
                try:
                    power = int(term[2:])
                    monos.add(power)
                except ValueError:
                    out.monos |= WeightSet.from_text(term[2:])
        out.monos |= WeightSet(monos)
        return out

    def __bool__(self): return bool(self.monos)
    def __copy__(self): return self.__class__(self.monos)
    def __invert__(self): return self.__class__(~self.monos)

    def __eq__(self, other):
        other = self._as_series(other)
        if other is None: return NotImplemented
        return self.monos == other.monos
    def __le__(self, other):
        other = self._as_series(other)
        if other is None: return NotImplemented
        return self.monos <= other.monos
    def __add__(self, other):
        other = self._as_series(other)
        if other is None: return NotImplemented
        return self.__class__(self.monos | other.monos)
#    def __iadd__(self, other):
#        other = self._as_series(other)
#        self.monos |= other.monos
#        return self
    def __mul__(self, other):
        other = self._as_series(other)
        if other is None: return NotImplemented
        return self.__class__(self.monos + other.monos)
    def conjugate(self):
        return self.__class__(-self.monos)
    def geometric_series(self):
        return self.__class__(self.monos.loop())

    __rmul__ = __mul__
    __radd__ = __add__

    def _as_series(self, other):
        if isinstance(other, Series):
            return other
        if isinstance(other, Matrix):
            return None  # HACK: punt scalar*vector etc.
        return self.one() if other else self.zero()


class Matrix(Poset, Semiring):
    def __init__(self, l_basis, r_basis=None, coefs=None):
        if r_basis is None: l_basis = r_basis = tuple(l_basis)
        self.l_basis = tuple(l_basis)
        self.r_basis = tuple(r_basis)
        self.c = [list(row) for row in coefs] if coefs else [[0 for _ in self.r_basis] for _ in self.l_basis]
    def zero(self):
        return self.__class__(self.l_basis, self.r_basis)
    def one(self, scale=1):
        assert self.l_basis == self.r_basis
        out = self.__class__(self.l_basis, self.r_basis)
        for i, ci in enumerate(out.c):
            ci[i] = scale
        return out

    def __str__(self):
        elem_names = (_elem_name(clr, l_name, r_name) for l_name, cl in zip(self.l_basis, self.c) for r_name, clr in zip(self.r_basis, cl))
        # As a special case, if a side's basis is ('',), display as a vector.
        return ' + '.join(filter(None, elem_names)).replace('*|><|', '').replace('|>', '').replace('<|', '') or '0'

    @classmethod
    def from_text(cls, text, l_basis, r_basis=None, coef_type=Series):
        out = cls(l_basis, r_basis)
        # Allow vectors.
        if len(out.l_basis) == 1 and '>' not in text:
            text = text.replace('<', f'|{out.l_basis[0]}><')
        if len(out.r_basis) == 1 and '<' not in text:
            text = text.replace('>', f'><{out.r_basis[0]}|')
        for coef_text, l, r in re.findall(r' \+ (.*?)\|(.*?)><(.*?)\|', ' + '+text):
            coef_text = coef_text.rstrip('*')
            if coef_text.startswith('(') and coef_text.endswith(')'):
                coef_text = coef_text[1:-1]
            out[out.l_basis.index(l), out.r_basis.index(r)] += coef_type.from_text(coef_text) if coef_text else 1
        return out

    def __bool__(self): return any(clr for cl in self.c for clr in cl)
    def __copy__(self): return self.__class__(self.l_basis, self.r_basis, self.c)
    def __invert__(self): return self.__class__(self.l_basis, self.r_basis, [[(1-clr if isinstance(clr,int) else ~clr) for clr in cl] for cl in self.c])

    def __eq__(self, other):
        other = self._with_same_basis(other)
        return self.c == other.c
    def __le__(self, other):
        other = self._with_same_basis(other)
        return all(slr <= olr for sl, ol in zip(self.c, other.c) for slr, olr in zip(sl, ol))
    def __add__(self, other): return self.__copy__().__iadd__(other)
    def __iadd__(self, other):
        other = self._with_same_basis(other)
        for sl, ol in zip(self.c, other.c):
            for i, olr in enumerate(ol):
                sl[i] += olr
        return self
    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.r_basis == other.l_basis
            out = self.__class__(self.l_basis, other.r_basis)
            for i in range(len(self.l_basis)):
                for j in range(len(self.r_basis)):
                    for k in range(len(other.r_basis)):
                        out.c[i][k] += self.c[i][j]*other.c[j][k]
            return out
        return self.scale(other)
    def scale(self, other):
        out = self.__class__(self.l_basis, self.r_basis)
        for sl, ol in zip(self.c, out.c):
            for i, slr in enumerate(sl):
                ol[i] = slr*other
        return out

    def geometric_series(self):
        cand = self
        while True:
            succ = cand.__copy__()
            for i, ci in enumerate(succ.c):
                ci[i] = ci[i].geometric_series() if ci[i] else 1
            succ *= succ
            if succ == cand:
                return succ
            cand = succ

    def oplus(self, *others):
        ''' the direct sum (block vector, or block diagonal matrix) self ⊕ other. '''
        blocks = (self,) + others
        vec_l = all(bi.l_basis == bj.l_basis for bi, bj in zip(blocks, blocks[1:]))
        vec_r = all(bi.r_basis == bj.r_basis for bi, bj in zip(blocks, blocks[1:]))
        l_basis = self.l_basis if vec_l else [bi for b in blocks for bi in b.l_basis]
        r_basis = self.r_basis if vec_r else [bi for b in blocks for bi in b.r_basis]
        out = self.__class__(l_basis, r_basis)
        sl = sr = 0
        for b in blocks:
            for l, cl in enumerate(b.c):
                out.c[sl + l][sr : sr + len(cl)] = cl
            if not vec_l: sl += len(b.l_basis)
            if not vec_r: sr += len(b.r_basis)
        return out

    def conjugate(self):
        return self.transpose()

    def transpose(self, conjugate=True):
        out = self.__class__(self.r_basis, self.l_basis)
        for i in range(len(self.l_basis)):
            for j in range(len(self.r_basis)):
                out.c[j][i] = self.c[i][j].conjugate() if conjugate else self.c[i][j]
        return out

    def __getitem__(self, lr):
        return self.c[lr[0]][lr[1]]
    def __setitem__(self, lr, v):
        self.c[lr[0]][lr[1]] = v

    __radd__ = __add__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__ = scale

    def _with_same_basis(self, other):
        if isinstance(other, Matrix):
            assert other.l_basis == self.l_basis and other.r_basis == self.r_basis
            return other
        assert self.l_basis == self.r_basis
        return self.one(scale=other)


class PeriodicNatSubset:
    def __init__(self, rems=(), mod=1, period_known_minimal=False):
        self.rems, self.mod = set(), mod
        for rem in sorted(rems):
            if rem not in self:
                self.rems.add(rem)
        if not period_known_minimal:
            for f in range(1, self.mod):
                if self.mod % f == 0:
                    superset = PeriodicNatSubset(rems, f, period_known_minimal=True)
                    if superset <= self:
                        self.rems, self.mod = superset.rems, superset.mod
                        return

    def loop(self):
        ''' Return the set of sums of finitely many elements of self. '''
        assert self, "Oops, {}.loop() is not periodic; don't do that."
        return sum((PeriodicNatSubset({0}, rem) for rem in self.rems), self)

    def __bool__(self):
        return bool(self.rems)

    def __eq__(self, other):
        return (self.rems, self.mod) == (other.rems, other.mod)

    def __lt__(self, other):
        return self != other and self <= other

    def __contains__(self, n):
        return any(base <= n and (n - base) % self.mod == 0 for base in self.rems) # NOTE: works before __init__'s simplifications are done

    def __le__(self, other):
        common = lcm(self.mod, other.mod)
        return all(rem + period in other for rem in self.rems for period in range(0, common, self.mod)) # NOTE: this does too

    def __str__(self):
        return f'{self.rems} + {self.mod}ℕ' if self.rems else '{}'

    @classmethod
    def from_text(cls, text):
        if not text.startswith('{'): raise ValueError(f'Malformed (unsigned) weight-set {text}')
        rems, _, rest = text[1:].partition('}')
        rems = {int(v.strip()) for v in rems.split(',')}
        mod = int(rest.strip(' +Nℕ')) if rest else 1
        return cls(rems, mod)

    def _scaled_rems(self, other):
        s_rems = {rem + i*self.mod for rem in self.rems for i in range(other.mod)}
        o_rems = {rem + i*other.mod for rem in other.rems for i in range(self.mod)}
        return s_rems, o_rems

    def __or__(self, other):
        s_rems, o_rems = self._scaled_rems(other)
        return PeriodicNatSubset(s_rems | o_rems, lcm(self.mod, other.mod))

    def __ror__(self, other):
        return self | other if other else self

    def __add__(self, other):
        s_rems, o_rems = self._scaled_rems(other)
        return PeriodicNatSubset({s+o for s in s_rems for o in o_rems}, lcm(self.mod, other.mod))

    def __sub__(self, other):
        s_rems, o_rems = self._scaled_rems(other)
        mod = gcd(self.mod, other.mod)  # Thanks to the Euclidean algorithm, for positive p,n, pN-nN = gcd(p,n)Z.
        p_rems = {(s-o)%mod for s in s_rems for o in o_rems}
        n_rems = {(-rem)%mod for rem in p_rems}
        return WeightSet(set(), pos=PeriodicNatSubset(p_rems, mod), neg=PeriodicNatSubset(n_rems, mod))

    def plus_signed_values(self, elements):
        """Add (+) a finite set to self. The result is a periodic subset of Z. Split this into a periodic subset of N and a finite set, and return them."""
        rems, neg = set(), set()
        for sum_ in (s + e for s in self.rems for e in elements):
            rems.add(sum_ if sum_ >= 0 else sum_ % self.mod)
            neg.update(range(sum_, 0, self.mod))
        return PeriodicNatSubset(rems, self.mod), neg

    def insert_or_yield(self, elements):
        """Insert as many of the given elements as possible (for a periodic subset of N). This generator yields the leftovers."""
        for e in sorted(elements, reverse=True):
            if e < 0: yield e
            elif e in self: continue
            else:
                for rem in sorted(self.rems):
                    if (rem - e) % self.mod == 0:
                        self.rems.remove(rem)
                        self.rems = {rem + mods for rem in self.rems for mods in range(0, rem - e, self.mod)}
                        self.rems.add(e)
                        self.mod = rem - e
                        break
                else: yield e

    def complement_minus(self, elements):
        """Return ℕ - (self U elements) as a PeriodicNatSubset (as large as possible) and the set of leftover natural numbers. """
        rems = set(range(self.mod))
        rest = set()
        for rem in self.rems:
            rems.remove(rem % self.mod)
            rest.update(range(rem % self.mod, rem, self.mod))
        for e in sorted(elements):
            if e in rest:
                rest.remove(e)
                continue
            for rem in sorted(rems):
                if rem > e: break
                if (e - rem) % self.mod == 0:
                    rems.remove(rem)
                    q = (e - rem) // self.mod
                    rest.update(range(rem, e, self.mod))
                    rems.add(rem + (q + 1) * self.mod)
        return PeriodicNatSubset(rems, self.mod), rest


class WeightSet:
    """A set of integers representing the possible total weights on a path between WFA nodes.
       If w is an integer and S, S' are weight sets, then {w}, S+S', S|S', and S.loop() := {sums of finitely many elements of S} are weight sets.
       Periodic subsets of N are closed under these operations, as worked out above. So the most general thing we get is P | S | -N, where
       P and N are periodic subsets of N and S is a finite set. We enforce a simplified form where P,N are as large, and S as small, as possible."""
    def __init__(self, elements=(), pos=None, neg=None):
        self.elements, self.pos, self.neg = set(elements), pos or PeriodicNatSubset(), neg or PeriodicNatSubset()
        if 0 in self: self.elements.add(0)
        self.elements = set(self.pos.insert_or_yield(self.elements))
        if 0 in self: self.elements.add(0)
        self.elements = set(map(int.__neg__, self.neg.insert_or_yield(map(int.__neg__, self.elements))))
        if 0 in self.pos.rems or 0 in self.neg.rems: self.elements.discard(0)

    @classmethod
    def singleton(cls, weight):
        return cls({weight})

    def loop_if(self, cond):
        return self.loop() if cond else self

    def loop(self):
        pos, neg = [], []
        if self.pos: pos.append(self.pos.loop())
        if self.neg: neg.append(self.neg.loop())
        for e in self.elements:
            if e < 0: neg.append(PeriodicNatSubset({0}, -e))
            if e > 0: pos.append(PeriodicNatSubset({0}, e))
        pos = sum(pos[1:], pos[0]) if pos else PeriodicNatSubset()
        neg = sum(neg[1:], neg[0]) if neg else PeriodicNatSubset()
        return pos - neg if pos and neg else WeightSet({0}, pos, neg)

    def __bool__(self):
        return any((self.elements, self.pos, self.neg))

    def __invert__(self):
        no = self.elements | {0} if 0 in self else self.elements
        pos, elements = self.pos.complement_minus(no)
        neg, elts_neg = self.neg.complement_minus(map(int.__neg__, no))
        elements.update(map(int.__neg__, elts_neg))
        return self.__class__(elements, pos, neg)

    def __eq__(self, other):
        return (self.elements, self.pos, self.neg) == (other.elements, other.pos, other.neg)

    def __lt__(self, other):
        return self != other and self <= other

    def __contains__(self, n):
        return n in self.elements or (n >= 0 and n in self.pos) or (n <= 0 and -n in self.neg)

    def __le__(self, other):
        return self.pos <= other.pos and self.neg <= other.neg and all(e in other for e in self.elements)

    def __str__(self):
        parts = []
        if self.pos: parts.append(f'({self.pos})')
        if self.elements: parts.append(str(self.elements))
        if self.neg: parts.append(f'-({self.neg})')
        return ' | '.join(parts) or '{}'

    @classmethod
    def from_text(cls, text):
        pos = neg = None
        elements = set()
        for term in map(str.strip, text.split('|')):
            if term.startswith('(') and term.endswith(')'):
                if pos is not None: raise ValueError(f'Unexpected weight-set term {term} following another positive term')
                pos = PeriodicNatSubset.from_text(term[1:-1])
            elif term.startswith('{') and term.endswith('}'):
                elements.update(int(v.strip()) for v in term[1:-1].split(','))
            elif term.startswith('-(') and term.endswith(')'):
                if neg is not None: raise ValueError(f'Unexpected weight-set term {term} following another positive term')
                neg = PeriodicNatSubset.from_text(term[2:-1])
            else: raise ValueError(f'Unexpected weight-set term {term}')
        return WeightSet(elements, pos, neg)

    def __or__(self, other):
        return WeightSet(self.elements | other.elements, self.pos | other.pos, self.neg | other.neg)

    def __add__(self, other):
        elements = {s+o for s in self.elements for o in other.elements}
        pos = self.pos + other.pos
        neg = self.neg + other.neg

        periodic, isolated = self.pos.plus_signed_values(other.elements)
        pos |= periodic
        elements |= isolated

        periodic, isolated = other.pos.plus_signed_values(self.elements)
        pos |= periodic
        elements |= isolated

        periodic, isolated = self.neg.plus_signed_values(map(int.__neg__, other.elements))
        neg |= periodic
        elements.update(map(int.__neg__, isolated))

        periodic, isolated = other.neg.plus_signed_values(map(int.__neg__, self.elements))
        neg |= periodic
        elements.update(map(int.__neg__, isolated))

        return WeightSet(elements, pos, neg) | (self.pos - other.neg) | (other.pos - self.neg)

    def __ior__(self, other):
        if self.pos == other.pos and self.neg == other.neg:
            self.elements.update(other.elements)
            return self
        return self | other

    def __neg__(self):
        return WeightSet({-e for e in self.elements}, self.neg, self.pos)


def _tuple_neg(t):
    return tuple(-i for i in t)
def _tuple_sum(ti, tj):
    return tuple([i+j for i,j in zip(ti, tj)])
def _pow_name(var_name, power):
    if power == 1: return var_name
    elif power: return f'{var_name}^{power}'
def _elem_name(c, l, r):
    if not c: return
    unit = f'|{l}><{r}|'
    if c == 1: return unit
    sc = str(c)
    return f'({sc})*{unit}' if ' ' in sc else f'{sc}*{unit}'
