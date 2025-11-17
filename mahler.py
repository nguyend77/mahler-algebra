import numpy as np

class Mahler: # {m: coeff}
    def __init__(self, terms):
        self.terms = {m: coeff for m, coeff in terms.items() if coeff != 0}
    def __repr__(self): # consistent print() representation
        if not self.terms:
            return '0'
        parts = []
        for m in sorted(self.terms.keys()):
            coeff = self.terms[m]
            if not parts: # first term
                if coeff == 1:
                    parts.append(f'[{m}]')
                elif coeff == -1:
                    parts.append(f'-[{m}]')
                else:
                    parts.append(f'{coeff}[{m}]')
            else: # subsequent terms
                if coeff == 1:
                    parts.append(f'+ [{m}]')
                elif coeff == -1:
                    parts.append(f'- [{m}]')
                else:
                    parts.append(f'{coeff:+} [{m}]')
        return ' '.join(parts)
    def __eq__(self, other): # equality check with ==
        if other == 0:
            return not self.terms
        return self.terms == other.terms
    def __add__(self, other): # addition with +
        new_terms = self.terms.copy()
        for m, coeff in other.terms.items():
            new_terms[m] = new_terms.get(m, 0) + coeff
        return Mahler(new_terms)
    def __sub__(self, other): # subtraction with -
        new_terms = self.terms.copy()
        for m, coeff in other.terms.items():
            new_terms[m] = new_terms.get(m, 0) - coeff
        return Mahler(new_terms)
    def __mul__(self, other): # multiplication with *
        if isinstance(other, int): # scalar multiplication
            return Mahler({m: coeff * other for m, coeff in self.terms.items()})
        new_terms = {}
        for m_self, coeff_self in self.terms.items():
            for m_other, coeff_other in other.terms.items():
                m = m_self * m_other
                coeff = coeff_self * coeff_other
                new_terms[m] = new_terms.get(m, 0) + coeff
        return Mahler(new_terms)
    def __rmul__(self, other): # hanlde scalar multiplication from the left
        return self.__mul__(other)
    def __mod__(self, other): # reduce modulo d with %
        d = other
        new_terms = {}
        for m, coeff in self.terms.items():
            new_terms[m%d] = new_terms.get(m%d, 0) + coeff
        return Mahler(new_terms)
    def apply(self, f): # [m](f(x)) = f(x^m)
        variables = f.variables()
        if not variables:
            return sum(self.terms.values()) * f
        x = variables[0]
        result = 0
        for m, coeff in self.terms.items():
            result += coeff * f.subs({x: x**m})
        return result
    
def solve_congruence(alpha, beta, d): # find gamma s.t. alpha*gamma \equiv beta (mod d)
    A = np.zeros((d,d)) # d*d matrix with column i is the vector of coefficients from (alpha*[i])%d
    for i in range(d): # column index, representing the [i] associated with gamma_i
        for m_a, coeff_a in alpha.terms.items():
            j = (m_a*i) % d
            A[j][i] += coeff_a
    b = np.zeros(d) # vector of coefficients from b reduced modular d
    for m_b, coeff_b in (beta%d).terms.items():
        b[m_b] = coeff_b
    x = np.round(np.linalg.solve(A, b)).astype(int)
    if np.allclose(A @ x, b): # check for potential float rounding errors
        gamma_terms = {i: x[i] for i in range(d) if x[i] != 0}
        return Mahler(gamma_terms)
        
def moebius_py(n):
    if n == 1:
        return 1 # mu(n) = 1  if n = 1
    prime_factor_count = 0
    temp_n = n
    if temp_n % 2 == 0:
        prime_factor_count += 1
        temp_n //= 2
        if temp_n % 2 == 0:
            return 0 # mu(n) = 0  if n has a squared prime factor
    i = 3
    while i * i <= temp_n:
        if temp_n % i == 0:
            prime_factor_count += 1
            temp_n //= i
            if temp_n % i == 0:
                return 0 # mu(n) = 0  if n has a squared prime factor
        i += 2
    if temp_n > 1: # last prime factor
        prime_factor_count += 1
    if prime_factor_count % 2 == 0:
        return 1
    else:
        return -1

def psi(n): # \psi_n = \sum_{d|n} \mu(d)[d]
    terms = {}
    divisors = [d for d in range(1,n+1) if n%d == 0]
    for d in divisors:
        mu = moebius_py(d)
        if mu != 0:
            terms[d] = mu
    return Mahler(terms)

def phi(n): # \phi_n = sum_{d|n} mu(d)[n/d]
    terms = {}
    divisors = [d for d in range(1,n+1) if n%d == 0]
    for d in divisors:
        mu = moebius_py(d)
        if mu != 0:
            terms[n//d] = mu
    return Mahler(terms)