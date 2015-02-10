import numpy as np
from info_buffer import InfoBuffer
from copy import deepcopy

nbits_to_dtype = {64 : np.int64,
                  32 : np.int32,
                  16 : np.int16}

class ExactRep(object):
    """Fixed-point representation of arrays with auxilliary bits such
    that + - * / ops are all exactly invertible (except for
    overflow)."""
    def __init__(self, val, from_intrep=False, nbits=64):
        self.dtype = nbits_to_dtype[nbits]
        self.radix_scale = 2**(nbits / 2)
        self.rational_bits = nbits / 4
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = self.float_to_intrep(val)

        self.aux = InfoBuffer(len(val))

    def randomize(self, rs):
        self.aux.randomize(rs)
        return self

    def copy(self):
        return deepcopy(self)

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += self.float_to_intrep(A)
        return self

    def sub(self, A):
        self.add(-A)
        return self

    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d) # Store remainder bits externally
        self.intrep /= d                  # Divide by denominator
        self.intrep *= n                  # Multiply by numerator
        self.intrep += self.aux.pop(n)    # Pack bits into the remainder

    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self

    def float_to_rational(self, a):
        assert np.all(a > 0.0)
        d = 2**self.rational_bits / np.fix(a+1).astype(int)
        n = np.fix(a * d).astype(int) # TODO: fix case n == 0
        return  n, d

    def float_to_intrep(self, x):
        return (x * self.radix_scale).astype(self.dtype)

    @property
    def val(self):
        return self.intrep.astype(np.float64) / self.radix_scale

