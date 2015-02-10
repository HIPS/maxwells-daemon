import numpy as np

# TODO: implement this with arithmetic coding to make it faster and scalable.
class InfoBuffer(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""
    def __init__(self, length):
        # Use an array of Python 'long' ints which conveniently grow as needed
        self.store = np.array([0L] * length, dtype=object)
        self.curmax = 0L # Keeps track of largest number allowable

    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert np.all(M <= 2**16)
        self.curmax *= M
        self.curmax += M - 1
        self.store *= M
        self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        self.curmax /= M
        self.store /= M
        return N

    def randomize(self, rs):
        self.store = rs.randint(self.curmax + 1, size=len(self.store))

    def is_empty(self):
        return np.all(self.store == 0)
