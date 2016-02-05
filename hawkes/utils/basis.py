import abc

import autograd.numpy as np
import scipy.linalg
import scipy.signal as sig

# basic convolving operation
def convolve(sig, win):
    assert isinstance(sig, np.ndarray) and sig.ndim == 1
    out = np.zeros(len(sig))
    for i in range(len(sig)):
        if sig[i] > 0:
            out[i:min(i+len(win), len(sig))] = sig[i] * win[:min(i+len(win), len(sig))-i]
    return out

class Basis(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, B, dt, dt_max):
        self.B = B
        self.dt = dt
        self.dt_max = dt_max
        self.basis = self.interpolate_basis()

    def create_basis(self):
        raise NotImplementedError("need to implement basis specification")

    def convolve_with_basis(self, signal):
        """
        Convolve each column of the event count matrix with this basis
        :param S:     signal: an array-like data, each series is (1, T) shape
        :return: TxB of inputs convolved with bases
        """
        (T,_) = signal.shape
        (R,B) = self.basis.shape


        # Initialize array for filtered stimulus
        F = np.empty((T,B))

        # Compute convolutions fo each basis vector, one at a time
        for b in np.arange(B):
            F[:,b] = sig.fftconvolve(signal,
                                       np.reshape(self.basis[:,b],(R,1)),
                                       'full')[:T,:]

        # Check for positivity
        if np.amin(self.basis) >= 0 and np.amin(signal) >= 0:
            np.clip(F, 0, np.inf, out=F)
            assert np.amin(F) >= 0, "convolution should be >= 0"

        return F

    def interpolate_basis(self, basis, dt, dt_max,
                          norm=True):
        # Interpolate basis at the resolution of the data
        L,B = basis.shape
        t_int = np.arange(0.0, dt_max, step=dt)
        t_bas = np.linspace(0.0, dt_max, L)

        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if norm:
            # ibasis /= np.trapz(ibasis,t_int,axis=0)
            ibasis /= (dt * np.sum(ibasis, axis=0))

        if not self.allow_instantaneous:
            # Typically, the impulse responses are applied to times
            # (t+1:t+R). That means we need to prepend a row of zeros to make
            # sure the basis remains causal
            ibasis = np.vstack((np.zeros((1,B)), ibasis))

        return ibasis

## TODO: Implement related basis families ##
class CosineBasis(Basis):

    def create_basis(self):
        return


class WeibullBasis(Basis):

    def create_basis(self):
        return


class ExponentialBasis(Basis):

    def create_basis(self):
        return