import abc
import copy

try:
    import cPickle as pickle
except:
    import pickle

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
from numpy import tile

from utils.basis import convolve

def default_exp(t, w=7):
    if (t >= 1 and t <= w + 1):
        return np.exp(-1 * (t - 1))
    elif t == 0:
        return 1
    else:
        return 0


def weibull(t, k=1.5, l=1, w=7):
    x = t/float(w)*2.5  # transform to [0, 3]
    return (k/l) * np.power((x/l), k-1) * np.exp(-1 * np.power((x/l), k)) if x > 0 else 0


class _discreteHawkesBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, K, dt, dt_max, lamb=0.1, basis=None, self_connect=False, extra_w=False, reg=None):
        """
        :param K: number of substreams in the multivariate process
        :param dt: time resolution (discrete case)
        :param dt_max: window size
        :param lamb: regularization factor
        :param basis: basis object
        :param self_connect: allows self-connection of influence (i.e. w_ii > 0)
        :param extra_w: use a separate weight matrix for cross process features
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max
        self.lamb = lamb
        self.self_connect = self_connect
        self.extra_w = extra_w
        self.F = 1  # default feature space
        self.win = None
        self.data = None
        self.conv_data = None
        self.conv_data_fea = None
        self.reg = reg  # specify regularization

        if basis is None:
            self.basis = default_exp  # default: exponential function with fixed window length
            self.basis_args = {'w': dt_max}

        # initialize weight matrix
        # direction: w_ij -> process i influences process j
        self.update_K(self.K)
        self.v = None

    def update_K(self, k):
        self.K = k
        # reinitialize K related parameters
        self.w = 1e-3 * np.ones((self.K + 1, self.K))
        if not self.self_connect:
            for i in xrange(self.K):
                self.w[i + 1, i] = 1e-32
        if self.extra_w:
            tmp = 1e-10 * np.ones((self.K, self.K))
            if not self.self_connect:
                for i in xrange(self.K):
                    tmp[i, i] = 1e-32
            self.w = np.concatenate([self.w, tmp], axis=0)

    def set_data(self, data, ignore_feat=True):
        """
        :param data: a TxKxV tensor of event-feature map where T- time; K- process id; V- corresponding feature vector
        :input feature: V+1 features, [V features, 1 bias], bias- denotes recency
        """
        assert isinstance(data, np.ndarray) and data.ndim == 3, "data needs to be a TxKxV tensor of event features"
        self.update_K(data.shape[1])
        self.F = data.shape[2]-1
        # calculate linear mark function: F expected to be KxKxTxF tensor
        if self.F < 1 or ignore_feat:
            f_mat = None
            self.F = 0
        else:
            f_mat = data[:, :, :-1]
        if self.F > 0:
            self.v = 1e-3 * np.ones((self.K, self.F))
        self.win = np.array([self.basis(i/float(self.dt), **self.basis_args) for i in xrange(data.shape[0])])

        self.data = data  # orignal event features, shape: TxKxF
        # convolve data with windowed signal
        self.conv_data = np.concatenate([convolve(data[:, k, -1], self.win)[:, None] for k in xrange(data.shape[1])], axis=1)
        self.conv_data_fea = f_mat  # convolved to feature matrix, shape: TxKxF

    def compute_modfeat(self, w):
        mod_feat = self.conv_data_fea.copy()
        mod_dem = np.ones(mod_feat.shape)  ## need to change to accommodate non-binary features
        ws = np.array([w[k*self.F:(k+1)*self.F] for k in range(self.K)])
        w_tiled = np.array([ws for i in range(mod_feat.shape[0])])
        mod_feat = mod_feat * w_tiled
        mod_dem = mod_dem * w_tiled
        mod_feat = np.sum(np.exp(mod_feat), axis=2)
        mod_dem = np.sum(np.exp(mod_dem), axis=2)
        return mod_feat / mod_dem

    def compute_rate(self, w, w_f=None):
        """
        :param w: Kx(1+K+(K)) weight matrix
        :return: inferred rate, shape: TxK
        """
        if not self.extra_w:
            w_t = np.concatenate([w[0, :].reshape((self.K, 1)), w[1:, :]], axis=1)
        else:
            w_t, w_t2 = np.concatenate([w[0, :].reshape((self.K, 1)), w[1:1+self.K, :]], axis=1), w[1+self.K:, :]
        Rs = []
        for k in range(self.K):
            Rs.append(np.dot(self.conv_data, w_t[:, k+1])[:, None] + w_t[k, 0])
        Rs = np.concatenate(Rs, axis=1)
        if self.conv_data_fea is not None:
            # calculates a softmax modulated feature matrix, shape: TxKxF
            mod_feat = self.compute_modfeat(w_f)
            Rs = Rs * mod_feat
        return Rs

    @property
    def bias(self):
        return self.w[0, :]

    @property
    def W(self):
        return self.w[1:, :]

    @property
    def Attr(self):
        assert self.conv_data_fea is not None, "attribution needs to have feature data"
        w = self.v
        mod_feat = self.conv_data_fea.copy()
        mod_dem = np.ones(mod_feat.shape)  ## need to change to accommodate non-binary features
        for t in xrange(mod_feat.shape[0]):
            for k in xrange(mod_feat.shape[1]):
                mod_feat[t, k, :] = mod_feat[t, k, :] * w[k, :]
                mod_dem[t, k, :] = mod_dem[t, k, :] * w[k, :]
        return mod_feat / mod_dem

    #  Compute influnece, return shape: TxK(influecing)xK(influenced)
    @property
    def Lambda(self):
        w = self.W
        Rs = []
        for k in range(self.K):
            wk = tile(w[:self.K, k], (self.conv_data.shape[0], 1))
            Rs.append((self.conv_data * wk)[:, None])
        return np.concatenate(Rs, axis=2).reshape((self.conv_data.shape[0], self.K, self.K), order='F')

    def copy_model(self):
        return copy.deepcopy(self)

    def save_model(self, fname):
        with open(fname, 'wb') as wf:
            pickle.dump(self.__dict__, wf, 2)

    def load_model(self, fname):
        with open(fname, 'rb') as rf:
            model_dict = pickle.load(rf)
            self.__dict__.update(model_dict)

    @abc.abstractmethod
    def link(self, psi):
        pass

    def base_objective(self, w):
        # calculate log(lambda)
        assert (len(w) - self.F * self.K) % self.K == 0, "w length should be K multiple + feature size"
        tmpw = w[:len(w) - self.F * self.K].reshape(((len(w) - (self.F * self.K))/self.K, self.K))
        if self.F > 0:
            w_f = w[-self.F*self.K:]
        else:
            w_f = None
        R = self.compute_rate(tmpw, w_f) + 1e-5
        obj = (self.data[:, :, -1] * np.log(R) - R).sum()

        return obj

    def objective(self, w):
        """
        :param w: weights + lambda parameters of likelihood
        :return: the regularized negative log-likelihood function of hawkes process
        """
        obj = self.base_objective(w)

        # link function
        obj = self.link(obj)

        # L1 or L2 regularization
        obj = obj + self.regularizer(w)

        return -1 * obj

    def regularizer(self, w):
        reg_score = 0.0
        def L1(x):
            return np.sqrt(np.dot(x,np.conj(x)))
        def L2(x):
            return np.sqrt(np.dot(x, x))
        if self.reg == "L2" or self.reg == "Elastic":
            reg_score = reg_score + self.lamb * L2(w[self.K:len(w)-self.F*self.K])
        elif self.reg == "L1" or self.reg == "Elastic":
            reg_score = reg_score + self.lamb * L1(w[self.K:len(w)-self.F*self.K])
        return reg_score

    def fit_bfgs(self):
        """
        Fit the model objective with bfgs, using autograd package for automatic gradient differentiation
        """
        itr = [0]

        def callback(w):
            if itr[0] % 10 == 0:
                print "iteration: %03d\t LP: %.5f" % (itr[0], self.objective(w))
            itr[0] = itr[0] + 1

        # initialize weight vector x0
        # x0 dim: bias (1xK) + influence (KxK) + softmax (1xF)
        x0 = self.w

        # reshape x0 to 1-dim
        if self.F > 0:
            x0 = np.concatenate([x0.reshape((x0.shape[0] * x0.shape[1],)), self.v.reshape((self.v.shape[0] * self.v.shape[1],))])
        else:
            x0 = x0.reshape((x0.shape[0] * x0.shape[1],))

        # set bounds conditions
        bnds = [(0, None)] * self.K + [(0, 1)] * (len(x0) - self.K)
        if not self.self_connect:
            pos = self.K
            c = 0
            while pos < len(x0):
                bnds[pos] = (0, 1e-32)
                if c > self.K * self.K - self.K:
                    c = 0
                    pos += 1
                else:
                    c += self.K + 1
                    pos += self.K + 1

        res = minimize(self.objective,
                       x0,
                       jac=grad(self.objective),
                       bounds=bnds,
                       callback=callback)

        if self.F > 0:
            self.w = res.x[:-(self.F * self.K)].reshape(((len(res.x) - (self.F * self.K))/self.K, self.K))
            self.v = res.x[-(self.F * self.K):].reshape((self.K, self.F))
        else:
            self.w = res.x.reshape((len(res.x)/self.K, self.K))

    def get_inferred_patterns(self, thres=1e-3):
        ixs = np.where(self.Lambda > thres)
        return ixs  # returns three dimensional arrays of TxKxK indices


class LinearDiscreteHawkes(_discreteHawkesBase):
    def link(self, psi):
        return psi


class RectlinearDiscreteHawkes(_discreteHawkesBase):
    def link(self, psi):
        return 1e-16 + np.log(1. + np.exp(psi))


class ExpDiscreteHawkes(_discreteHawkesBase):
    def link(self, psi):
        return np.exp(psi)


class NeighborLinDiscreteHawkes(LinearDiscreteHawkes):
    def __init__(self, *argv, **kwargs):
        super(LinearDiscreteHawkes, self).__init__(*argv, **kwargs)
        self.extra_w = True
        self.update_K(self.K)
        self.conv_dyad_data = None

    def set_data(self, data, dyadic_data, ignore_feat=True):
        """
        :param dyadic_data: expect dyadic interaction data between processes, shape: KxKxT
        """
        assert isinstance(dyadic_data,
                          np.ndarray) and data.ndim == 3, "dyadic_data needs be a KxKxT tensor of interaction events"
        super(LinearDiscreteHawkes, self).set_data(data, ignore_feat)
        self.conv_dyad_data = dyadic_data
        # convolution step
        for k1 in xrange(self.conv_dyad_data.shape[0]):
            for k2 in xrange(self.conv_dyad_data.shape[1]):
                self.conv_dyad_data[k1, k2, :] = self.conv_dyad_data[k1, k2, :] + convolve(self.conv_dyad_data[k1, k2, :], self.win)

    def compute_rate(self, w, w_f=None):
        Rs = super(LinearDiscreteHawkes, self).compute_rate(w, w_f)
        Rs2 = []
        if self.extra_w:
            w_dyad = w[1+self.K:, :]
        else:
            w_dyad = w[1:1+self.K, :]
        for t in range(self.conv_dyad_data.shape[2]):
            Rs2.append(self.conv_dyad_data[:, :, t] * w_dyad)
        Rs2 = np.concatenate(Rs2, axis=1).reshape(self.conv_dyad_data.shape, order='F')
        return Rs + 1e-5, np.sum(Rs2, axis=0).T

    def base_objective(self, w):
        # calculate log(lambda)
        assert (len(w) - self.F*self.K) % self.K == 0, "w length should be K multiple + feature size"
        tmpw = w[:len(w) - self.F*self.K].reshape(((len(w) - self.F*self.K)/self.K, self.K))
        if self.F > 0:
            w_f = w[-self.F*self.K:]
        else:
            w_f = None
        R, R_dyad = self.compute_rate(tmpw, w_f)
        obj = (self.data[:, :, -1] * np.log(R + self.data.shape[0] * R_dyad) - (R + R_dyad)).sum()
        return obj

    def regularizer(self, w):
        reg_score = 0.0
        def L1(x):
            return np.sqrt(np.dot(x,np.conj(x)))
        def L2(x):
            return np.sqrt(np.dot(x, x))
        if self.reg == "L2" or self.reg == "Elastic":
            reg_score = reg_score + self.lamb * L2(w[self.K:self.K + self.K**2])
            if self.extra_w:
                reg_score = reg_score + L2(w[self.K + self.K**2: len(w) - self.F*self.K])
        elif self.reg == "L1" or self.reg == "Elastic":
            reg_score = reg_score + self.lamb * L1(w[self.K:self.K + self.K**2])
            if self.extra_w:
                reg_score = reg_score + L1(w[self.K + self.K**2: len(w) - self.F*self.K])
        return reg_score

    @property
    def Lambda(self):
        lam = super(LinearDiscreteHawkes, self).Lambda
        w = self.W[self.K:, :]
        assert w.shape == (self.K, self.K), "unmatched w shape: should be KxK"
        Rs = []
        for t in range(self.conv_dyad_data.shape[2]):
            Rs.append(self.conv_dyad_data[:, :, t] * w)
        Rs = np.concatenate(Rs, axis=1).reshape(self.conv_dyad_data.shape, order='F')
        return np.concatenate([lam[i, :, :] + Rs[:, :, i] for i in range(self.data.shape[0])]).reshape(lam.shape)
