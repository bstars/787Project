""" Utility functions adapted from https://github.com/befelix/SafeOpt/blob/master/safeopt/utilities.py """

from collections import Sequence
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from ProbabilisticRegressor import GaussianProc


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.
    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).
    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)

    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T

class RandomFunction():
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

class GPRandomFunction(RandomFunction):
    def __init__(self, dim, xrange, kernel_func, noise=0.1, means=0., seed=0):
        """
        :param dim:
        :param xrange:
        :param means:
        """
        super().__init__()
        self.dim = dim
        self.kernel_func = kernel_func
        if self.dim == 1 and len(xrange.shape) == 1:
            xrange = xrange[:, None]
        self.gp = GaussianProc(dim, kernel_func=kernel_func, measure_noise=noise, gaussian_mean=means)
        self.K = pairwise_distances(xrange, xrange, metric=self.kernel_func)
        self.means = means
        self.xrange = xrange

        original_state = np.random.get_state()
        np.random.seed(seed)
        self.ys = np.random.multivariate_normal(
            mean=self.means * np.ones(len(xrange)),
            cov=self.K,
        )
        np.random.set_state(original_state)

        self.gp.fit(self.xrange, self.ys, K=self.K)

    def __call__(self, x):
        mean, var = self.gp.predict(x)
        # sample = np.random.normal(mean, np.sqrt(var))
        sample = mean
        return sample.squeeze()

class NNRandomFunction(RandomFunction):
    """ Generate Neural network random function by random sampling points and overfitting a neural network """
    def __init__(self, dim, xrange, yrange, nn_dim, act=nn.LeakyReLU):
        super().__init__()
        self.dim = dim
        if self.dim == 1 and len(xrange.shape) == 1:
            xrange = xrange[:, None]

        layers = []
        for i in range(len(nn_dim) - 2):
            layers.append(nn.Linear(nn_dim[i], nn_dim[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(nn_dim[-2], nn_dim[-1]))
        self.nn = nn.Sequential(*layers)
        ys = np.random.uniform(low=yrange[0], high=yrange[1], size=[len(xrange)])
        ys = np.random.normal(np.mean(ys), np.std(ys), size=[len(xrange)])

        # plt.plot(xrange[:,0], ys)
        # plt.show()

        xrange = torch.from_numpy(xrange).float()
        ys = torch.from_numpy(ys).float()
        loss = nn.L1Loss()
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-4, weight_decay=0.)
        for i in range(3000):
            optimizer.zero_grad()
            y_pred = self.nn(xrange)
            output = loss(y_pred[:,0], ys)
            output.backward()
            print(output.item())
            optimizer.step()

        self.ys = y_pred.squeeze().detach().numpy()

    def __call__(self, x):
        if isinstance(x, float) and self.dim == 1:
            x = torch.from_numpy(np.array([[x]])).float()
        elif self.dim == 1 and len(x.shape) == 1:
            x = torch.from_numpy(x[:, None]).float()

        return self.nn(x).squeeze().detach().numpy()





def eg_gp_random_function():
    xrange = np.linspace(-5, 5, 500)
    gp = GPRandomFunction(1, xrange, kernel_func=GaussianProc.RBFKernel(0.1), means=0., seed=787)
    plt.plot(xrange, gp(xrange))
    plt.show()

def eg_nn_random_function():
    xrange = np.linspace(-5, 5, 50)
    yrange = np.array([-5, 5])
    nrf = NNRandomFunction(1, xrange, yrange, nn_dim=[1, 64, 64, 64, 1], act=nn.Tanh)

    xs = np.linspace(-5, 5, 400)
    plt.plot(xs, nrf(xs))
    plt.show()

if __name__ == '__main__':
    # eg_gp_random_function()
    eg_nn_random_function()






