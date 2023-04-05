import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from GaussianProc import GaussianProc



class SafeOpt():
	def __init__(self,
	             dim : int,
	             f : Callable,
	             L : float,
	             xrange : np.array,
	             seed_set : np.array,
	             threshold : float,
	             beta : Callable,
	             meassure_noise : float,
	             gp_kernel_func : Callable,
	             gp_prior_mean : float = 0.):
		"""
		:param dim:
		:param f:
		:param L:
		:param xrange:
		:param seed_set:
		:param threshold:
		:param beta:
		:param meassure_noise:
		"""
		self.dim = dim

		self.gp = GaussianProc(
			dim=dim,
			kernel_func=gp_kernel_func,
			measure_noise=meassure_noise,
			gaussian_mean=gp_prior_mean
		)
		f_seed_set = f(seed_set)
		self.gp.add_observation(seed_set, f_seed_set)

		self.L = L
		self.beta = beta
		self.threshold = threshold
		self.meassure_noise = meassure_noise

		if self.dim == 1:
			if len(xrange.shape) == 1:
				xrange = xrange[:, None]
			if len(seed_set.shape) == 1:
				seed_set = seed_set[:, None]

		self.X = np.concatenate([seed_set, xrange], axis=0)

		self.safe_set = [i for i in range(len(seed_set))]

		# line 2-4 in Algorithm 1
		# We modify the initialization of C at seed set.
		# In the original algorithm, if the lower bound of C is initialized with safe threshold, then we can never expand the safe set.
		self.C = np.zeros([len(self.X), 2])
		self.C[:, 0] = -np.inf
		self.C[:, 1] = np.inf
		self.C[self.safe_set, 0] = f_seed_set - self.meassure_noise
		self.C[self.safe_set, 1] = f_seed_set + self.meassure_noise

		self.Q = np.zeros([len(self.X), 2])
		self.Ld = self.L * pairwise_distances(self.X, self.X, metric='euclidean')
		self.t = 0

		self.update_set()

	def update_set(self):
		mus, stds = self.gp.predict(self.X)
		self.Q[:, 0] = mus - self.beta(self.t) * stds
		self.Q[:, 1] = mus + self.beta(self.t) * stds

		# line 6 in Algorithm 1: C := C \intersect Q
		self.C[:, 0] = np.maximum(self.C[:, 0], self.Q[:, 0])
		self.C[:, 1] = np.minimum(self.C[:, 1], self.Q[:, 1])

		# line 7 in Algorithm 1
		Ld = self.Ld[:, self.safe_set]
		l = self.C[self.safe_set,0][None, :]
		self.safe_set = np.where(np.any(l - Ld > self.threshold, axis=1))[0]

		self.t += 1

	def propose_evaluation(self):

		# Line 8 in Algorithm 1
		# Excluding the points that upon evaluation, cannot expand the safe set
		non_safe = np.setdiff1d(np.arange(len(self.X)), self.safe_set)
		expand = self.C[self.safe_set, 1][:, None] - self.Ld[self.safe_set, :][:, non_safe] >= self.threshold
		G = np.where(np.sum(expand, axis=1) > 0)[0]
		G = self.safe_set[G] # idx of safe points in self.X

		# Line 9 in Algorithm 1
		# The subset of safe region where it's likely to contain the maximum
		lmax = np.max(self.C[self.safe_set, 0])
		likely = np.where(self.C[self.safe_set, 1] >= lmax)[0]
		M = self.safe_set[likely]

		MIG, r1, r2 = np.intersect1d(G, M, return_indices=True)
		if len(MIG) == 0:
			MIG = M
		# idx = np.argmax(self.C[MIG, :][:, 1] - self.C[MIG, :][:, 0])
		idx = np.argmax(self.C[MIG, :][:, 1])
		idx = MIG[idx]

		if self.dim == 1:
			return self.X[idx, 0], M, G
		return self.X[idx], M, G

	def add_observation(self, x, y):
		self.gp.add_observation(x, y)
		self.update_set()


def safe_optimization():
	def truef(xs):
		xs = xs * 1
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	def rbf_kernel(x1, x2):
		return np.exp(-0.5 * np.sum( (x1-x2)**2 ))

	num = 500
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)

	# plt.plot(xs, ys)
	# plt.show()

	seed_set = np.array([-3, 3])
	L = np.max(np.abs(
		(ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
	)) + 0.01  # overapproximation of Lipschitz constant


	so = SafeOpt(
		dim=1,
		f=truef,
		L=L,
		xrange=xs,
		seed_set=seed_set,
		threshold=0.1,
		beta=lambda t: 3 * 0.9 ** t,
		meassure_noise=0.05,
		gp_kernel_func=rbf_kernel,
		gp_prior_mean=0.2
	)

	for i in range(30):

		x, M, G = so.propose_evaluation()
		# plt.scatter(x, truef(x), label='Proposed evaluation point')

		plt.figure(figsize=(10, 6))
		plt.plot(xs, ys, label='True function')
		plt.plot(xs, np.ones([len(xs)]) * so.threshold, label='Threshold')
		plt.fill_between(so.X[2:, 0], so.C[2:, 0], so.C[2:, 1], alpha=0.5, label='Confidence Interval')
		plt.scatter(so.gp.X[:, 0], so.gp.Y, label='Observations')
		plt.scatter(so.X[so.safe_set][:, 0], np.ones([len(so.safe_set)]) * so.threshold, label='Safe Region')
		plt.scatter(x, truef(x), label='Proposed evaluation point')



		plt.legend()
		plt.show()

		so.add_observation(x, truef(x))

if __name__ == '__main__':
	safe_optimization()

