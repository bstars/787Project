import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from RobustRegression import RobustRegression


class RRSafeOpt():
	def __init__(self,
	             dim : int,
	             f : Callable,
	             xrange : np.array,
	             seed_set : np.array,
	             threshold : float,
	             rr_kde_bandwidth : float = 0.1):
		"""
		:param dim:
		:param f:
		:param xrange:
			region to explore
			if dim = 1, then xrange has shape [batch,] or [batch, 1]
			otherwise, xrange has shape [batch, dim]
		:param seed_set:
			safe set to begin with
			if dim = 1, then seed_set has shape [batch,] or [batch, 1]
			otherwise, seed_set has shape [batch, dim]
		:param threshold:
		:param beta:
		"""

		self.rr = RobustRegression(dim = dim, kde_bandwidth=rr_kde_bandwidth)
		self.dim = dim
		self.f = f
		if self.dim == 1:
			if len(xrange.shape) == 1:
				xrange = xrange[:, None]
			if len(seed_set.shape) == 1:
				seed_set = seed_set[:, None]
		self.xrange = xrange
		self.threshold = threshold

		self.observations = np.concatenate([seed_set, f(seed_set)[:,None], np.ones([len(seed_set),1])], axis=1)
		self.rr.fit(self.observations[:, :-2], self.observations[:, -2], self.xrange, safe_threshold=self.threshold)

		self.C = np.zeros([len(self.xrange), 2])
		self.C[:, 0] = -np.inf
		self.C[:, 1] = np.inf
		self.S = None
		self.update_set()

	def update_set(self):
		mus, sigmas = self.rr.predict(self.xrange)
		stds = np.sqrt(sigmas)

		lbs = mus - 1.96 * stds
		ubs = mus + 1.96 * stds

		self.S = np.where(lbs > self.threshold)[0]
		self.C[:,0] = lbs
		self.C[:,1] = ubs

	def propose_evaluation(self):

		if len(self.S) == 0:
			idx = np.argmax(self.C[:,1] - self.C[:,0])
		else:
			# The subset of safe region where it's likely to contain the maximum
			lmax = np.max(self.C[self.S, 0])
			likely = np.where(self.C[self.S, 1] >= (lmax - 1e-3))[0]
			M = self.S[likely]

			# M = self.S

			idx = np.argmax(self.C[M, :][:, 1] - self.C[M, :][:, 0])
			idx = M[idx]
		if self.dim == 1:
			return self.xrange[idx, 0]
		return self.xrange[idx]

	def add_observation(self, x, y):
		if isinstance(x, float):
			append = np.array([[x, y, y > self.threshold]])

		elif len(x.shape) == 1 and self.dim == 1:
			x = x[:, None]
			y = y[:, None]
			safe = y > self.threshold
			append = np.concatenate([x, y, safe], axis=1)
		elif len(x.shape) == 1 and self.dim > 1:
			x = x[None, :]
			y = y[None, :]
			safe = y > self.threshold
			append = np.concatenate([x, y, safe], axis=1)
		else:
			y = y[:, None]
			safe = y > self.threshold
			append = np.concatenate([x, y, safe], axis=1)
		self.observations = np.concatenate([self.observations, append], axis=0)
		self.rr.fit(self.observations[:, :-2], self.observations[:, -2], self.xrange, safe_threshold=self.threshold)
		self.update_set()

def eg():
	def truef(xs):
		xs = xs[:,0]
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	num = 500
	xs = np.linspace(-5, 5, num)
	xs = np.stack([xs, np.sin(xs) * np.cos(xs + 0.7), np.log(np.abs(xs) + 3)]).T
	ys = truef(xs)
	threshold = 0.1


	# seed_idx = np.random.randint(0, num, size=10)
	seed_idx = np.random.choice(np.where(ys > threshold)[0], size=3, replace=False)
	seed_set = xs[seed_idx]
	rrsafeopt = RRSafeOpt(dim=xs.shape[1], f=truef, xrange=xs, seed_set=seed_set, threshold=threshold, rr_kde_bandwidth=0.2)

	for i in range(100):

		x = rrsafeopt.propose_evaluation()

		plt.figure(figsize=(10, 6))
		plt.plot(xs[:,0], ys, label='True function')
		plt.plot(xs[:,0], np.ones([len(xs)]) * rrsafeopt.threshold, label='Threshold')

		mus, sigmas = rrsafeopt.rr.predict(xs)
		plt.plot(xs[:,0], mus, label='$\mu$')
		# plt.fill_between(xs[:,0], mus - 1.96 * np.sqrt(sigmas), mus + 1.96 * np.sqrt(sigmas), alpha=0.5, label='$95\%$ Confidence Interval')
		plt.fill_between(rrsafeopt.xrange[:,0], rrsafeopt.C[:,0], rrsafeopt.C[:,1], alpha=0.5, label='$95\%$ Confidence Interval')
		plt.scatter(rrsafeopt.observations[:, 0], rrsafeopt.observations[:, -2], label='Observations')
		plt.scatter(rrsafeopt.xrange[rrsafeopt.S,:][:,0], np.ones([len(rrsafeopt.S)]) * rrsafeopt.threshold, label='Safe sets')
		plt.scatter(x[0], truef(x[None,:]), label='Proposed evaluation point')
		plt.legend()
		plt.show()


		rrsafeopt.add_observation(x, truef(x[None,:]))

if __name__ == '__main__':
	eg()