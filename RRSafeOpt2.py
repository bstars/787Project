import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from RobustRegression import RobustRegression
from RobustNNRegression import RobustNNRegression

class RRSafeOpt():
	def __init__(self,
	             dim:int,
	             xrange : np.array,
	             seed_set : np.array,
	             rewards_seed_set : np.array,
	             safety_seed_set : np.array,
	             thresholds : np.array,
	             beta : Callable,
	             rr_kde_bandwidth : float = 0.1):
		"""

		:param dim: Problem dimension
		:param xrange:
			region to explore
			if dim = 1, then xrange has shape [batch,] or [batch, 1]
			otherwise, xrange has shape [batch, dim]
		:param seed_set:
			[batch, dim]
			safe set to begin with
			if dim = 1, then seed_set has shape [batch,] or [batch, 1]
			otherwise, seed_set has shape [batch, dim]
		:param rewards_seed_set:
			[batch,]
			reward function of the seed set
		:param safety_seed_set:
			[batch, num_safety]
			safety function of the seed set, num_safety is the number of safety constraints
		:param thresholds:
			[num_safety]
			threshold function of the seed set, num_safety is the number of safety constraints
		:param beta:
			Schedule of number of stds
		:param rr_kde_bandwidth:
		"""

		self.dim = dim
		self.beta = beta
		self.t = 1
		self.thresholds = thresholds

		if self.dim == 1:
			if len(xrange.shape) == 1:
				xrange = xrange[:, None]
			if len(seed_set.shape) == 1:
				seed_set = seed_set[:, None]

		self.xrange = xrange

		# Observations: [batch, dim + num_safety + 1]
		# [batch, :dim] is the observation points
		# [batch, dim:dim+num_safety] is the safety values
		# [batch, -1] is the reward values
		self.observations = np.concatenate([seed_set, safety_seed_set, rewards_seed_set[:,None]], axis=1)

		batch, self.num_safety = safety_seed_set.shape
		self.reward_rr = RobustNNRegression(dim = dim, nn_dim = 16, kde_bandwidth=rr_kde_bandwidth)
		self.safety_rrs = [
			RobustNNRegression(dim = dim, nn_dim = 16, kde_bandwidth=rr_kde_bandwidth)
			for _ in range(self.num_safety)
		]
		# print(self.observations[:, :dim].shape, self.observations[:, -1].shape)
		self.reward_rr.fit(
			self.observations[:, :dim], self.observations[:, -1], self.xrange,
			safe_threshold=None, max_iteration=1000
		)

		[
			self.safety_rrs[i].fit(self.observations[:, :dim], self.observations[:, dim + i], self.xrange,
				safe_threshold=self.thresholds[i]-0.1, max_iteration=1000
			)
			for i in range(self.num_safety)
		]

		# The confidence interval for the reward function and safety functions
		# C[:,[-2,-1]] is the confidence interval for the reward function
		# C[:,0] is the estimated lower bound for the first safety function
		# C[:,1] is the estimated lower bound for the second safety function, etc
		self.C = np.zeros([len(self.xrange), 2 + self.num_safety * 2])
		self.C[::2] = -np.inf
		self.C[:, 1::2] = np.inf
		self.S = None

		self.update_set()

	def update_set(self):
		"""
		Update the confidence interval and safe set
		:return:
		:rtype:
		"""
		mus, sigmas = self.reward_rr.predict(self.xrange)
		stds = np.sqrt(sigmas)

		lbs = mus - 1.96 * self.beta(self.t) * stds
		ubs = mus + 1.96 * self.beta(self.t) * stds

		self.C[:,-2] = lbs
		self.C[:,-1] = ubs

		for i in range(self.num_safety):
			mus, sigmas = self.safety_rrs[i].predict(self.xrange)
			stds = np.sqrt(sigmas)

			lbs = mus - 1.96 * self.beta(self.t) * stds
			ubs = mus + 1.96 * self.beta(self.t) * stds
			self.C[:,2 * i] = lbs
			self.C[:, 2*i+1] = ubs


		self.S = np.where(np.all(self.C[:,:2*self.num_safety:2] > self.thresholds[None,:], axis=1, keepdims=False))[0]
		self.t += 1

	def propose_evaluation(self):
		if len(self.S) == 0:
			raise ValueError("No safe set found")

		# The subset of safe region where it's likely to contain the maximum
		lmax = np.max(self.C[self.S, -2])
		likely = np.where(self.C[self.S, -1] >= lmax)[0]
		M = self.S[likely]

		M = self.S
		if self.t % 2 == 0:
			idx = np.argmax(self.C[M, :][:, -1] - self.C[M, :][:, -2])  # maximum uncertainty
		else:
			idx = np.argmax(self.C[M, :][:, -1])  # upper confidence bound

		idx = M[idx]
		if self.dim == 1:
			return self.xrange[idx, 0]
		return self.xrange[idx]

	def add_observation(self, x, reward, safety):
		"""
		:param x: [batch, dim] or [batch,] or float
		:param reward: [batch] or float
		:param safety: [batch, num_safety] or [num_safety]
		:return:
		:rtype:
		"""

		if isinstance(x, float): # 1d case, 1 example
			# print(self.observations.shape)
			append = np.concatenate([[x], safety, [reward]])[None,:]

		elif len(x.shape) == 1 and self.dim == 1: # 1d case, multiple examples
			x = x[:, None]
			reward = reward[:, None]
			append = np.concatenate([x, safety, reward], axis=1)
		elif len(x.shape) == 1 and self.dim > 1: # nd case, 1 example
			append = np.concatenate(
				[x[None, :], safety[None,:], np.array([[reward]])],
				axis=1
			)
		else: # nd case, multiple examples
			append = np.concatenate([x, safety, reward[:, None]], axis=1)

		self.observations = np.concatenate([self.observations, append], axis=0)

		self.reward_rr.fit(
			self.observations[:, :self.dim], self.observations[:, -1], self.xrange,
			safe_threshold=None, max_iteration=1000
		)

		[
			self.safety_rrs[i].fit(self.observations[:, :self.dim], self.observations[:, self.dim + i], self.xrange,
			                       safe_threshold=self.thresholds[i]-0.1, max_iteration=1000)
			for i in range(self.num_safety)
		]

		self.update_set()





def eg():
	def truef(xs):
		return 2 * (np.sin(xs) + np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	num = 500
	xs = np.linspace(-10, 10, num)
	ys = truef(xs)
	threshold = 0.1

	# seed_idx = np.random.randint(0, num, size=10)
	seed_idx = np.random.choice(np.where(ys > threshold)[0], size=3, replace=False)
	seed_set = xs[seed_idx]

	rewards_seed_set = ys[seed_idx]
	safety_seed_set = ys[seed_idx, None]

	rrsafeopt = RRSafeOpt(
		dim=1,
		xrange=xs,
		seed_set=seed_set,
		rewards_seed_set=rewards_seed_set,
		safety_seed_set=safety_seed_set,
		thresholds=np.array([threshold]),
		beta=lambda t: 1 / np.sqrt(np.sqrt(t + 1)),
		# beta=lambda t: 1,
		rr_kde_bandwidth=0.1,
	)

	fbest = np.max(rewards_seed_set)
	true_bets = np.max(ys)

	for i in range(100):
		print('iteration %d, true best: %.3f, best achieved: %.3f' % (i, true_bets, fbest))
		x = rrsafeopt.propose_evaluation()

		plt.figure(figsize=(10, 6))
		plt.plot(xs, ys, label='True function')
		plt.plot(xs, np.ones([len(xs)]) * rrsafeopt.thresholds[0], label='Threshold')

		# plot the reward
		plt.fill_between(rrsafeopt.xrange[:,0], rrsafeopt.C[:,-2], rrsafeopt.C[:,-1], alpha=0.1, label='reward Confidence Interval', color='r')
		# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,-2]+ rrsafeopt.C[:,-1]) / 2, label='reward mean', color='r')
		plt.fill_between(rrsafeopt.xrange[:,0], rrsafeopt.C[:,0], rrsafeopt.C[:,1], alpha=0.1, label='safety Confidence Interval', color='b')
		# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,0]+ rrsafeopt.C[:,1]) / 2, label='safety mean', color='b')

		plt.scatter(rrsafeopt.observations[:, 0], rrsafeopt.observations[:, -1], label='Observations')
		plt.scatter(rrsafeopt.xrange[rrsafeopt.S, :][:, 0], np.ones([len(rrsafeopt.S)]) * rrsafeopt.thresholds[0],
		            label='Safe sets')
		plt.scatter(x, truef(x), label='Proposed evaluation point', marker='v')
		plt.legend()
		plt.show()

		rrsafeopt.add_observation(x, truef(x), np.array([truef(x)]))
		fbest = max(fbest, truef(x))

if __name__ == '__main__':
	eg()
	# l = [1,2,3,4,5,6,7,8,9,0]
	# print(l[:4:2])



