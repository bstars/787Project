import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt

from ProbabilisticRegressor import GaussianProc, RobustRegression, RobustNNRegression, ProbabilisticRegressor

class SaferOpt():
	def __init__(self,
	             dim:int,
	             xrange : np.array,
	             seed_set: np.array,
	             rewards_seed_set : np.array,
	             safety_seed_set : np.array,
	             thresholds : np.array,
	             beta : Callable,
	             regressor_cls,
	             **kwargs):
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
		:param regressor_cls:
		:param kwargs: Initialize parameters for regressor
		"""
		self.dim = dim
		self.beta = beta
		self.t = 0
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
		self.observations = np.concatenate([seed_set, safety_seed_set, rewards_seed_set[:, None]], axis=1)
		batch, self.num_safety = safety_seed_set.shape
		self.reward_regressor = regressor_cls(dim=dim, **kwargs)
		self.safety_regressor = [
			regressor_cls(dim=dim, **kwargs) for _ in range(self.num_safety)
		]
		self.reward_regressor.fit(
			self.observations[:, :dim], self.observations[:, -1], Xtrg=self.xrange,
			safe_threshold=None, max_iteration=1000
		)

		[
			self.safety_regressor[i].fit(self.observations[:, :dim], self.observations[:, dim + i], Xtrg=self.xrange,
			                       safe_threshold=self.thresholds[i] - 0.1, max_iteration=1000
			                       )
			for i in range(self.num_safety)
		]

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
		mus, sigmas = self.reward_regressor.predict(self.xrange)
		stds = np.sqrt(sigmas)

		lbs = mus - 1.96 * self.beta(self.t) * stds
		ubs = mus + 1.96 * self.beta(self.t) * stds

		self.C[:,-2] = lbs
		self.C[:,-1] = ubs

		for i in range(self.num_safety):
			mus, sigmas = self.safety_regressor[i].predict(self.xrange)
			stds = np.sqrt(sigmas)

			lbs = mus - 1.96 * self.beta(self.t) * stds
			ubs = mus + 1.96 * self.beta(self.t) * stds
			self.C[:,2 * i] = lbs
			self.C[:, 2*i+1] = ubs


		self.S = np.where(np.all(self.C[:,:2*self.num_safety:2] > self.thresholds[None,:], axis=1, keepdims=False))[0]
		self.t += 1

	def propose_evaluation(self, strategy='ucb'):
		"""

		:param strategy: can be "UCB" or "M
		:return:
		:rtype:
		"""
		if len(self.S) == 0:
			raise ValueError("No safe set found")

		# The subset of safe region where it's likely to contain the maximum
		# lmax = np.max(self.C[self.S, -2])
		# likely = np.where(self.C[self.S, -1] >= lmax)[0]
		# M = self.S[likely]

		M = self.S
		if self.t % 3 == 0:
			idx = np.argmax(self.C[M, :][:, -1] - self.C[M, :][:, -2])  # maximum uncertainty
		else:
			idx = np.argmax(self.C[M, :][:, -1])  # upper confidence bound

		# idx = np.argmax(self.C[M, :][:, -1])

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

		self.reward_regressor.fit(
			self.observations[:, :self.dim], self.observations[:, -1], Xtrg=self.xrange,
			safe_threshold=None, max_iteration=1500
		)

		[
			self.safety_regressor[i].fit(self.observations[:, :self.dim], self.observations[:, self.dim + i], Xtrg=self.xrange,
			                       safe_threshold=self.thresholds[i]-0.1, max_iteration=1500)
			for i in range(self.num_safety)
		]

		self.update_set()

