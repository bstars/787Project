import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt


class GaussianProc():
	"""
	Gaussian process that assume 0 mean and 0 variance (perfect measurement)
	"""
	def __init__(self, dim:int, kernel_func : Callable, measure_noise=0.1, gaussian_mean = 0.):
		super().__init__()
		self.dim = dim
		self.kernel_func = kernel_func
		self.X = np.zeros([0, self.dim])
		self.Y = np.zeros([0])
		self.K = np.zeros([0, 0])
		self.measure_noise = measure_noise
		self.gaussian_mean = gaussian_mean

	def add_observation(self, x, y):
		"""
		:param x: np.array, [dim] or [batch, dim]
		:param y: float or np.array with shape [batch]
		:return:
		:rtype:
		"""
		if isinstance(x, float):
			# single point dim 1
			x = np.array([[x]])
			y = np.array([y])
		elif len(x.shape) == 1 and self.dim == 1:
			# multiple points dim 1
			x = x[:,None]
		elif len(x.shape) == 1:
			# single point high dim
			x = x[None, :]
			y = np.array([y])

		K_22 = pairwise_distances(x, x, metric=self.kernel_func)

		if len(self.X) == 0:
			self.X = x
			self.Y = y
			self.K = K_22
			return

		K_12 = pairwise_distances(self.X, x, metric=self.kernel_func)
		self.K = np.concatenate([
			np.concatenate([self.K, K_12.T], axis=0),
			np.concatenate([K_12, K_22], axis=0)
		], axis=1)

		self.X = np.concatenate([self.X, x], axis=0)
		self.Y = np.concatenate([self.Y, y], axis=0)

	def predict(self, x):
		"""
		:param x: np.array, [dim] or [batch, dim]
		:return:
			mu : posterior mean
			sigma : posterior std
		"""
		if isinstance(x, float):
			x = np.array([[x]])
		elif len(x.shape) == 1 and self.dim == 1:
			x = x[:,None]
		elif len(x.shape) == 1:
			x = x[None, :]

		K_12 = pairwise_distances(self.X, x, metric=self.kernel_func)
		K_22 = pairwise_distances(x, x, metric=self.kernel_func)
		temp = np.linalg.solve(self.K + np.eye(len(self.X)) * self.measure_noise**2, K_12)

		u_21 = self.gaussian_mean + temp.T @ (self.Y - self.gaussian_mean)
		# u_21 = temp.T @ self.Y
		sigma_21 = np.sqrt(np.diag(K_22 - K_12.T @ temp))

		if len(u_21) == 0:
			return u_21[0], sigma_21[0]

		return u_21, sigma_21

def bayesian_optimization():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.)

	def rbf_kernel(x1, x2):
		return np.exp(-0.5 * np.sum( (x1-x2)**2 ))

	num = 500
	xs = np.linspace(-5, 5, num)
	start_idx = 200

	xs_true = xs.copy()
	f_true = truef(xs_true)

	gp = GaussianProc(dim=1, kernel_func=rbf_kernel, measure_noise=0.01, gaussian_mean=0.1)
	gp.add_observation(xs[start_idx], truef(xs[start_idx]))
	evaluates = np.array([[xs[start_idx], truef(xs[start_idx])]])
	xs = np.concatenate([xs[:start_idx], xs[start_idx + 1:]], axis=0)



	for i in range(10):

		mu, std = gp.predict(xs)
		best_idx = np.argmax(mu + std)
		print(xs.shape, best_idx, mu.shape, std.shape)
		x_choice = xs[best_idx]
		f_choice = truef(x_choice)

		plt.plot(xs_true, f_true, label='true function')
		plt.scatter(x_choice, f_choice, label='Next Choice', c='red')
		plt.scatter(evaluates[:,0], evaluates[:,1], label='Evaluations', c='blue')
		plt.fill_between(xs, mu - std, mu + std, color='red', alpha=0.15, label='$\mu \pm \sigma$')
		plt.legend()
		# plt.savefig('./%d.jpg' % i)
		# plt.close()
		plt.show()

		evaluates = np.concatenate([evaluates, np.array([[x_choice, f_choice]]) ], axis=0)
		xs = np.concatenate([ xs[:best_idx], xs[best_idx+1:] ], axis=0)

		gp.add_observation(x_choice, f_choice)



if __name__ == '__main__':
	bayesian_optimization()
	# gp = GaussianProc(dim=1, kernel_func=lambda x1, x2: np.exp(-0.5 * np.sum( (x1-x2)**2 )), measure_noise=0.01)
	# gp.add_observation(1., 1.)
	# mu, sigma = gp.predict(np.array([1., 1.]))
	# print(mu, sigma)