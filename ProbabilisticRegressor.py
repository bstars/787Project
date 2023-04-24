import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import torch
from torch import nn

class ProbabilisticRegressor():
	def __init__(self, **kwargs):
		super().__init__()

	def fit(self, **kwargs):
		pass

	def predict(self, **kwargs):
		pass

class GaussianProc(ProbabilisticRegressor):
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

	def fit(self, x, y, safe_threshold=None, **kwargs):
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

		self.K = pairwise_distances(x, x, metric=self.kernel_func)
		self.X = x
		self.Y = y
		self.gaussian_mean = np.mean(y)
		if safe_threshold is not None:
			self.gaussian_mean = safe_threshold

		# print(self.K.shape, self.X.shape, self.y.shape)



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
		# print(temp.shape,(self.Y - self.gaussian_mean).shape)
		u_21 = self.gaussian_mean + temp.T @ (self.Y - self.gaussian_mean)
		# u_21 = temp.T @ self.Y
		sigma_21 = np.sqrt(np.diag(K_22 - K_12.T @ temp))

		if len(u_21) == 0:
			return u_21[0], sigma_21[0]

		return u_21, sigma_21
	
class RobustRegression(ProbabilisticRegressor):
	def __init__(self, dim, kde_bandwidth=0.1, reg=1e-4, lr=2e-3):
		super().__init__()
		self.dim = dim
		M = np.eye(dim + 2)
		self.M11 = M[0,0]
		self.M12 = M[0,1:]

		self.kde_bandwidth = kde_bandwidth
		self.reg = reg
		self.lr = lr

		self.kde_src = None
		self.kde_trg = None


		self.X_mean = None
		self.X_std = None
		self.y_mean = None
		self.X_std = None

	def fit(self, Xsrc, ysrc, Xtrg, max_iteration=10000, beta1=0.9, beta2=0.999, verbose=False, safe_threshold=None):
		"""

		Args:
			Xsrc (np.array):
				Samples from source domain
				Shape [n_src_samples, dim] if dim != 1
				Shape [n_src_samples,] if dim == 1
			ysrc (np.array):
				Labels of samples from source domain
				Shape [n_src_samples,]
			Xtrg (np.array):
				Samples from target domain
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1

		Returns:
		"""

		# pre-computation


		self.X_mean = np.mean(Xsrc, axis=0)
		self.X_std = np.std(Xsrc, axis=0)
		self.y_mean = np.mean(ysrc, axis=0)
		self.y_std = np.std(ysrc, axis=0)

		Xsrc = (Xsrc - self.X_mean) / self.X_std
		Xtrg = (Xtrg - self.X_mean) / self.X_std
		ysrc = (ysrc - self.y_mean) / self.y_std

		self.mu0 =  (np.max(ysrc) + np.min(ysrc)) / 2
		self.sigma0 = (0.5 * (np.max(ysrc) - self.mu0)) ** 2

		if safe_threshold is not None:
			self.mu0 = (safe_threshold - self.y_mean) / self.y_std



		# self.mu0 = np.mean(ysrc)
		# self.sigma0 = np.std(ysrc) ** 2

		n_src_samples = len(ysrc)
		if len(Xsrc.shape) == 1:
			Xsrc = Xsrc[:, None]
			Xtrg = Xtrg[:, None]

		X1 = np.concatenate([Xsrc, np.ones([n_src_samples, 1])], axis=1)
		C = np.concatenate([ysrc[:, None], X1], axis=1)
		C = C.T @ C / n_src_samples

		kde_src = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth)
		kde_src.fit(Xsrc)
		kde_trg = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth)
		kde_trg.fit(Xtrg)

		self.kde_src = kde_src
		self.kde_trg = kde_trg


		Pratio = np.exp(kde_src.score_samples(Xsrc) - kde_trg.score_samples(Xsrc))


		# Adam momentum
		M11_1stM = 0
		M11_2ndM = 0
		M12_1stM = np.zeros_like(self.M12)
		M12_2ndM = np.zeros_like(self.M12)

		# Dual gradient
		for t in range(1, max_iteration + 1):

			# P^*(y|x)
			# Evaluate primal variable at current dual variable
			sigma = 1 / (1 / self.sigma0 + 2 * self.M11 * Pratio)
			mus = sigma * (-2 * X1 @ self.M12 * Pratio + self.mu0 / self.sigma0)

			# apply gradient descent on dual variable
			g11 = 0
			g12 = np.zeros_like(self.M12)

			for i in range(n_src_samples):
				g12 += mus[i] * X1[i] / n_src_samples
				g11 += (mus[i] ** 2 + sigma[i]) / n_src_samples

			g11 -= C[0, 0]
			g12 -= C[0, 1:]


			# Adam moentum
			M11_1stM = beta1 * M11_1stM + (1 - beta1) * g11
			M11_2ndM = beta2 * M11_2ndM + (1 - beta2) * g11 * g11
			M12_1stM = beta1 * M12_1stM + (1 - beta1) * g12
			M12_2ndM = beta2 * M12_2ndM + (1 - beta2) * g12 * g12

			first_unbias = 1 / (1 - beta1 ** t)
			second_unbias = 1 / (1 - beta2 ** t)

			dg11 = (M11_1stM / first_unbias) / np.sqrt(M11_2ndM / second_unbias + 1e-6)
			dg12 = (M12_1stM / first_unbias) / np.sqrt(M12_2ndM / second_unbias + 1e-6)

			self.M11 += self.lr * (
					dg11 - self.reg * self.M11
			)
			self.M12 += self.lr * (
					dg12 - self.reg * self.M12
			)


			# check convergence
			# gnorm = np.linalg.norm(g11) ** 2 + np.linalg.norm(g12) ** 2
			gnorm = np.sqrt(g11 **2 + np.sum(g12 ** 2))
			print(t, gnorm) if verbose else None
			if gnorm < 1e-7:
				return

	def predict(self, Xtrg):
		"""

		Args:
			Xtrg (np.array):
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1
		Returns:
		"""
		Xtrg = (Xtrg - self.X_mean) / self.X_std
		if len(Xtrg.shape) == 1:
			Xtrg = Xtrg[:,None]
		X1 = np.concatenate([Xtrg, np.ones([len(Xtrg), 1])], axis=1)

		Pratio = np.exp(self.kde_src.score_samples(Xtrg) - self.kde_trg.score_samples(Xtrg))

		sigma = 1 / (1 / self.sigma0 + 2 * self.M11 * Pratio)
		mus = sigma * (-2 * X1 @ self.M12 * Pratio + self.mu0 / self.sigma0)

		return mus * self.y_std + self.y_mean, sigma * self.y_std ** 2

class RobustNNRegression(ProbabilisticRegressor):
	def __init__(self, dim, nn_dim, kde_bandwidth=0.1):
		super().__init__()
		self.dim = dim
		self.nn_dim = nn_dim
		self.kde_bandwidth = kde_bandwidth

		self.M11 = torch.ones([1,], requires_grad=False) * 0.1
		self.M12 = torch.ones([nn_dim,], requires_grad=False) * 0.1

		self.kde_src = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
		self.kde_trg = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)

		self.X_mean = None
		self.X_std = None
		self.y_mean = None
		self.X_std = None

		self.mu0 = None
		self.sigma0 = None


		act = nn.LeakyReLU

		self.nn = nn.Sequential(
			nn.Linear(dim, 16), act(),
			# nn.Linear(16, 32), act(),
			# nn.Linear(32, 32), act(),
			nn.Linear(16, 32), act(),
			nn.Linear(32, nn_dim)
		)

	def gaussian_params(self, nn_x, M11, M12, mu0, sigma0, Pratio):
		"""
		:param nn_x: torch.Tensor, shape [batch, dim]
		:param M11: torch.Tensor, shape [1,]
		:param M12:  torch.Tensor, shape [dim,]
		:param mu0: scalar or torch.Tensor, shape [1,]
		:param sigma0: scalar or torch.Tensor, shape [1,]
		:param Pratio: torch.Tensor, shape [batch,]
		"""
		sigmas = 1 / (1 / sigma0 + 2 * M11 * Pratio)
		mus = sigmas * (-2 * Pratio * (nn_x @ M12) + mu0 / sigma0)
		return mus, sigmas

	def fit(self,
	        Xsrc,
	        ysrc,
	        Xtrg,
	        verbose=False,
	        max_iteration=10000,
	        primal_lr=5e-4,
	        dual_lr=5e-4,
	        safe_threshold=None):
		"""
		Args:
			Xsrc (np.array):
				Samples from source domain
				Shape [n_src_samples, dim] if dim != 1
				Shape [n_src_samples,] if dim == 1
			ysrc (np.array):
				Labels of samples from source domain
				Shape [n_src_samples,]
			Xtrg (np.array):
				Samples from target domain
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1

		Returns:
		"""

		if len(Xsrc.shape) == 1:
			Xsrc = Xsrc[:, None]
			Xtrg = Xtrg[:, None]


		# Normalization
		self.X_mean = np.mean(Xsrc, axis=0, keepdims=True)
		self.X_std = np.std(Xsrc, axis=0, keepdims=True)
		self.y_mean = np.mean(ysrc, axis=0, keepdims=False)
		self.y_std = np.std(ysrc, axis=0, keepdims=False)

		Xsrc = (Xsrc - self.X_mean) / self.X_std
		Xtrg = (Xtrg - self.X_mean) / self.X_std
		ysrc = (ysrc - self.y_mean) / self.y_std

		self.mu0 = (np.max(ysrc) + np.min(ysrc)) / 2
		self.sigma0 = (0.5 * (np.max(ysrc) - self.mu0)) ** 2

		if safe_threshold is not None:
			self.mu0 = (safe_threshold - self.y_mean) / self.y_std


		self.kde_src.fit(Xsrc)
		self.kde_trg.fit(Xtrg)

		# P_src(x) / P_trg(x)
		Pratio_src = np.exp(self.kde_src.score_samples(Xsrc) - self.kde_trg.score_samples(Xsrc))
		Pratio_src = torch.from_numpy(Pratio_src).float()
		Xsrc = torch.from_numpy(Xsrc).float()
		ysrc = torch.from_numpy(ysrc).float()


		optimizer = torch.optim.Adam(self.nn.parameters(), lr=primal_lr, weight_decay=1e-4)
		for t in range(1, max_iteration):

			# Dual update
			self.nn.eval()
			nn_x = self.nn(Xsrc)
			mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio_src)
			dM11 = torch.mean(mus**2 + sigmas - ysrc**2)
			dM12 = 2 * torch.mean( torch.einsum('ij,i->ij', nn_x, mus - ysrc) , dim=0)

			if verbose:
				print(t, dM11, dM12)

			self.M11.data += dual_lr * dM11
			self.M12.data += dual_lr * dM12

			# Primal update
			for _ in range(1):
				self.nn.train()
				nn_x = self.nn(Xsrc)
				mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio_src)
				mus = mus.detach()

				# loss = 2 * torch.mean(
				# 	torch.einsum('ij,i->ij', nn_x, mus - ysrc) @ self.M12
				# )

				loss = 2 * torch.mean(
					torch.einsum('ij,i->ij', nn_x, ysrc - mus) @ self.M12
				) # This is where I got confused

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	def predict(self, Xtrg):
		"""
		Args:
			Xtrg (np.array):
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1
		Returns:
		"""

		if len(Xtrg.shape) == 1:
			Xtrg = Xtrg[:, None]

		Xtrg = (Xtrg - self.X_mean) / self.X_std

		Pratio = np.exp(self.kde_src.score_samples(Xtrg) - self.kde_trg.score_samples(Xtrg))
		Pratio = torch.from_numpy(Pratio).float()
		Xtrg_th = torch.from_numpy(Xtrg).float()
		nn_x = self.nn(Xtrg_th)


		mus, sigmas = self.gaussian_params(nn_x, self.M11, self.M12, self.mu0, self.sigma0, Pratio)
		mus = mus.detach().numpy()
		sigmas = sigmas.detach().numpy()
		return mus * self.y_std + self.y_mean, sigmas * self.y_std ** 2

