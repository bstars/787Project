import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import torch
from torch import nn


class RobustNNRegression():
	def __init__(self, dim, nn_dim, kde_bandwidth=0.1):
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
			nn.Linear(16, 32), act(),
			nn.Linear(32, 32), act(),
			nn.Linear(32, 32), act(),
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
		self.X_mean = np.mean(Xsrc, axis=0)
		self.X_std = np.std(Xsrc, axis=0)
		self.y_mean = np.mean(ysrc, axis=0)
		self.y_std = np.std(ysrc, axis=0)

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


def eg2():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)

	def truef_(xs):
		ys = np.zeros_like(xs)
		ys[xs<0] = - xs[xs<0]
		ys[xs>0] = xs[xs>0]
		return ys

	num = 500
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	n = len(ys)

	# pts = np.linspace(-2, 2, n)
	# ps = norm(0.35, 0.95).pdf(pts)
	# src_idx = np.random.choice(np.arange(n), size=20, replace=False, p=ps / np.sum(ps)) # gaussian sample source data


	src_idx = np.random.randint(0, n, size=20) # uniformly sample source data
	trg_idx = np.setdiff1d(np.arange(n), src_idx)

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = xs[src_idx]
	ysrc = ys[src_idx]
	Xtrg = xs[trg_idx]
	ytrg = ys[trg_idx]

	rr = RobustNNRegression(dim=1, nn_dim=8, kde_bandwidth=0.1)
	rr.fit(Xsrc, ysrc, Xtrg, max_iteration=10000, verbose=True, primal_lr=1e-4, dual_lr=1e-4, safe_threshold=None)

	mus, sigma = rr.predict(Xtrg)

	plt.scatter(Xsrc, ysrc, facecolor='none', edgecolor='red', s=30, label='Source data')
	plt.scatter(Xtrg, ytrg, c='black', s=5, label='Target data')
	plt.plot(Xtrg, mus, label='$\mu$')

	plt.fill_between(Xtrg, mus - 1.96 * np.sqrt(sigma), mus + 1.96 * np.sqrt(sigma),
	                 alpha=0.2,
	                 color='red', label='$95\%$ confidence interval')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	eg2()



