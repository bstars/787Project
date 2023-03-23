import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from scipy.stats import norm


class RobustRegression():
	def __init__(self, dim, kde_bandwidth=0.3, reg=1e-4, lr=2e-3):
		self.dim = dim
		M = np.eye(dim + 2)
		self.M11 = M[0,0]
		self.M12 = M[0,1:]
		# self.M22 = M[1:,1:]
		self.M22 = np.ones([dim + 1, dim + 1])
		self.kde_bandwidth = kde_bandwidth
		self.reg = reg
		self.lr = lr

		self.kde_src = None
		self.kde_trg = None



	def fit(self, Xsrc, ysrc, Xtrg, max_iteration=10000):
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
		self.mu0 =  (np.max(ysrc) + np.min(ysrc)) / 2
		self.sigma0 = (0.5 * (np.max(ysrc) - self.mu0))**2

		# self.mu0 = np.mean(ysrc)
		# self.sigma0 = np.var(ysrc)

		n_src_samples = len(ysrc)
		if len(Xsrc.shape) == 1:
			Xsrc = Xsrc[:,None]
			Xtrg = Xtrg[:, None]


		X1 = np.concatenate([Xsrc, np.ones([n_src_samples, 1])], axis=1)
		C = np.concatenate([ysrc[:, None], X1], axis=1)
		E_src_X1X1 = X1.T @ X1 / n_src_samples
		C = C.T @ C / n_src_samples

		kde_src = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth)
		kde_src.fit(Xsrc)
		kde_trg = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth)
		kde_trg.fit(Xtrg)

		self.kde_src = kde_src
		self.kde_trg = kde_trg


		Pratio = np.exp(kde_src.score_samples(Xsrc) - kde_trg.score_samples(Xsrc))

		# Dual gradient
		for _ in range(max_iteration):

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

			g11 -= C[0,0]
			g12 -= C[0,1:]
			g22 = E_src_X1X1 - C[1:,1:]

			self.M11 += self.lr * (
				g11  - self.reg * self.M11
			)
			self.M12 += self.lr * (
				g12  - self.reg * self.M12
			)
			self.M22 += self.lr * (
				g22  - self.reg * self.M22
			)

			# check convergence
			gnorm = np.linalg.norm(g11)**2 + np.linalg.norm(g12)**2 + np.linalg.norm(g22)**2
			print(gnorm)
			if gnorm < 1e-4:
				return

	def predict(self, Xtrg):
		"""

		Args:
			Xtrg (np.array):
				Shape [n_trg_samples, dim] if dim != 1
				Shape [n_trg_samples,] if dim == 1
		Returns:

		"""
		if len(Xtrg.shape) == 1:
			Xtrg = Xtrg[:,None]
		X1 = np.concatenate([Xtrg, np.ones([len(Xtrg), 1])], axis=1)

		Pratio = np.exp(self.kde_src.score_samples(Xtrg) - self.kde_trg.score_samples(Xtrg))

		sigma = 1 / (1 / self.sigma0 + 2 * self.M11 * Pratio)
		mus = sigma * (-2 * X1 @ self.M12 * Pratio + self.mu0 / self.sigma0)

		return mus, sigma


def eg():

	# An example that tends to reimplement figure 2 in the paper
	# Get some wierd result, need to check the detail with Angie

	data = np.loadtxt('./Hahn1.txt')
	y = data[:, 0]
	x = data[:, 1]

	idx = np.argsort(x)
	x = x[idx]
	y = y[idx]

	n = len(y)
	pts = np.linspace(-2, 2, n)
	ps = norm(0, 1).pdf(pts)

	src_idx = np.random.choice(np.arange(n), size=90, replace=False, p=ps / np.sum(ps))
	trg_idx = np.setdiff1d(np.arange(n), src_idx)

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = x[src_idx]
	ysrc = y[src_idx]
	Xtrg = x[trg_idx]
	ytrg = y[trg_idx]

	rr = RobustRegression(1, kde_bandwidth=0.1, reg=1e-6, lr=2e-3)
	rr.fit((Xsrc - 300) / 600, (ysrc - 10) / 20, (Xtrg - 300) / 600, max_iteration=10000)
	mus, sigma = rr.predict((Xtrg - 300) / 600)

	plt.scatter(Xsrc, ysrc, facecolor='none', edgecolor='red', s=30, label='Source')
	plt.scatter(Xtrg, ytrg, c='black', s=5, label='Target')
	plt.plot(Xtrg, mus * 20 + 10, label='$\mu$')
	plt.fill_between(Xtrg, 20 * (mus + 1 * np.sqrt(sigma)) + 10, 20 * (mus - 1 * np.sqrt(sigma)) + 10, alpha=0.2,
	                 color='red', label='$ \mu \pm 1 * \sigma$')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	eg()










