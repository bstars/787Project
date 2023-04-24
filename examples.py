import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import norm

from ProbabilisticRegressor import GaussianProc, RobustRegression, RobustNNRegression
from SaferOpt import SaferOpt

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
	evaluates = np.array([[xs[start_idx], truef(xs[start_idx])]])
	xs = np.concatenate([xs[:start_idx], xs[start_idx + 1:]], axis=0)
	gp.fit(evaluates[:,0], evaluates[:,1])




	for i in range(5):

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
		plt.title("Bayesian Optimization")
		plt.show()

		evaluates = np.concatenate([evaluates, np.array([[x_choice, f_choice]]) ], axis=0)
		xs = np.concatenate([ xs[:best_idx], xs[best_idx+1:] ], axis=0)

		gp.fit(evaluates[:,0], evaluates[:,1])

def robust_regression_1():

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
	ps = norm(0.35, 0.95).pdf(pts)

	src_idx = np.random.choice(np.arange(n), size=90, replace=False, p=ps / np.sum(ps))
	trg_idx = np.setdiff1d(np.arange(n), src_idx)

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = x[src_idx]
	ysrc = y[src_idx]
	Xtrg = x[trg_idx]
	ytrg = y[trg_idx]


	rr = RobustRegression(1, kde_bandwidth=0.15, reg=1e-9, lr=1e-2)
	rr.fit(Xsrc, ysrc, Xtrg, max_iteration=300, verbose=False)
	mus, sigma = rr.predict(Xtrg)

	plt.scatter(Xsrc, ysrc, facecolor='none', edgecolor='red', s=30, label='Source data')
	plt.scatter(Xtrg, ytrg, c='black', s=5, label='Target data')
	plt.plot(Xtrg, mus, label='$\mu$')


	plt.fill_between(Xtrg, mus - 1.96 * np.sqrt(sigma), mus + 1.96 * np.sqrt(sigma),
	                 alpha=0.2,
	                 color='red', label='$95\%$ confidence interval')
	plt.legend()
	plt.title("Robust Regression 1")
	plt.show()

def robust_regression_2():
	def truef(xs):
		xs = xs * 1
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2)



	num = 500
	xs = np.linspace(-5, 5, num)
	ys = truef(xs)
	xs = np.stack([xs, np.sin(xs) * np.cos(xs + 0.7), np.log(np.abs(xs) + 3)]).T

	n = len(ys)
	pts = np.linspace(-2, 2, n)
	ps = norm(0.35, 0.95).pdf(pts)
	src_idx = np.random.choice(np.arange(n), size=100, replace=False, p=ps / np.sum(ps))

	# src_idx = np.random.choice(np.where(ys > 0)[0], size=30, replace=False)

	trg_idx = np.setdiff1d(np.arange(n), src_idx)
	# trg_idx = np.arange(len(ys))

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = xs[src_idx]
	ysrc = ys[src_idx]
	Xtrg = xs[trg_idx]
	ytrg = ys[trg_idx]

	rr = RobustRegression(dim = xs.shape[1], kde_bandwidth=0.15, reg=1e-13, lr=1e-3)
	rr.fit(Xsrc, ysrc, xs, max_iteration=1000, verbose=True)


	mus, sigma = rr.predict(Xtrg)

	plt.scatter(Xsrc[:,0], ysrc, facecolor='none', edgecolor='red', s=30, label='Source data')
	plt.scatter(Xtrg[:,0], ytrg, c='black', s=5, label='Target data')
	plt.plot(Xtrg[:,0], mus, label='$\mu$')

	plt.fill_between(Xtrg[:,0], mus - 1.96 * np.sqrt(sigma), mus + 1.96 * np.sqrt(sigma),
	                 alpha=0.2,
	                 color='red', label='$95\%$ confidence interval')
	plt.legend()
	plt.title("Robust Regression 2")
	plt.show()

def robsut_nn_regression():
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
	plt.title("Robust NN Regression ")
	plt.show()

def safer_optimization():
	def truef(xs):
		return 2 * (np.sin(xs) + np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.2) + 0.2

	num = 500
	xs = np.linspace(-10, 10, num)
	ys = truef(xs)
	threshold = 0.3

	# seed_idx = np.random.randint(0, num, size=10)
	# seed_idx = np.random.choice(np.where(ys > threshold)[0], size=4, replace=False)
	seed_idx = np.array([20, 300, 480])
	seed_set = xs[seed_idx]




	rewards_seed_set = ys[seed_idx]
	safety_seed_set = ys[seed_idx, None]


	# rrsafeopt = SaferOpt(
	# 	dim=1,
	# 	xrange=xs,
	# 	seed_set=seed_set,
	# 	rewards_seed_set=rewards_seed_set,
	# 	safety_seed_set=safety_seed_set,
	# 	thresholds=np.array([threshold]),
	# 	beta=lambda t: 1 / np.sqrt(np.sqrt(t + 1)),
	# 	regressor_cls=RobustNNRegression,
	# 	nn_dim=16,
	# 	kde_bandwidth=0.1,
	# )

	rrsafeopt = SaferOpt(
		dim=1,
		xrange=xs,
		seed_set=seed_set,
		rewards_seed_set=rewards_seed_set,
		safety_seed_set=safety_seed_set,
		thresholds=np.array([threshold]),
		beta=lambda t: 1 / np.sqrt(np.sqrt(t + 1)),
		regressor_cls=GaussianProc,
		kernel_func = lambda x1, x2 : np.exp(-0.5 * np.sum((x1 - x2) ** 2))
	)

	# rrsafeopt = SaferOpt(
	# 	dim=1,
	# 	xrange=xs,
	# 	seed_set=seed_set,
	# 	rewards_seed_set=rewards_seed_set,
	# 	safety_seed_set=safety_seed_set,
	# 	thresholds=np.array([threshold]),
	# 	beta=lambda t: 1 / np.sqrt(np.sqrt(t + 1)),
	# 	regressor_cls=RobustRegression,
	# )

	fbest = np.max(rewards_seed_set)
	true_bets = np.max(ys)

	for i in range(100):
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
		print('iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f' % (i, true_bets, fbest, truef(x)))
		plt.savefig('./%d.jpg' % i)
		plt.title("Safer Optimization")
		plt.show()

		rrsafeopt.add_observation(x, truef(x), np.array([truef(x)]))
		fbest = max(fbest, truef(x))

if __name__ == '__main__':
	# bayesian_optimization()
	# robust_regression_1()
	# robust_regression_2()
	# robsut_nn_regression()
	safer_optimization()
