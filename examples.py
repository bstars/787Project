import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch import nn

from ProbabilisticRegressor import GaussianProc, RobustRegression, RobustNNRegression
from SaferOpt import SaferOpt
from AcquisitionFunction import UCB, DynamicUCB, ExpectedImprovement, ProbOfImprovement, EntropySearch
from RandomFunction import GPRandomFunction, NNRandomFunction
from CartPole import CartPole



def bayesian_optimization():
	def truef(xs):
		return 2 * (np.sin(xs) * np.cos(xs + 0.7) + np.log(np.abs(xs) + 3) - 1.)


	num = 500
	xs = np.linspace(-5, 5, num)
	truef = GPRandomFunction(1, xs, kernel_func=GaussianProc.RBFKernel(0.1), means=0., seed=66)
	# truef = NNRandomFunction(
	# 	dim=1, xrange=np.linspace(-10, 10, 150),
	# 	yrange=[-5., 5.], nn_dim=[1, 128, 128, 128, 1],
	# 	act=nn.ReLU
	# )
	start_idx = 200

	xs_true = xs.copy()
	f_true = truef(xs_true)

	gp = GaussianProc(dim=1, kernel_func=GaussianProc.RBFKernel(1.), measure_noise=0.01, gaussian_mean=0.1)
	evaluates = np.array([[xs[start_idx], truef(xs[start_idx])]])
	xs = np.concatenate([xs[:start_idx], xs[start_idx + 1:]], axis=0)
	gp.fit(evaluates[:,0], evaluates[:,1])

	# acqui = DynamicUCB()
	acqui = UCB()
	# acqui = ProbOfImprovement()
	# acqui = ExpectedImprovement()



	for i in range(10):

		fbest = np.max(evaluates[:,1])

		mu, std = gp.predict(xs)
		# best_idx = np.argmax(mu + std)
		best_idx = acqui(mu, std, r_best = fbest)
		print(xs.shape, best_idx, mu.shape, std.shape)
		x_choice = xs[best_idx]
		f_choice = truef(x_choice)

		plt.plot(xs_true, f_true, label='true function')
		plt.scatter(x_choice, f_choice, label='Next Choice', c='red')
		plt.scatter(evaluates[:,0], evaluates[:,1], label='Evaluations', c='blue')
		plt.fill_between(xs, mu - std, mu + std, color='red', alpha=0.15, label='$\mu \pm \sigma$')
		plt.legend()
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

def robust_nn_regression():
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


	src_idx = np.random.randint(0, n, size=5) # uniformly sample source data
	trg_idx = np.setdiff1d(np.arange(n), src_idx)

	src_idx = np.sort(src_idx)
	trg_idx = np.sort(trg_idx)

	Xsrc = xs[src_idx]
	ysrc = ys[src_idx]
	Xtrg = xs[trg_idx]
	ytrg = ys[trg_idx]

	rr = RobustNNRegression(dim=1, nn_dim=[1, 4, 4, 4], kde_bandwidth=0.1)
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



	# truef = GPRandomFunction(1, xs, kernel_func=GaussianProc.RBFKernel(0.5), means=0.2, seed=787)
	truef = NNRandomFunction(
		dim=1, xrange=np.linspace(-20, 20, 100),
		yrange=[-5.,5.], nn_dim=[1, 64, 64, 64, 1],
		act=nn.LeakyReLU
	)

	num = 500
	xs = np.linspace(-10, 10, num)
	ys = truef(xs)
	threshold = -0.3

	# seed_idx = np.random.randint(0, num, size=10)
	original_state = np.random.get_state()
	np.random.seed(0)
	seed_idx = np.random.choice(np.where(ys > threshold)[0], size=10, replace=False)
	np.random.set_state(original_state)
	# seed_idx = np.array([20, 300, 400, 480])
	seed_set = xs[seed_idx]

	rewards_seed_set = ys[seed_idx]
	safety_seed_set = ys[seed_idx, None]

	saferopt = SaferOpt(
		dim=1,
		xrange=xs,
		seed_set=seed_set,
		rewards_seed_set=rewards_seed_set,
		safety_seed_set=safety_seed_set,
		thresholds=np.array([threshold]),
		regressor_cls=RobustNNRegression,
		acquisition_cls=DynamicUCB,
		# acquisition_cls=ExpectedImprovement,
		# acquisition_cls=ProbOfImprovement,
		nn_dim=[1, 8, 8, 4],
		kde_bandwidth=0.1,
		act=nn.Tanh
	)

	# saferopt = SaferOpt(
	# 	dim=1,
	# 	xrange=xs,
	# 	seed_set=seed_set,
	# 	rewards_seed_set=rewards_seed_set,
	# 	safety_seed_set=safety_seed_set,
	# 	thresholds=np.array([threshold]),
	# 	regressor_cls=GaussianProc,
	# 	# acquisition_cls=DynamicUCB,
	# 	acquisition_cls=ExpectedImprovement,
	# 	kernel_func = GaussianProc.RBFKernel(1.),
	# 	cholesky_factor=True,
	# )

	fbest = np.max(rewards_seed_set)
	true_best = np.max(ys)
	history = []


	for i in range(40):
		x = saferopt.propose_evaluation()
		print('iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f' % (i, true_best, fbest, truef(x)))

		plt.figure(figsize=(10, 6))
		plt.plot(xs, ys, label='True function')
		plt.plot(xs, np.ones([len(xs)]) * saferopt.thresholds[0], label='Threshold')

		plt.fill_between(saferopt.xrange[:,0], saferopt.C[:,-2] - 1.96 * saferopt.C[:,-1], saferopt.C[:,-2] + 1.96 * saferopt.C[:,-1], alpha=0.1, label='reward Confidence Interval', color='r')
		# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,-2]+ rrsafeopt.C[:,-1]) / 2, label='reward mean', color='r')
		plt.fill_between(saferopt.xrange[:,0], saferopt.C[:,0] - 1.96 * saferopt.C[:,1], saferopt.C[:,0] + 1.96 * saferopt.C[:,1], alpha=0.1, label='safety Confidence Interval', color='b')
		# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,0]+ rrsafeopt.C[:,1]) / 2, label='safety mean', color='b')

		plt.scatter(saferopt.observations[:, 0], saferopt.observations[:, -1], label='Observations')
		plt.scatter(saferopt.xrange[saferopt.S, :][:, 0], np.ones([len(saferopt.S)]) * saferopt.thresholds[0],
		            label='Safe sets')
		plt.scatter(x, truef(x), label='Proposed evaluation point', marker='v')
		plt.legend()
		plt.title("Safer Optimization")
		plt.savefig('./%d.jpg' % i)
		plt.show()

		saferopt.add_observation(x, truef(x), np.array([truef(x)]))
		fbest = max(fbest, truef(x))
		history.append(truef(x))

	plt.plot(np.arange(len(history)), history)
	plt.plot(np.arange(len(history)), np.ones([len(history)]) * true_best)
	plt.plot(history)
	plt.show()

def inverted_pendulum():
	env = CartPole()
	state = [0., 0., 0., 0.]
	while True:
		env.show_cart(state, 0.1)
		action = np.random.uniform(-1, 1, size=1)
		state, reward, terminate = env.simulate(action, state)
		if terminate:
			break




if __name__ == '__main__':
	bayesian_optimization()
	# safer_optimization()
	# robust_nn_regression()
	# inverted_pendulum()