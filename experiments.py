import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch import nn

from ProbabilisticRegressor import GaussianProc, RobustRegression, RobustNNRegression
from SaferOpt import SaferOpt
from AcquisitionFunction import DynamicUCB, ExpectedImprovement, ProbOfImprovement, EntropySearch
from RandomFunction import GPRandomFunction, NNRandomFunction


def experiment(truef, saferopt, rewards_seed_set, ys, xs, acqui_str):
	fbest = np.max(rewards_seed_set)
	true_best = np.max(ys)
	proposed = []
	best_achieved = []

	for i in range(50):
		print(saferopt.beta(saferopt.t))
		x = saferopt.propose_evaluation()
		print('iteration %d, true best: %.3f, best achieved: %.3f, proposed: %.3f' % (i, true_best, fbest, truef(x)))

		if i % 10 == 0:
			plt.figure(figsize=(10, 6))
			plt.plot(xs, ys, label='True function')
			plt.plot(xs, np.ones([len(xs)]) * saferopt.thresholds[0], label='Threshold')

			plt.fill_between(saferopt.xrange[:, 0], saferopt.C[:, -2] - 1.96 * saferopt.C[:, -1],
			                 saferopt.C[:, -2] + 1.96 * saferopt.C[:, -1], alpha=0.1, label='reward Confidence Interval',
			                 color='r')
			# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,-2]+ rrsafeopt.C[:,-1]) / 2, label='reward mean', color='r')
			bt = saferopt.beta(saferopt.t)
			plt.fill_between(saferopt.xrange[:, 0], saferopt.C[:, 0] - bt * 1.96 * saferopt.C[:, 1],
			                 saferopt.C[:, 0] + bt * 1.96 * saferopt.C[:, 1], alpha=0.1, label='safety Confidence Interval',
			                 color='b')
			# plt.plot(rrsafeopt.xrange[:,0], (rrsafeopt.C[:,0]+ rrsafeopt.C[:,1]) / 2, label='safety mean', color='b')

			plt.scatter(saferopt.observations[:, 0], saferopt.observations[:, -1], label='Observations')
			plt.scatter(saferopt.xrange[saferopt.S, :][:, 0], np.ones([len(saferopt.S)]) * saferopt.thresholds[0],
			            label='Safe sets')
			plt.scatter(x, truef(x), label='Proposed evaluation point', marker='v')
			plt.legend()
			plt.title(acqui_str + ', iteration %d' % i)
			plt.savefig('./temp_result/%s_%d.jpg' % (acqui_str, i))
			# plt.show()
			plt.close()
		saferopt.add_observation(x, truef(x), np.array([truef(x)]))
		fbest = max(fbest, truef(x))
		proposed.append(truef(x))
		best_achieved.append(fbest)

	plt.plot(np.arange(len(proposed)), proposed, label='proposed')
	plt.plot(np.arange(len(proposed)), np.ones([len(proposed)]) * true_best, label='true best')
	plt.plot(np.arange(len(best_achieved)), best_achieved, label='best achieved')
	plt.legend()
	plt.savefig('./temp_result/history_%s.jpg' % (acqui_str))
	plt.close()
	plt.show()

def experiment_acquisition(truef, rewards_seed_set, safety_seed_set, ys, xs):
	for acqui in [DynamicUCB, ExpectedImprovement, ProbOfImprovement]:
		saferopt = SaferOpt(
			dim=1,
			xrange=xs,
			seed_set=seed_set,
			rewards_seed_set=rewards_seed_set,
			safety_seed_set=safety_seed_set,
			thresholds=np.array([threshold]),
			regressor_cls=RobustNNRegression,
			acquisition_cls=acqui,
			# acquisition_cls=ExpectedImprovement,
			# acquisition_cls=ProbOfImprovement,
			nn_dim=[1, 8, 16, 8, 4],
			kde_bandwidth=0.1,
			act=nn.Tanh,
			beta=lambda t : 1 / np.log(t / 4 + 2),
		)
		experiment(truef, saferopt, rewards_seed_set, ys, xs, acqui_str=acqui.__name__)

def experiment_regressor(truef, rewards_seed_set, safety_seed_set, ys, xs):
	opts = [
		SaferOpt(
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
			nn_dim=[1, 8, 16, 8, 4],
			kde_bandwidth=0.1,
			act=nn.Tanh,
			beta=lambda t: 1 / np.log(t / 4 + 2),
		),
		SaferOpt(
			dim=1,
			xrange=xs,
			seed_set=seed_set,
			rewards_seed_set=rewards_seed_set,
			safety_seed_set=safety_seed_set,
			thresholds=np.array([threshold]),
			regressor_cls=GaussianProc,
			acquisition_cls=DynamicUCB,
			# acquisition_cls=ExpectedImprovement,
			kernel_func = GaussianProc.RBFKernel(1.),
			cholesky_factor=True,
			beta=lambda t: 1 / np.log(t / 4 + 2)
		)
	]

	for safer_opt in opts:
		experiment(truef, safer_opt, rewards_seed_set, ys, xs, acqui_str=str(type(safer_opt.reward_regressor)))



if __name__ == '__main__':
	pass
	truef = NNRandomFunction(
		dim=1, xrange=np.linspace(-20, 20, 100),
		yrange=[-5., 5.], nn_dim=[1, 64, 64, 64, 1],
		act=nn.LeakyReLU
	)


	num = 500
	xs = np.linspace(-10, 10, num)
	truef = NNRandomFunction(
		dim=1, xrange=np.linspace(-10, 10, 50),
		yrange=[-1.5, 2.], nn_dim=[1, 64, 64, 64, 1],
		act=nn.ReLU
	)
	# truef = GPRandomFunction(1, xs, kernel_func=GaussianProc.RBFKernel(0.5), means=0.3)
	ys = truef(xs)
	threshold = -0.3

	# seed_idx = np.random.randint(0, num, size=10)
	original_state = np.random.get_state()

	seed_idx = np.random.choice(
		np.where( np.logical_and(threshold < ys, ys < threshold + 1))[0],
		size=6, replace=False
	)
	# seed_idx = np.array([20, 300, 400, 480])
	seed_set = xs[seed_idx]

	rewards_seed_set = ys[seed_idx]
	safety_seed_set = ys[seed_idx, None]

	# experiment_acquisition(truef, rewards_seed_set, safety_seed_set, ys, xs)
	experiment_regressor(truef, rewards_seed_set, safety_seed_set, ys, xs)




