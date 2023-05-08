import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class AcquisitionFunction():
	def __init__(self):
		self.t = 0

	def __call__(self, means, stds, **kwargs):
		pass

class UCB(AcquisitionFunction):
	def __init__(self):
		super().__init__()

	def __call__(self, means, stds, alpha = 1, **kwargs):
		return np.argmax(means + alpha * stds, axis=0)

class DynamicUCB(AcquisitionFunction):
	def __init__(self):
		super().__init__()

	# def __call__(self, means, stds, **kwargs):
	# 	# t = np.sqrt(self.t)
	# 	# s = sigmoid(t - 5)
	#
	# 	t = self.t
	# 	s = sigmoid(t - 3.5)
	#
	# 	print(t, s)
	# 	acq = s * means + (1 - s) * stds
	# 	self.t += 0.05
	# 	return np.argmax(acq, axis=0)

	def __call__(self, means, stds, **kwargs):
		# t = np.sqrt(self.t)
		# s = sigmoid(t - 5)

		t = self.t
		s = sigmoid(t / 20 - 2.6)
		print(t, s)
		acq = s * means + (1 - s) * stds
		self.t += 1
		return np.argmax(acq, axis=0)



class ExpectedImprovement(AcquisitionFunction):
	def __init__(self):
		super().__init__()

	def __call__(self, means, stds, r_best, **kwargs):
		# means: [batch,]
		# stds: [batch,]
		# r_best: float
		EIs = (means - r_best) * (1 - norm.cdf((r_best - means) / stds)) \
			+ stds * norm.pdf((r_best - means) / stds)
		return np.argmax(EIs, axis=0)



class ProbOfImprovement(AcquisitionFunction):
	def __init__(self):
		super().__init__()

	def __call__(self, means, stds, r_best, **kwargs):
		# means: [batch,]
		# stds: [batch,]
		# r_best: float
		PoI = norm.cdf((means - r_best) / stds)
		return np.argmax(PoI, axis=0)

class EntropySearch(AcquisitionFunction):
	def __init__(self):
		super().__init__()
		raise NotImplementedError

if __name__ == '__main__':
	xs = np.linspace(-5, 5, 100)
	ys = sigmoid(xs)
	b = 3
	a = 2

	ts = np.arange(10)
	ys2 = sigmoid(ts / a - b)

	plt.plot(xs, ys, 'b-', label='sigmoid')
	plt.plot(ts / a - b, ys2, 'ro', label='t = 1, 2, ..., 10')
	plt.legend()
	plt.show()

