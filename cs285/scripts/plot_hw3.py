import tensorflow as tf
from tensorflow.python.summary import summary_iterator
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _summary_iterator(test_dir):
	"""Reads events from test_dir/events.

	Args:
	test_dir: Name of the test directory.

	Returns:
	A summary_iterator
	"""
	event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
	return tf.train.summary_iterator(event_paths[-1])


if __name__ == "__main__":
	# import argparse
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--question', '-q', type=int, required=True)
	# args = parser.parse_args()


	## QUESTION 1


	for logdir_prefix in ["dqn_q1"]:
		if os.path.exists("../data"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
		elif os.path.exists("../../run_logs"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../run_logs')
		else:
			raise "Data path doesn't exist"
		with_prefixes = [p for p in os.listdir(data_path) if p.startswith(logdir_prefix)]
		logdirs = [os.path.join(data_path, p) for p in with_prefixes]

		iterations = []
		returns = []
		best_returns = []
		best_iterations = []
		labels = [p[6:p.find("_Pong")] for p in with_prefixes]

		for logdir in logdirs:
			iterations.append([])
			returns.append([])
			best_returns.append([])
			best_iterations.append([])
			for summary in _summary_iterator(logdir):
				if len(summary.summary.value) == 1:
					if summary.summary.value[0].tag == "Train_AverageReturn":
						iterations[-1].append(summary.step)
						returns[-1].append(summary.summary.value[0].simple_value)
					if summary.summary.value[0].tag == "Train_BestReturn":
						best_returns[-1].append(summary.summary.value[0].simple_value)
						best_iterations[-1].append(summary.step)

		for i, r, l in zip(iterations, returns, labels):
			plt.plot(i, r, label=l + " avg")
		for i, r, l in zip(best_iterations, best_returns, labels):
			plt.plot(i, r, label=l + " best")
		plt.title(logdir_prefix)
		plt.ylabel("return")
		plt.xlabel("iteration")
		plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
		plt.legend()
		plt.savefig(logdir_prefix)
		plt.close()

	## QUESTION 2

	for logdir_prefix, c in zip(["dqn_q2_dqn", "dqn_double"], ["orange", "blue"]):
		if os.path.exists("../data"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
		elif os.path.exists("../../run_logs"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../run_logs')
		else:
			raise "Data path doesn't exist"
		with_prefixes = [p for p in os.listdir(data_path) if p.startswith(logdir_prefix)]
		logdirs = [os.path.join(data_path, p) for p in with_prefixes]

		iterations = []
		returns = []
		# best_returns = []
		# best_iterations = []
		labels = [p[:p.find("_Lunar")] for p in with_prefixes]

		for logdir in logdirs:
			iterations.append([])
			returns.append([])
			# best_returns.append([])
			# best_iterations.append([])
			for summary in _summary_iterator(logdir):
				if len(summary.summary.value) == 1:
					if summary.summary.value[0].tag == "Train_AverageReturn":
						iterations[-1].append(summary.step)
						returns[-1].append(summary.summary.value[0].simple_value)
					# if summary.summary.value[0].tag == "Train_BestReturn":
					# 	best_returns[-1].append(summary.summary.value[0].simple_value)
					# 	best_iterations[-1].append(summary.step)

		for i, r, l in zip(iterations, returns, labels):
			plt.plot(i, r, label=l, alpha=.2, color=c)
		it = np.array(iterations[0])
		ret = np.mean([np.array(r) for r in returns], axis=0)
		plt.plot(it, ret, label=logdir_prefix + ' AVG', color=c)
		# for i, r, l in zip(best_iterations, best_returns, labels):
		# 	plt.plot(i, r, label=l + " best", alpha=.3)
	plt.title("DQN vs. DOUBLE DQN")
	plt.ylabel("return")
	plt.xlabel("iteration")
	plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	plt.legend()
	plt.savefig("dqn_double")
	plt.close()


	## QUESTION 3

	for logdir_prefix in ["dqn_co", "dqn_in", "dqn_de", "dqn_q1"]:
		if os.path.exists("../data"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
		elif os.path.exists("../../run_logs"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../run_logs')
		else:
			raise "Data path doesn't exist"
		with_prefixes = [p for p in os.listdir(data_path) if p.startswith(logdir_prefix)]
		logdirs = [os.path.join(data_path, p) for p in with_prefixes]

		iterations = []
		returns = []
		# best_returns = []
		# best_iterations = []
		labels = [p[:p.find("_Pong")] for p in with_prefixes]

		for logdir in logdirs:
			iterations.append([])
			returns.append([])
			# best_returns.append([])
			# best_iterations.append([])
			for summary in _summary_iterator(logdir):
				if len(summary.summary.value) == 1:
					if summary.summary.value[0].tag == "Train_AverageReturn":
						iterations[-1].append(summary.step)
						returns[-1].append(summary.summary.value[0].simple_value)
					# if summary.summary.value[0].tag == "Train_BestReturn":
					# 	best_returns[-1].append(summary.summary.value[0].simple_value)
					# 	best_iterations[-1].append(summary.step)

		for i, r, l in zip(iterations, returns, labels):
			plt.plot(i, r, label=l)
		# for i, r, l in zip(best_iterations, best_returns, labels):
		# 	plt.plot(i, r, label=l + " best")
	plt.title("Hyper Parameter Search Learning Rate")
	plt.ylabel("return")
	plt.xlabel("iteration")
	plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	plt.legend()
	plt.savefig("dqn_hyperparameter")
	plt.close()

	## QUESTION 4

	for logdir_prefix in ["CartPole"]:
		if os.path.exists("../data"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
		elif os.path.exists("../../run_logs"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../run_logs')
		else:
			raise "Data path doesn't exist"
		with_prefixes = [p for p in os.listdir(data_path) if logdir_prefix in p]
		logdirs = [os.path.join(data_path, p) for p in with_prefixes]

		iterations = []
		returns = []
		# best_returns = []
		# best_iterations = []
		labels = [p[:p.find("_CartPole")] for p in with_prefixes]

		for logdir in logdirs:
			iterations.append([])
			returns.append([])
			# best_returns.append([])
			# best_iterations.append([])
			for summary in _summary_iterator(logdir):
				if len(summary.summary.value) == 1:
					if summary.summary.value[0].tag == "Train_AverageReturn":
						iterations[-1].append(summary.step)
						returns[-1].append(summary.summary.value[0].simple_value)
					# if summary.summary.value[0].tag == "Train_BestReturn":
					# 	best_returns[-1].append(summary.summary.value[0].simple_value)
					# 	best_iterations[-1].append(summary.step)

		for i, r, l in zip(iterations, returns, labels):
			plt.plot(i, r, label=l)
		# for i, r, l in zip(best_iterations, best_returns, labels):
		# 	plt.plot(i, r, label=l + " best")
	plt.title("Actor Critic CartPole")
	plt.ylabel("return")
	plt.xlabel("iteration")
	plt.legend()
	plt.savefig("ac_cartpole")
	plt.close()

	## QUESTION 5

	for logdir_prefix in ["InvertedPendulum", "HalfCheetah"]:
		if os.path.exists("../data"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
		elif os.path.exists("../../run_logs"):
			data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../run_logs')
		else:
			raise "Data path doesn't exist"
		with_prefixes = [p for p in os.listdir(data_path) if logdir_prefix in p]
		logdirs = [os.path.join(data_path, p) for p in with_prefixes]

		iterations = []
		returns = []
		# best_returns = []
		# best_iterations = []
		labels = [p[:p.find("_" + logdir_prefix)] for p in with_prefixes]

		for logdir in logdirs:
			iterations.append([])
			returns.append([])
			# best_returns.append([])
			# best_iterations.append([])
			for summary in _summary_iterator(logdir):
				if len(summary.summary.value) == 1:
					if summary.summary.value[0].tag == "Train_AverageReturn":
						iterations[-1].append(summary.step)
						returns[-1].append(summary.summary.value[0].simple_value)
					# if summary.summary.value[0].tag == "Train_BestReturn":
					# 	best_returns[-1].append(summary.summary.value[0].simple_value)
					# 	best_iterations[-1].append(summary.step)

		for i, r, l in zip(iterations, returns, labels):
			plt.plot(i, r, label=l)
		# for i, r, l in zip(best_iterations, best_returns, labels):
		# 	plt.plot(i, r, label=l + " best")
		plt.title("Difficult Actor Critic " + logdir_prefix)
		plt.ylabel("return")
		plt.xlabel("iteration")
		plt.legend()
		plt.savefig("ac_" + logdir_prefix)
		plt.close()



