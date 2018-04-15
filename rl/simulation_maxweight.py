import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random

timestamp = int(time.mktime(datetime.now().timetuple()))
np.random.seed(timestamp)

def maxweight(G,Q,U):
	N = len(G[0])
	mu = np.zeros([N,N])
 
	for i in range(N):
		if i in U:
			continue
		index_set = []
		w = np.zeros(N)
		for j in range(N):
			w[j] = G[i][j]*(Q[i] - Q[j])
		jj = np.argmax(w)
		for j in range(N):
			if w[j] == w[jj]:
				index_set.append(j)
		random.seed(datetime.now())
		random.shuffle(index_set)
		jj = index_set[0]
		if w[jj] > 0:
			mu[i][jj] = G[i][jj]

	return mu


def uncontrollable_action(Q):
	threshold = 5
	scale = 1
	action = {}
	if Q[1] <= threshold and Q[2] <= threshold:
		action[1] = 0.5 * scale
		action[2] = 0 * scale
	if Q[1] > threshold and Q[2] <= threshold:
		action[1] = 0.5 * scale
		action[2] = 0 * scale
	if Q[1] <= threshold and Q[2] > threshold:
		action[1] = 0 * scale
		action[2] = 1 * scale
	if Q[1] > threshold and Q[2] > threshold:
		action[1] = 0.25 * scale
		action[2] = 0.25 * scale
	a = {}
	a[1] = int(np.random.uniform(0,1) < action[1])
	a[2] = int(np.random.uniform(0,1) < action[2])
	return a

def main(T,load):
	scale = 1
	N = 4
	B = 1
	action_space = [1,2]
	edges = [(0,1,1), (0,2,1), (1,3,1), (2,3,1)]
	U = [1,2]
	G = np.zeros([N,N])
	for e in edges:
		G[e[0]][e[1]] = e[2] * scale
	Q = np.zeros(N)
	src = 0
	dst = 3
	truncated_Q = {}
	truncated_Q[1] = 0
	truncated_Q[2] = 0
	accu_visit_state_action = {}
	accu_visit_transition = {}
	

	rate = 1 * scale * load


	elements = range(N+1)
	Q = np.zeros(N)
	G = np.zeros([N,N])
	Q_stat = np.zeros(T)

	throughput_stat = np.zeros(T)
	for e in edges:
		G[e[0]][e[1]] = e[2]
		G[e[1]][e[0]] = e[2]

	for t in range(T):
		print t
		### create new arrivals
		new_arrival = int(np.random.uniform(0,1) < rate)
		Q[src] += new_arrival
		mu = maxweight(G,Q,U)
		true_mu = np.zeros([N,N])
		mu_u = uncontrollable_action(Q)
		### apply controllable action and update queues
		instant_throughput = 0
		for i in range(N):
			if i not in U:
				for j in range(N):
					old_Q = Q[i]
					Q[i] = np.maximum(Q[i]-mu[i][j], 0)
					actual_mu = old_Q - Q[i]
					true_mu[i][j] = actual_mu
					if j != dst:
						Q[j] += actual_mu
			else:
				## uncontrollable nodes take actions
				old_Q = Q[i]
				instant_throughput += (np.minimum(mu_u[i],old_Q))
				Q[i] = np.maximum(Q[i]-mu_u[i], 0)
				

		throughput_stat[t] = instant_throughput



		Q_stat[t] = np.sum(Q)

	print np.mean(throughput_stat)
	# plt.plot(range(T), throughput_stat)
	# plt.show()

	# plt.plot(range(T), Q_stat)
	# plt.show()

	return throughput_stat


if __name__ == "__main__":
    main(load=0.95, T=5000)



