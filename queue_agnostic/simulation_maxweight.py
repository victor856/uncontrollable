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

def main(load, T):
	### topology 1
	# N = 6
	# U = [4,5]
	# edges = [(0,1,5),(1,2,5),(2,3,10),(4,3,10),(5,4,10),(0,5,10),(5,2,10)]
	# uncontrollable_policy = {}
	# uncontrollable_policy[5] = np.zeros(N+1)
	# uncontrollable_policy[5][4] = 0
	# uncontrollable_policy[5][2] = 1
	# uncontrollable_policy[4] = np.zeros(N+1)
	# uncontrollable_policy[4][3] = 0
	# uncontrollable_policy[4][N] = 1
	# elements = range(N+1)
	# src = 0
	# dst = 3
	# rate = 10

	# ### topology 2
	# N = 6
	# U = [3,4,5]
	# edges = [(0,1,50), (0,4,100), (0,5,100), (1,2,200), (5,1,100), (5,2,100), (4,3,100), (3,2,100)]
	# uncontrollable_policy = {}
	# for i in U:
	# 	uncontrollable_policy[i] = np.zeros(N+1)
	# uncontrollable_policy[3][N] = 1
	# uncontrollable_policy[4][3] = 1
	# uncontrollable_policy[5][1] = 0.3
	# uncontrollable_policy[5][2] = 0.7
	# src = 0
	# dst = 2
	# rate = 100

 	### topology 3
	N = 5
	U = [1,2]
	edges = [(0,1,40), (1,2,40), (0,4,20), (2,3,10), (4,3,20), (1,4,40)]
	uncontrollable_policy = {}
	for i in U:
		uncontrollable_policy[i] = np.zeros(N+1)
	uncontrollable_policy[1][2] = 0.5
	uncontrollable_policy[1][4] = 0.5
	uncontrollable_policy[2][3] = 0.5
	uncontrollable_policy[2][N] = 0.5
	src = 0
	dst = 3
	rate = 25


	elements = range(N+1)
	Q = np.zeros(N)
	X = np.zeros(N)
	Y = np.zeros(N)
	G = np.zeros([N,N])
	Q_stat = np.zeros(T)
	q_stat = np.zeros(T)
	X_stat = np.zeros(T)
	Y_stat = np.zeros(T)
	for e in edges:
		G[e[0]][e[1]] = e[2]

	for t in range(T):
		print t
		### create new arrivals
		new_arrival = np.random.poisson(float(load)*float(rate))
		print new_arrival
		Q[src] += new_arrival
		mu = maxweight(G,Q,U)
		true_mu = np.zeros([N,N])
		### apply controllable action and update queues
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
				next_hop = np.random.choice(elements, 1, p=uncontrollable_policy[i])
				if next_hop != N:
					old_Q = Q[i]
					Q[i] = np.maximum(Q[i]-G[i][next_hop], 0)
					actual_mu = old_Q - Q[i]
					true_mu[i][next_hop] = actual_mu
					if next_hop != dst:
						Q[next_hop] += actual_mu

		Q_stat[t] = np.sum(Q)

	# plt.plot(range(T), Q_stat)
	# plt.show()

	return Q_stat


if __name__ == "__main__":
    main(load=0.35, T=5000)



