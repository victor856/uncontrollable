import numpy as np
import matplotlib.pyplot as plt

def maxweight_fic(G,Q,Y,Z):
	N = len(G[0])
	mu = np.zeros([N,N])

    ### calculate actions
	for i in range(N):
		w = np.zeros(N)
		for j in range(N):
			w[j] = G[i][j]*(Q[i] + Y[i] + Z[i]- Y[j] - Q[j] - Z[j])
		jj = np.argmax(w)
		if w[jj] > 0:
			mu[i][jj] = G[i][jj]
		# if i == 5:
		# 	print mu[i][jj]

	return mu

def main():
	## topology
	N = 6
	U = [4,5]
	edges = [(0,1,5),(1,2,5),(2,3,5),(3,4,10),(4,5,10),(0,5,10)]
	uncontrollable_policy = {}
	uncontrollable_policy[5] = np.zeros(N+1)
	uncontrollable_policy[5][4] = 1
	uncontrollable_policy[4] = np.zeros(N+1)
	uncontrollable_policy[4][N] = 1
	elements = range(N+1)
	src = 0
	dst = 3
	load = 0.9
	rate = 5
	T = 1000
	Q = np.zeros(N)
	Y = np.zeros(N)
	Z = np.zeros(N)
	G = np.zeros([N,N])
	Q_stat = np.zeros(T)
	Y_stat = np.zeros(T)
	Z_stat = np.zeros(T)
	for e in edges:
		G[e[0]][e[1]] = e[2]
		G[e[1]][e[0]] = e[2]
	for t in range(T):
		print t
		### create new arrivals
		new_arrival = np.random.poisson(load*rate)
		Q[src] += new_arrival
		mu = maxweight_fic(G,Q,Y,Z)
		true_mu = np.zeros([N,N])
		### apply controllable action and update queues
		for i in range(N):
			if i not in U:
				for j in range(N):
					true_mu[i][j] = mu[i][j]
					old_Q = Q[i]
					Q[i] = np.maximum(Q[i]-mu[i][j], 0)
					actual_mu = old_Q - Q[i]
					
					if j != dst:
						Q[j] += actual_mu
			else:
				## uncontrollable nodes take actions
				next_hop = np.random.choice(elements, 1, p=uncontrollable_policy[i])
				if next_hop != N:
					true_mu[i][next_hop] = G[i][next_hop]
					old_Q = Q[i]
					Q[i] = np.maximum(Q[i]-G[i][next_hop], 0)
					actual_mu = old_Q - Q[i]
					if next_hop != dst:
						Q[next_hop] += actual_mu

		## update debt queue
		for i in range(N):
			Y[i] = np.maximum(Y[i]+np.sum(true_mu,1)[i]-np.sum(mu,1)[i],0)
			Z[i] = np.maximum(Z[i]+np.sum(true_mu,0)[i]-np.sum(mu,0)[i],0)

		Q_stat[t] = np.sum(Q)
		Y_stat[t] = np.sum(Y)
		Z_stat[t] = np.sum(Z)

	plt.plot(range(T), Q_stat)
	plt.show()

	# plt.plot(range(T), Y_stat)
	# plt.show()

	# plt.plot(range(T), Z_stat)
	# plt.show()

if __name__ == "__main__":
    main()



