import numpy as np
import matplotlib.pyplot as plt

def maxweight_fic(G,U,X,q):
	N = len(G[0])
	mu = np.zeros([N,N])


    ### calculate actions
	for i in range(N):
		w = np.zeros(N)
		if i in U:
			for j in range(N):
				w[j] = G[i][j]*(X[i] - X[j] - q[i][j])
		else:
			for j in range(N):
				w[j] = G[i][j]*(X[i] - X[j])
		jj = np.argmax(w)
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
	

	rate = 1 * scale * load



	elements = range(N+1)
	Q = np.zeros(N)
	q = np.zeros([N,N])
	X = np.zeros(N)
	Y = np.zeros(N)
	Q_stat = np.zeros(T)
	q_stat = np.zeros(T)
	X_stat = np.zeros(T)
	Y_stat = np.zeros(T)
	instant_throughput = np.zeros(T)
	throughput_stat = np.zeros(T)
	
	total_Q = 0

	for t in range(T):
		print t
		### create new arrivals
		new_arrival = int(np.random.uniform(0,1) < rate)
		Q[src] += new_arrival
		X[src] += new_arrival
		mu = maxweight_fic(G,U,X,q)
		mu_u = uncontrollable_action(Q)
		true_mu = np.zeros([N,N])
		throughput = 0
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
						instant_throughput[t] += actual_mu
					true_mu[i][j] =actual_mu
			else:
				## uncontrollable nodes take actions
				old_Q = Q[i]
				throughput += (np.minimum(mu_u[i],old_Q))
				Q[i] = np.maximum(Q[i]-mu_u[i], 0)
				true_mu[i][dst] = mu_u[i]

			
			for j in range(N):
				old_X = X[i]
				X[i] = np.maximum(X[i]-mu[i][j], 0)
				actual_mu = old_X - X[i]
				if j != dst:
					X[j] += actual_mu
				mu[i][j] = actual_mu

		throughput_stat[t] = throughput
				

		# ## update debt queue
		for i in range(N):
			for j in range(N):
				q[i][j] = q[i][j] + mu[i][j] - true_mu[i][j]


		total_Q += np.sum(Q)


		Q_stat[t] = np.sum(Q)
		q_stat[t] = np.sum(np.abs(q))
		X_stat[t] = np.sum(X)
		Y_stat[t] = np.sum(np.abs(Y))



	print np.mean(throughput_stat)

	return throughput_stat

if __name__ == "__main__":
    main(load=0.99, T=5000)



