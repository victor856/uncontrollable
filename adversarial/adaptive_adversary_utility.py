import numpy as np
import matplotlib.pyplot as plt
import math

def Tracking_MaxWeight(G, X, Y, U):
	N = len(X)
	K = len(X[0])
	mu = np.zeros([N,N,K])

	for i in range(N):
		for j in range(N):
			w = np.zeros(K)
			for k in range(K):
				if i in U:
					w[k] = X[i][k] - X[j][k]
				else:
					w[k] = X[i][k] - X[j][k] + Y[i][k] - Y[j][k]
			kk = np.argmax(w)
			if w[kk] > 0:
				mu[i][j][kk] = G[i][j]
	return mu

def MaxWeight(Q, G):
	N = len(Q)
	K = len(Q[0])
	mu = np.zeros([N,N,K])

	for i in range(N):
		for j in range(N):
			w = np.zeros(K)
			for k in range(K):
				w[k] = Q[i][k] - Q[j][k]
			kk = np.argmax(w)
			if w[kk] > 0:
				mu[i][j][kk] = G[i][j]

	return mu

def main(T, VT, V, alg):
	scale = 5
	N = 4
	edges = [(0,1,2), (1,2,2), (1,3,2)]
	G = np.zeros([N,N])
	for e in edges:
		G[e[0]][e[1]] = e[2] * scale
	K = 2
	src_dst = [(0,2), (0,3)]
	U = [1]
	D = 2*scale
	Z = int(VT)

	Q = np.zeros([N,K])
	X = np.zeros([N,K])
	Y = np.zeros([N,K])
	Q_stat = np.zeros(T)
	Q_checkpoint = np.zeros(K)
	utility_regret = 0

	for t in range(T):
		# print t
		if t % Z == 0:
			current_frame_start = t

		## generate arrivals
		A = np.zeros([N,K])
		for k in range(K):
			s = src_dst[k][0]
			d = src_dst[k][1]
			if t - current_frame_start <= np.floor(Z/2) - 1:
				arrival = 2 * scale
			else:
				arrival = 0
			A[s][k] = arrival


		if t - current_frame_start == np.floor(Z/2) - 1:
			Q_checkpoint = np.sum(Q, 0)

		## perform admission control
		for k in range(K):
			s = src_dst[k][0]
			d = src_dst[k][1]
			if X[s][k] == 0:
				admitted_arrival = A[s][k]
			else:
				admitted_arrival = np.minimum(A[s][k], V/X[s][k] -1)
			admitted_arrival = np.maximum(admitted_arrival, 0)
			Q[s][k] += admitted_arrival
			X[s][k] += admitted_arrival
			utility_regret += (math.log(1+A[s][k], 2)- math.log(1+admitted_arrival, 2) )


		## compute routing decisions
		if alg == "MW":
			mu = MaxWeight(Q, G)
		if alg == "TMW":
			mu = Tracking_MaxWeight(G, X, Y, U)


		## execute routing decisions
		actual_mu = np.zeros([N,N,K])
		if t - current_frame_start <= np.floor(Z/2) - 1:
			actual_mu[1][2][0] = np.minimum(2 * scale, Q[1][0])
			actual_mu[1][3][1] =  np.minimum(2 * scale, Q[1][1])
		else:
			if Q_checkpoint[0] <= Q_checkpoint[1]:
				actual_mu[1][2][0] = np.minimum(2 * scale, Q[1][0])
			else:
				actual_mu[1][3][1] = np.minimum(2 * scale, Q[1][1])

		for i in range(N):
			for k in range(K):
				for j in range(N):

					if i not in U:
						actual_mu[i][j][k] = np.minimum(Q[i][k], mu[i][j][k])

					## update physical queues
					Q[i][k] -= actual_mu[i][j][k]
					if j != src_dst[k][1]:
						Q[j][k] += actual_mu[i][j][k]

					## update virtual queues
					if alg == "TMW":
						actual_mu_X = np.minimum(X[i][k], mu[i][j][k])
						X[i][k] = X[i][k] - actual_mu_X
						if j != src_dst[k][1]:
							X[j][k] += actual_mu_X
					

				if alg == "TMW":
					arrival_diff = 0
					departure_diff = 0
					for j in range(N):
						arrival_diff += (actual_mu[j][i][k] - mu[j][i][k])
						departure_diff += (mu[i][j][k] - actual_mu[i][j][k])
					Y[i][k] = np.maximum(Y[i][k]+arrival_diff-departure_diff, 0)

		Q_stat[t] = np.sum(Q)


	return Q_stat[T-1], utility_regret


if __name__ == "__main__":
    main(T=10000, VT=100, V=100, alg="TMW" )