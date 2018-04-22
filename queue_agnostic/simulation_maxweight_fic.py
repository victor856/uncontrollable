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
	q = np.zeros([N,N])
	X = np.zeros(N)
	Y = np.zeros(N)
	G = np.zeros([N,N])
	Q_stat = np.zeros(T)
	q_stat = np.zeros(T)
	X_stat = np.zeros(T)
	Y_stat = np.zeros(T)
	instant_throughput = np.zeros(T)
	Q_oracle = np.zeros(N)
	Q_oracle_stat = np.zeros(T)
	oracle_mu = np.zeros([N,N])
	oracle_mu[0][4] = 16
	oracle_mu[4][3] = 25
	oracle_mu[0][1] = 10
	oracle_mu[1][2] = 5
	oracle_mu[1][2] = 5
	oracle_mu[2][3] = 5
	
	for e in edges:
		G[e[0]][e[1]] = e[2]
		# G[e[1]][e[0]] = e[2]

	estimated_mu = np.zeros(T)
	estimated_mu_true = np.zeros(T)
	# total_vec_true = np.zeros([T,N,N])
	total_mu = 0
	total_mu_true = 0
	total_Q = 0
	ct1 = 0
	ct4 = 0
	for t in range(T):
		print t
		### create new arrivals
		new_arrival = np.random.poisson(float(load)*float(rate))
		Q[src] += new_arrival
		X[src] += new_arrival
		Q_oracle[src] += new_arrival
		mu = maxweight_fic(G,U,X,q)
		if mu[0][1] >0:
			ct1 += 1
		if mu[0][4] > 0:
			ct4 += 1
		true_mu = np.zeros([N,N])
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
					## update oracle queues
					old_Q_oracle = Q_oracle[i]
					Q_oracle[i] = np.maximum(Q_oracle[i]-oracle_mu[i][j], 0)
					actual_mu_oracle = old_Q_oracle - Q_oracle[i]
					if j != dst:
						Q_oracle[j] += actual_mu_oracle					
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
					else:
						instant_throughput[t] += actual_mu
					true_mu[i][next_hop] = actual_mu

					## update oracle queues
					old_Q_oracle = Q_oracle[i]
					Q_oracle[i] = np.maximum(Q_oracle[i]-G[i][next_hop], 0)
					actual_mu_oracle = old_Q_oracle - Q_oracle[i]
					if next_hop != dst:
						Q_oracle[next_hop] += actual_mu_oracle
			
			for j in range(N):
				old_X = X[i]
				X[i] = np.maximum(X[i]-mu[i][j], 0)
				actual_mu = old_X - X[i]
				if j != dst:
					X[j] += actual_mu
				mu[i][j] = actual_mu
				

		# ## update debt queue
		for i in range(N):
			for j in range(N):
				q[i][j] = q[i][j] + mu[i][j] - true_mu[i][j]


		total_mu += mu[2][3]
		total_mu_true += true_mu[2][3]
		estimated_mu[t] = float(total_mu)/float(t+1)/float(G[2][3])
		estimated_mu_true[t] = float(total_mu_true)/float(t+1)/float(G[2][3])
		total_Q += np.sum(Q)


		Q_stat[t] = np.sum(Q)
		q_stat[t] = np.sum(np.abs(q))
		X_stat[t] = np.sum(X)
		Y_stat[t] = np.sum(np.abs(Y))
		Q_oracle_stat[t] = np.sum(Q_oracle)




	# print "avg throughput: " + str(float(np.sum(instant_throughput))/float(T))
	# print "optimal throughput: " + str(float(rate)*float(load))


	# print "avg queue length: " + str(float(total_Q)/float(T))
	upper_bound = np.zeros(T)
	for t in range(T):
		if t == 0:
			continue
		vt = np.max(Q_oracle_stat[0:t])
		# for adversarial
		upper_bound[t] = 10*np.sqrt(vt*t)


	
	# plt.xlabel('Time', fontsize=12)
	# plt.ylabel('Queue Length', fontsize=12)
	# plt.yscale('log')
	# line1, = plt.plot(range(T), Q_stat, label='Physical Queue Q')
	# line2, = plt.plot(range(T), X_stat, linestyle=":", label='Virtual Queue X')
	# line3, = plt.plot(range(T), q_stat, linestyle="-.", label='Virtual Queue Y')
	# line4, = plt.plot(range(T), X_stat + q_stat, linestyle="--", label='X+Y')
	# line5, = plt.plot(range(T), upper_bound, label='Upper Bound')
	# plt.legend(handles=[line1, line2, line3, line4, line5],prop={'size': 12})
	# ymin, ymax = plt.ylim()
	# plt.ylim(ymin=0, ymax=ymax)
	# plt.xlim(xmin=0, xmax=T)
	# plt.tick_params(axis='x', labelsize=12)
	# plt.tick_params(axis='y', labelsize=12)
	# plt.show()


	# estimated_mu = np.multiply(estimated_mu, 10)
	# true_policy = np.multiply(5, np.ones(T))
	# plt.xlabel('Time', fontsize=12)
	# plt.ylabel('Allocated Rate', fontsize=12)
	# line1, = plt.plot(range(T), estimated_mu, linestyle="--", label='Estimated Uncontrollable Policy')
	# line2, = plt.plot(range(T), true_policy, label='True Uncontrollable Policy')
	# plt.legend(handles=[line1, line2], prop={'size': 12})
	# ymin, ymax = plt.ylim()
	# plt.ylim(ymin=0, ymax=ymax)
	# plt.xlim(xmin=0, xmax=T)
	# plt.show()

	return Q_stat

if __name__ == "__main__":
    main(load=0.99, T=5000)



