import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
import random
import itertools
from datetime import datetime



def dict_sum(p):
	s = 0
	for k in p: 
		s += p[k]
	return s

def dict_multiply(a,b):
	s = 0
	for k in a:
		s += (a[k]*b[k])
	return s


def value_iteration(state_space,action_space, p_sample, tk, state_to_index):
	# random.seed(datetime.now())
	# random.shuffle(action_space)
	n_state = len(state_space)
	value_function = np.zeros(n_state, dtype=np.float32)
	prev_value_function = np.zeros(n_state, dtype=np.float32)
	policy = {}
	ct = 0
	while True:
		# print ct
		ct += 1
		for s in state_space:
			max_val = -sys.maxint
			reward = -(s[0] + s[1])**2
			for a in action_space:
				current_value = reward + np.dot(p_sample[(s,a)],prev_value_function)
				if current_value > max_val:
					max_val = current_value
					policy[s] = a
			value_function[state_to_index[s]] = max_val

		max_diff = -sys.maxint
		min_diff = sys.maxint
		for s in state_space:
			diff = value_function[state_to_index[s]] - prev_value_function[state_to_index[s]]
			if diff > max_diff:
				max_diff = diff
			if diff < min_diff:
				min_diff = diff

		if max_diff - min_diff <= 1/np.sqrt(tk):
			break

		for i in range(n_state):
			prev_value_function[i] = value_function[i]

	return policy





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


def main(T,load,V):
	scale = 1
	N = 4
	B = 1
	action_space = [1,2]
	edges = [(0,1,4), (0,2,4), (1,3,2), (2,3,2)]
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
	
	#T = 100000
	#c = 2*(2*B+1)**2
	c = 0.5
	#V = 10
	k = 0
	t = 0
	#load = 0.95
	rate = 1 * scale * load

	state_space = []
	index = 0
	state_to_index = {}
	for i in range(V):
		for j in range(V):
			state_to_index[(i,j)] = index
			state_space.append((i,j))
			index += 1

	estimated_p = {}
	for s in state_space:
		for a in action_space:
			estimated_p[(s,a)] = {}
			for ss in state_space:
				estimated_p[(s,a)][ss] = 0

	episode_length = 0

	loss = np.zeros(T)
	loss_rate_checkpoint = []
	t_checkpoint = []
	Q_checkpoint = []
	Q_vector_checkpoint = []
	throughput_checkpoint = []
	total_pkt = 0

	### prior distribution
	alpha0 = 1
	alpha = {}
	for s in state_space:
		q1_upper = np.minimum(s[0] + B, V-1)
		q2_upper = np.minimum(s[1] + B, V-1)
		q1_lower = np.maximum(s[0] - B, 0)
		q2_lower = np.maximum(s[1] - B, 0)
		for a in action_space:
			alpha[(s,a)] = np.zeros(len(state_space), dtype=np.float32)
			for i in range(q1_lower, q1_upper):
				for j in range(q2_lower, q2_upper):
					index = state_to_index[(i,j)]
					alpha[(s,a)][index] = alpha0
				

	while t < T:
		print t
		print Q
		if total_pkt > 0:
			print float(np.sum(loss))/float(total_pkt)
		### initialize an episode
		tk = t
		episode_visit_state_action = {}
		episode_visit_transition = {}

		### sample MDP from posterior distribution
		p_sample = {}
		for s in state_space:
			for a in action_space:
				p_sample[(s,a)] = np.random.dirichlet(alpha[(s,a)])
		### calculate the optimal optimistic policy
		policy = value_iteration(state_space,action_space, p_sample, tk, state_to_index)
		# print dict_sum(policy)
		### execute policy
		while True:

			## observe current queue length
			for i in U:
				truncated_Q[i] = int(np.minimum(V-1, Q[i]))
			current_state = (truncated_Q[1], truncated_Q[2])
			mu_u = uncontrollable_action(Q)			
			## arrival
			arrival =  int(np.random.uniform(0,1) < rate)
			total_pkt += arrival
			## take an anction
			action = policy[(truncated_Q[1], truncated_Q[2])]
			## update queue
			actual_mu = arrival
			Q[action] += actual_mu
			if Q[action] > V-1:
				loss[t] = Q[action] - (V-1)
				Q[action] = V-1
			# Q[action] = np.minimum(V-1, Q[action] + actual_mu)
			instant_througput = 0
			for i in U:
				Q[i] = np.maximum(Q[i] - mu_u[i], 0)
				instant_througput += mu_u[i]
				truncated_Q[i] = int(np.minimum(V-1, Q[i]))

			if t > 0 and t % 1 == 0:
				t_checkpoint.append(t)
				loss_rate_checkpoint.append(float(np.sum(loss))/float(total_pkt))
				Q_checkpoint.append(np.sum(Q))
				Q_vector_checkpoint.append((Q[1],Q[2]))
				throughput_checkpoint.append(instant_througput)

			## update visit counts
			current_state_action = (current_state, action)
			next_state = (truncated_Q[1], truncated_Q[2])
			transition = (current_state, action, next_state)

			if transition not in episode_visit_transition:
				episode_visit_transition[transition] = 1
			else:
				episode_visit_transition[transition] += 1


			if current_state_action not in episode_visit_state_action:
				episode_visit_state_action[current_state_action] = 1
			else:
				episode_visit_state_action[current_state_action] += 1

			t = t + 1
			## update posterior probability
			index = state_to_index[next_state]
			alpha[(current_state, action)][index] += 1
			## stopping condition
			if t >= T:
				break
			if (current_state, action) not in accu_visit_state_action:
				episode_length = t - tk
				break
			if episode_visit_state_action[(current_state, action)] == accu_visit_state_action[(current_state, action)]:
				episode_length = t - tk
				break
			# if t - tk >= episode_length + 1:
			# 	episode_length = t - tk
			# 	break

		## update accumulative counts
		for key in episode_visit_transition:
			if key not in accu_visit_transition:
				accu_visit_transition[key] = episode_visit_transition[key]
			else:
				accu_visit_transition[key] += episode_visit_transition[key]
		for key in episode_visit_state_action:
			if key not in accu_visit_state_action:
				accu_visit_state_action[key] = episode_visit_state_action[key]
			else:
				accu_visit_state_action[key] += episode_visit_state_action[key]

	# t_checkpoint.append(t)
	# loss_rate_checkpoint.append(float(np.sum(loss))/float(total_pkt))
	# Q_checkpoint.append(np.sum(Q))
	# Q_vector_checkpoint.append((Q[1],Q[2]))


	# plt.plot(t_checkpoint, Q_checkpoint)
	# plt.show()

	# plt.plot(t_checkpoint, loss_rate_checkpoint)
	# plt.show()

	# plt.plot(t_checkpoint, throughput_checkpint)
	# plt.show()

	with open("queue_vector.csv",'w+') as f:
		print("writing queue length vector to file...")
	with open("queue_vector.csv",'a') as f:
		for i in range(len(Q_vector_checkpoint)):
			vec = Q_vector_checkpoint[i]
			f.write("%s,%s" % (vec[0], vec[1]))
			if i < len(Q_vector_checkpoint) - 1:
				f.write("\n")

	return (t_checkpoint, Q_checkpoint, loss_rate_checkpoint, throughput_checkpoint)


if __name__ == "__main__":
    main(T=200000,load=0.95,V=10)