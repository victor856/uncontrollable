import numpy as np
import sys
import operator
import matplotlib.pyplot as plt
import random

def d(s, a, action_space,V,T,c,visit):
	state_action = (s, a) 
	if state_action not in visit:
		v = 1
	else:
		v = visit[state_action]
	A = len(action_space)
	bound = np.sqrt(c*np.log(2*A*T*V)/v/np.log(2))
	return bound

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


def get_optimistic_transition(p_hat,d,u,B,s,a,state_space,V):
	# q1_upper = np.minimum(s[0] + B, V-1)
	# q2_upper = np.minimum(s[1] + B, V-1)
	# q1_lower = np.maximum(s[0] - B, 0)
	# q2_lower = np.maximum(s[1] - B, 0)
	# local_state_sapce = []
	# u_truncated = {}
	# for i in range(q1_lower, q1_upper + 1):
	# 	for j in range(q2_lower, q2_upper + 1):
	# 		local_state_sapce.append((i,j))
	# 		u_truncated[(i,j)] = u[(i,j)]
	# sorted_u = sorted(u_truncated.items(), key=operator.itemgetter(1),reverse=True)

	sorted_u = sorted(u.items(), key=operator.itemgetter(1),reverse=True)
	sorted_s = []
	p = {}
	# print q1_lower,q1_upper
	# print q2_lower,q2_upper
	# print u_truncated
	for sv in sorted_u:
		sorted_s.append(sv[0])
	s0 = sorted_s[0]
	p[s0] = np.minimum(1,p_hat[s0] + float(d)/float(2))
	for j in range(len(sorted_s)):
		if j == 0:
			continue
		sj = sorted_s[j]
		p[sj] = p_hat[sj]
	l = len(sorted_s) - 1
	while dict_sum(p) > 1 and l >= 0:
		sl = sorted_s[l]
		reduction = np.minimum(dict_sum(p) - 1, p[sl])
		p[sl] -= reduction
		l = l - 1
	return p



def extended_value_iteration(G,V,T,c,state_space,action_space,estimated_p,visit,tk,B):
	# for s in state_space:
	# 	for a in action_space:
	# 		ss = dict_sum(estimated_p[(s,a)])
	# 		if ss > 0:
	# 			print (s,a)
	# 			print ss
	value_function = {}
	policy = {}
	prev_value_function = {}
	for s in state_space:
		value_function[s] = 0
		prev_value_function[s] = 0
	ct = 0
	while True:
		#print ct
		ct += 1
		for s in state_space:
			max_val = -sys.maxint
			reward = -(s[0] + s[1])*10
			for a in action_space:
				p_hat = estimated_p[(s,a)]
				bound = d(s, a, action_space,V,T,c,visit)
				p = get_optimistic_transition(p_hat, bound, prev_value_function,B,s,a,state_space,V)
				current_value = reward + dict_multiply(p,prev_value_function)
				if current_value > max_val:
					max_val = current_value
					policy[s] = a
			value_function[s] = max_val

		max_diff = -sys.maxint
		min_diff = sys.maxint
		for s in state_space:
			diff = value_function[s] - prev_value_function[s]
			if diff > max_diff:
				max_diff = diff
			if diff < min_diff:
				min_diff = diff

		if max_diff - min_diff <= 1/np.sqrt(tk+1):
			break

		for s in state_space:
			prev_value_function[s] = value_function[s]

	return policy





def uncontrollable_action(Q):
	threshold = 2
	scale = 0.5
	action = {}
	if Q[1] <= threshold and Q[2] <= threshold:
		action[1] = 0.5 * scale
		action[2] = 0.5 * scale
	if Q[1] > threshold and Q[2] <= threshold:
		action[1] = 1 * scale
		action[2] = 0 * scale
	if Q[1] <= threshold and Q[2] > threshold:
		action[1] = 0 * scale
		action[2] = 2 * scale
	if Q[1] > threshold and Q[2] > threshold:
		action[1] = 0.5 * scale
		action[2] = 0.5 * scale
	a = {}
	a[1] = int(np.random.uniform(0,1) < action[1])
	a[2] = int(np.random.uniform(0,1) < action[2])
	return a


def main():
	scale = 0.5
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
	
	T = 5000000
	#c = 2*(2*B+1)**2
	c = 0.5
	V = 5
	#V = int(2*np.log(T))
	print V
	k = 0
	t = 0
	load = 0.9
	rate = 2 * scale * load

	state_space = []
	for i in range(V):
		for j in range(V):
			state_space.append((i,j))

	estimated_p = {}
	for s in state_space:
		for a in action_space:
			estimated_p[(s,a)] = {}
			for ss in state_space:
				estimated_p[(s,a)][ss] = 0

	episode_length = 0

	while t < T:
		print t
		print Q
		### initialize an episode
		tk = t
		episode_visit_state_action = {}
		episode_visit_transition = {}
		
		for key in accu_visit_transition:
			current_state = key[0]
			action = key[1]
			next_state = key[2]
			estimated_p[(current_state, action)][next_state] = float(accu_visit_transition[key])/float(accu_visit_state_action[(current_state,action)])
		### calculate the optimal optimistic policy
		policy = extended_value_iteration(G,V,T,c,state_space,action_space,estimated_p,accu_visit_state_action,tk,B)
		print dict_sum(policy)
		### execute policy
		while True:
			## observe current queue length
			for i in U:
				truncated_Q[i] = int(np.minimum(V-1, Q[i]))
			current_state = (truncated_Q[1], truncated_Q[2])
			mu_u = uncontrollable_action(Q)			
			## arrival
			arrival =  int(np.random.uniform(0,1) < rate)
			## take an anction
			action = policy[(truncated_Q[1], truncated_Q[2])]
			## update queue
			actual_mu = arrival
			#Q[action] += actual_mu
			Q[action] = np.minimum(V-1, Q[action] + actual_mu)
			for i in U:
				Q[i] = np.maximum(Q[i] - mu_u[i], 0)
				truncated_Q[i] = int(np.minimum(V-1, Q[i]))

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
			## stopping condition
			if t >= T:
				break
			if (current_state, action) not in accu_visit_state_action:
				episode_length = t - tk
				break
			if episode_visit_state_action[(current_state, action)] == accu_visit_state_action[(current_state, action)]:
				episode_length = t - tk
				break
			if t - tk >= episode_length + 1:
				episode_length = t - tk
				break

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



if __name__ == "__main__":
    main()