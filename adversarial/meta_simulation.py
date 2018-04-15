import adaptive_adversary_stability
import adaptive_adversary_utility
import numpy as np
import matplotlib.pyplot as plt
import csv

# ### EXPERIMENT 1: stability-1
# T_set = [100, 400, 900, 4900, 8100, 1e4, 2.25e4, 4e4]
# Q_tmw = np.zeros(len(T_set))
# Q_mw = np.zeros(len(T_set))
# lower_bound = np.zeros(len(T_set))
# for i in range(len(T_set)):
# 	T = int(T_set[i])
# 	print T
# 	#VT = 200
# 	VT = np.floor(np.sqrt(T))
# 	#VT = np.floor(0.1*T)
# 	Q_tmw[i] = adaptive_adversary_stability.main(T=T, VT=VT, alg="TMW")
# 	Q_mw[i] = adaptive_adversary_stability.main(T=T, VT=VT, alg="MW")
# 	lower_bound[i] = VT

# plt.xlabel('Time Horizon T', fontsize=14)
# plt.ylabel('Queue Length', fontsize=14)
# line1, = plt.plot(T_set, Q_tmw, label='MaxWeight', linewidth=2, marker="o", markersize=8)
# line2, = plt.plot(T_set, Q_mw, label='Tracking-MaxWeight', linewidth=2, marker="*", markersize=10)
# line3, = plt.plot(T_set, lower_bound, label='Lower Bound',linestyle=":", linewidth=2)
# plt.legend(handles=[line1, line2, line3], prop={'size': 14})
# ymin, ymax = plt.ylim()
# plt.ylim(ymin=0, ymax=ymax)
# plt.xlim(xmin=0, xmax=np.max(T_set))
# plt.show()

### EXPERIMENT 2: stability-2
VT_set = [20, 30, 40, 50, 100, 200, 500]
T = int(1e4)
Q_tmw = np.zeros(len(VT_set))
Q_mw = np.zeros(len(VT_set))
lower_bound = np.zeros(len(VT_set))
upper_bound = np.zeros(len(VT_set))
for i in range(len(VT_set)):
	VT = VT_set[i]
	print VT
	Q_tmw[i] = adaptive_adversary_stability.main(T=T, VT=VT, alg="TMW")
	Q_mw[i] = adaptive_adversary_stability.main(T=T, VT=VT, alg="MW")
	lower_bound[i] = VT
	upper_bound[i] = 5*np.sqrt(T*VT)

plt.xlabel('$V_T$', fontsize=14)
plt.ylabel('Queue Length', fontsize=14)
#line1, = plt.plot(VT_set, Q_mw, label='MaxWeight', linewidth=2, marker="o", markersize=8)
line2, = plt.plot(VT_set, Q_tmw, label='Tracking-MaxWeight', linewidth=2, marker="*", markersize=10)
line3, = plt.plot(VT_set, lower_bound, label='Lower Bound',linewidth=2, linestyle=":")
line4, = plt.plot(VT_set, upper_bound, label='Upper bound',linewidth=2, linestyle="--")
plt.legend(handles=[line2,  line3, line4], prop={'size': 14})
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax=ymax)
plt.xlim(xmin=0, xmax=np.max(VT_set))
plt.show()

# ### EXPERIMENT 3: utility-1
# # T_set = [100, 900, 4000, 1e4, 4e4, 9e4, 1.6e5, 3.6e5, 4.9e5]
# T_set = [100, 1000, 5000, 1e4, 2e4, 5e4, 1e5]
# Q_tmw1 = np.zeros(len(T_set))
# utility_tmw1 = np.zeros(len(T_set))
# Q_tmw2 = np.zeros(len(T_set))
# utility_tmw2 = np.zeros(len(T_set))
# Q_tmw3 = np.zeros(len(T_set))
# utility_tmw3 = np.zeros(len(T_set))

# for i in range(len(T_set)):
# 	T = int(T_set[i])
# 	print T
# 	#VT = np.floor(2*np.sqrt(T))
# 	VT = np.floor(0.05*T) 
# 	V1 = 200
# 	V2 = 2 * (T**0.75)
# 	V3 = 0.25 * (T**2)
# 	Q_tmw1[i], utility_tmw1[i] = adaptive_adversary_utility.main(T=T, VT=VT, V=V1, alg="TMW")
# 	Q_tmw2[i], utility_tmw2[i] = adaptive_adversary_utility.main(T=T, VT=VT, V=V2, alg="TMW")
# 	Q_tmw3[i], utility_tmw3[i] = adaptive_adversary_utility.main(T=T, VT=VT, V=V3, alg="TMW")

# ## dump results to disk
# with open('utility_linear.csv','w+') as f:
# 	f.write("T,V=200,V=T^{3/4},V=T^2\n")

# with open('utility_linear.csv','a') as f:
# 	row = []
# 	for i in range(len(T_set)):
# 		T = int(T_set[i])
# 		f.write("%s,%s,%s,%s" % (T, utility_tmw1[i], utility_tmw2[i], utility_tmw3[i]))
# 		if i < len(T_set) - 1:
# 			f.write("\n")

# with open('Q_linear.csv','w+') as f:
# 	f.write("T,V=200,V=T^{3/4},V=T^2\n")

# with open('Q_linear.csv','a') as f:
# 	row = []
# 	for i in range(len(T_set)):
# 		T = int(T_set[i])
# 		f.write("%s,%s,%s,%s" % (T, Q_tmw1[i], Q_tmw2[i], Q_tmw3[i]))
# 		if i < len(T_set) - 1:
# 			f.write("\n")

# # # T_set = [100, 900, 4000, 1e4, 4e4, 9e4, 1.6e5, 3.6e5, 4.9e5]
# # T_set = [100, 1000, 5000, 1e4, 2e4, 5e4, 1e5]
# # Q_tmw1 = []
# # utility_tmw1 = []
# # Q_tmw2 = []
# # utility_tmw2 = []
# # Q_tmw3 = []
# # utility_tmw3 = []
# # ct = 0
# # with open('Q_linear.csv', 'rb') as f:
# #     reader = csv.reader(f)
# #     for row in reader:
# #     	if ct == 0:
# #     		ct += 1
# #     		continue
# #         Q_tmw1.append(row[1])
# #         Q_tmw2.append(row[2])
# #         Q_tmw3.append(row[3])


# # ct = 0
# # with open('utility_linear.csv', 'rb') as f:
# #     reader = csv.reader(f)
# #     for row in reader:
# #     	if ct == 0:
# #     		ct += 1
# #     		continue
# #         utility_tmw1.append(row[1])
# #         utility_tmw2.append(row[2])
# #         utility_tmw3.append(row[3])

# plt.xlabel('Time Horizon T', fontsize=14)
# plt.ylabel('Queue Length', fontsize=14)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# line1, = plt.plot(T_set, Q_tmw1, label='V=$\Theta(1)$', linewidth=2, marker="o", markersize=8)
# line2, = plt.plot(T_set, Q_tmw2, label='V=$\Theta(T^{3\slash 4})$', linewidth=2, marker="s", markersize=8)
# line3, = plt.plot(T_set, Q_tmw3, label='V=$\Theta(T^2)$', linewidth=2, marker="*", markersize=10)
# plt.legend(handles=[line1, line2, line3], prop={'size': 14})
# ymin, ymax = plt.ylim()
# plt.ylim(ymin=0, ymax=ymax)
# plt.xlim(xmin=0, xmax=np.max(T_set))
# plt.show()

# plt.xlabel('Time Horizon T', fontsize=14)
# plt.ylabel('Utility Regret', fontsize=14)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# line1, = plt.plot(T_set, utility_tmw1, label='V=$\Theta(1)$', linewidth=2, marker="o", markersize=8)
# line2, = plt.plot(T_set, utility_tmw2, label='V=$\Theta(T^{3\slash 4})$', linewidth=2, marker="s", markersize=8)
# line3, = plt.plot(T_set, utility_tmw3, label='V=$\Theta(T^2)$', linewidth=2, marker="*", markersize=10)
# plt.legend(handles=[line1, line2, line3], prop={'size': 14})
# ymin, ymax = plt.ylim()
# plt.ylim(ymin=0, ymax=ymax)
# plt.xlim(xmin=0, xmax=np.max(T_set))
# plt.show()


# ### EXPERIMENT 4: utility-2
# T = int(1e4)
# VT = np.floor(2*np.sqrt(T))
# V_set = [200,500,800,1000,2000,3000, 4000, 5000]
# Q_tmw = np.zeros(len(V_set))
# utility_tmw = np.zeros(len(V_set))
# lower_bound_Q = np.zeros(len(Q_tmw))
# lower_bound_utility = np.zeros(len(utility_tmw))


# for i in range(len(V_set)):
# 	V = int(V_set[i])
# 	print V
# 	Q_tmw[i], utility_tmw[i] = adaptive_adversary_utility.main(T=T, VT=VT, V=V, alg="TMW")
# 	lower_bound_Q[i] = Q_tmw[i]
# 	lower_bound_utility[i] = VT - lower_bound_Q[i]

# V_set2 = [10,50,100,200,500,800,1000]
# upper_bound_Q = np.zeros(len(V_set2))
# upper_bound_utility = np.zeros(len(V_set2))
# for i in range(len(V_set2)):
# 	V = int(V_set2[i])
# 	upper_bound_utility[i] = VT*T/V
# 	upper_bound_Q[i] = 0.2*np.sqrt(T*(V+VT))


# plt.xlabel('Queue Length', fontsize=14)
# plt.ylabel('Utility Regret', fontsize=14)
# plt.yscale('log')
# plt.xscale('log')
# line1, = plt.plot(Q_tmw, utility_tmw, label='TMW')
# line2, = plt.plot(lower_bound_Q, lower_bound_utility, label='Lower Bound', linewidth=2, linestyle=":")
# line3, = plt.plot(upper_bound_Q, upper_bound_utility, label='Upper Bound', linewidth=2, linestyle="--")
# plt.legend(handles=[line1, line2, line3], prop={'size': 14})
# ymin, ymax = plt.ylim()
# plt.ylim(ymin=0, ymax=ymax)
# plt.xlim(xmin=0, xmax=np.max(upper_bound_Q))
# plt.show()

