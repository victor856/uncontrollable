import simulation_maxweight_fic
import simulation_maxweight
import numpy as np
import matplotlib.pyplot as plt
import csv

load_set_fic = np.linspace(0.05,0.99, num=30)
load_set_fic = load_set_fic.tolist()
load_set_fic.append(1)
load_set_fic.append(1.01)
load_set_max = np.linspace(0.05,0.45, num=30)
load_set_max = load_set_max.tolist()
T=5000
Q_stat_fic = np.zeros(len(load_set_fic))
Q_stat_max = np.zeros(len(load_set_max))
for i in range(len(load_set_fic)):
	load = load_set_fic[i]
	Q_fic = simulation_maxweight_fic.main(load,T)
	Q_stat_fic[i] = np.mean(Q_fic)

for i in range(len(load_set_max)):
	load = load_set_max[i]
	Q_max = simulation_maxweight.main(load,T)
	Q_stat_max[i] = np.mean(Q_max)

print Q_stat_max
print Q_stat_fic

plt.xlabel('Load', fontsize=12)
plt.ylabel('Physical Queue Length', fontsize=12)
line1, = plt.plot(load_set_fic, Q_stat_fic, label='Tracking-MaxWeight')
line2, = plt.plot(load_set_max, Q_stat_max, label='MaxWeight')
plt.legend(handles=[line1, line2], prop={'size': 12})
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax=ymax)
plt.xlim(xmin=0, xmax=1.01)
plt.show()

# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Physical Queue Length', fontsize=12)
# line1, = plt.plot(range(T), Q_fic, label='MaxWeight')
# line2, = plt.plot(range(T), Q_maxweight, label='MaxWeight with Ficticious Play')
# plt.legend(handles=[line1, line2], prop={'size': 12})
# ymin, ymax = plt.ylim()
# plt.ylim(ymin=0, ymax=ymax)
# plt.xlim(xmin=0, xmax=T)
# plt.show()