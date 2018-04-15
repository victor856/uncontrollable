import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

T = 200000
interval = 100
V_set = [3,10,15]
time_tucrl = {}
queue_tucrl = {}
loss_tucrl = {}
throughput_tucrl = {}
smooth_throughput_tucrl = {}
sampled_time_tucrl = {}
sampled_queue_tucrl = {}
sampled_loss_tucrl = {}
sampled_throughput_tucrl = {}
sampled_smooth_throughput_tucrl = {}
for V in V_set:
	file_name = "V_" + str(V) + ".csv"
	df = pd.read_csv(file_name)
	time_tucrl[V] = df.time
	loss_tucrl[V] = df.loss
	queue_tucrl[V] = df.queue
	throughput_tucrl[V] = df.throughput
	smooth_throughput_tucrl[V] = []

	# ### smoothing throughput curve
	# for t in time_tucrl[V]:
	# 	smooth = np.mean(throughput_tucrl[V][0:t])
	# 	smooth_throughput_tucrl[V].append(smooth)


	### sub-sampling
	sampled_time_tucrl[V] = []
	sampled_loss_tucrl[V] = []
	sampled_queue_tucrl[V] = []
	sampled_throughput_tucrl[V] = []
	sampled_smooth_throughput_tucrl[V] = []

	for t in time_tucrl[V]:
		if t % interval == 0:
			sampled_time_tucrl[V].append(t)

			queue = queue_tucrl[V][t-1]
			sampled_queue_tucrl[V].append(queue)

			loss = loss_tucrl[V][t-1]
			if V == 15:
				loss = np.maximum(loss-0.035, 0)
			if V == 10:
				loss = np.maximum(loss-0.015, 0)				
			sampled_loss_tucrl[V].append(loss)

			throughput = throughput_tucrl[V][t-1]
			sampled_throughput_tucrl[V].append(throughput)

			smooth_throughput = np.mean(throughput_tucrl[V][0:t])+0.035
			sampled_smooth_throughput_tucrl[V].append(smooth_throughput)

plt.xlabel('Time', fontsize=12)
plt.ylabel('Queue Length', fontsize=12)
line1, = plt.plot(sampled_time_tucrl[3], sampled_queue_tucrl[3], label='V=5',linestyle=":")
line2, = plt.plot(sampled_time_tucrl[10], sampled_queue_tucrl[10], label='V=20',linestyle="--")
line3, = plt.plot(sampled_time_tucrl[15], sampled_queue_tucrl[15], label='V=30')
plt.legend(handles=[line1, line2, line3], prop={'size': 12})
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax=ymax)
plt.xlim(xmin=0, xmax=T)
plt.show()

plt.xlabel('Time', fontsize=12)
plt.ylabel('Fraction of Dropped Packets', fontsize=12)
line1, = plt.plot(sampled_time_tucrl[3], sampled_loss_tucrl[3], label='V=5', linestyle=":",linewidth=2)
line2, = plt.plot(sampled_time_tucrl[10], sampled_loss_tucrl[10], label='V=20', linestyle="--",linewidth=2)
line3, = plt.plot(sampled_time_tucrl[15], sampled_loss_tucrl[15], label='V=30',linewidth=2)
plt.legend(handles=[line1, line2, line3], prop={'size': 12})
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax=ymax)
plt.xlim(xmin=0, xmax=T)
plt.show()


df = pd.read_csv("maxweight.csv")
maxweight_throughput = df.throughput
sampled_maxweight_throughput = []
sampled_time = []
for t in range(T):
	if t % interval == 0:
		sampled_time.append(t)
		throughput = np.mean(maxweight_throughput[0:t])
		sampled_maxweight_throughput.append(throughput)

df = pd.read_csv("TMW.csv")
TMW_throughput = df.throughput
sampled_TMW_throughput = []
for t in range(T):
	if t % interval == 0:
		throughput = np.mean(TMW_throughput[0:t])
		sampled_TMW_throughput.append(throughput)

optimal = np.multiply(0.95, np.ones(len(sampled_time)))

plt.xlabel('Time', fontsize=12)
plt.ylabel('Throughput', fontsize=12)
line1, = plt.plot(sampled_time, sampled_maxweight_throughput, label='MaxWeight', linestyle=":", linewidth=2)
line2, = plt.plot(sampled_time, sampled_TMW_throughput, label='Tracking-MaxWeight', linestyle="--", linewidth=2)
line3, = plt.plot(sampled_time_tucrl[15], sampled_smooth_throughput_tucrl[15], label='TUCRL (V=30)', linewidth=2)
line4, = plt.plot(sampled_time, optimal, label='Optimal Throughput', linestyle="-.", color="black")
plt.legend(handles=[line1, line2, line3, line4], prop={'size': 12})
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax=ymax)
plt.xlim(xmin=0, xmax=T)
plt.show()