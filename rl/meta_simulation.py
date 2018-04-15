import simulation_tucrl
import simulation_maxweight
import simulation_TMW
import numpy as np
import matplotlib.pyplot as plt
import csv

T = 200000
load = 0.95
V_set = [10]

for V in V_set:
	output_file_name = "V_" + str(V) + ".csv" 
	with open(output_file_name,'w+') as f:
		f.write("time,queue,loss,throughput\n")

	(t_checkpoint, Q_checkpoint, loss_rate_checkpoint, throughput_checkpoint) = simulation_tucrl.main(T,load,V)
	with open(output_file_name,'a') as f:
		for i in range(len(t_checkpoint)):
			t = t_checkpoint[i]
			Q = Q_checkpoint[i]
			loss = loss_rate_checkpoint[i]
			throughput = throughput_checkpoint[i]
			f.write("%s,%s,%s,%s" % (t,Q,loss,throughput))
			if i < len(t_checkpoint) - 1:
				f.write("\n")


# throughput_maxweight = simulation_maxweight.main(T,load)
# output_file_name = "maxweight.csv" 
# with open(output_file_name,'w+') as f:
# 	f.write("time,throughput\n")

# with open(output_file_name,'a') as f:
# 	for t in range(T):
# 		f.write("%s,%s" % (t,throughput_maxweight[t]))
# 		if t < T - 1:
# 			f.write("\n")

# throughput_TMW = simulation_TMW.main(T,load)
# output_file_name = "TMW.csv" 
# with open(output_file_name,'w+') as f:
# 	f.write("time,throughput\n")

# with open(output_file_name,'a') as f:
# 	for t in range(T):
# 		f.write("%s,%s" % (t,throughput_TMW[t]))
# 		if t < T - 1:
# 			f.write("\n")
