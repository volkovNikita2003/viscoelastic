import numpy as np
import matplotlib.pyplot as plt


filename1 = "impulse.txt"
source1 = np.loadtxt(filename1)

f0 = 18.0
t0 = 1.2 / f0
filename2 = "plot_source_time_function.txt"
source2 = np.loadtxt(filename2)
source2[:, 0] += t0
# source2[:, 1] *= (-1)

plt.plot(source1[:, 0], source1[:, 1], label="my")
plt.plot(source2[:, 0], source2[:, 1], label="specfem")
plt.xlabel("t, s")
plt.ylabel("amplitude")
plt.legend()
plt.savefig("impulses.png")
