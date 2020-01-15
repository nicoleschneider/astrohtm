import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

input_filename = 'nu80002092008A01_x2_bary_binned10.csv'
output_filename = 'spectra.png'
df = read_csv(input_filename)

print "size:", len(df.columns)

fig, axs = plt.subplots(len(df.columns)-1, sharex=True, sharey=True)
fig.suptitle('Spectra')

for i in range(len(df.columns)-1):
	xs = np.array(df['timestamp'])
	ys = np.array(df[df.columns[1::1]])[:,i]
	axs[i].plot(xs, ys)
	

for ax in axs:
	ax.label_outer()
	ax.get_yaxis().set_visible(False)


plt.xlabel('Timestamp')    
plt.show()
plt.savefig(output_filename)

print "Plot was saved to", output_filename
