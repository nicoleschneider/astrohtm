import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

input_filename = 'nu80002092008A01_x2_bary_binned10.csv'
output_filename = 'spectra.png'
_MIN_TIMESTAMP = 1000
_MAX_TIMESTAMP = 1500

_MIN_SPECTRA = 0  # must be >= 0
_MAX_SPECTRA = 30  # must be <= df size - 1

def trim_timestamp(xs, ys, min, max):
	xs = xs[min:max]
	ys = ys[min:max]
	return xs, ys

def choose_spectra(df, start, fin):
	droplist = [x for x in range(0, start)] + [y for y in range(fin, len(df.columns)-1)]
	labels = ['b' + str(z) for z in droplist]
	print(labels)
	return df.drop(labels, axis='columns')


df = read_csv(input_filename)
df = choose_spectra(df, _MIN_SPECTRA, _MAX_SPECTRA)
print "size:", len(df.columns)

fig, axs = plt.subplots(len(df.columns)-1, sharex=True, sharey=True)
fig.suptitle('Spectra')

for i in range(len(df.columns)-1):
	xs = np.array(df['timestamp'])
	ys = np.array(df[df.columns[1::1]])[:,i]
	xs, ys = trim_timestamp(xs, ys, _MIN_TIMESTAMP, _MAX_TIMESTAMP)
	axs[i].plot(xs, ys)
	
	
for ax in axs:
	ax.label_outer()
	ax.get_yaxis().set_visible(False)


plt.xlabel('Timestamp')    
plt.show()
plt.savefig(output_filename)

print "Plot was saved to", output_filename
