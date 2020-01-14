import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

input_filename = 'nu80002092008A01_x2_bary_binned10.csv'
output_filename = 'spectra.png'
df = read_csv(input_filename)

print(df)
print("size: ", len(df.columns))

for i in range(len(df.columns)-1):
	xs = np.array(df['timestamp']) 
	ys = np.array(df[df.columns[1::1]])[:,i]
	plt.subplot(len(df.columns)-1, 1, i+1)
	plt.plot(xs, ys)


plt.xlabel('Timestamp')
plt.ylabel('Photon Count')

plt.grid(alpha=0.1)    
plt.yscale('log')

plt.show()
plt.savefig(output_filename)
