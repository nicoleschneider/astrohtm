import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

class Viz(object):
	"""
	This class represents the visualization of the spectra in a dataset.
	"""
	_MIN_TIMESTAMP = 1000
	_MAX_TIMESTAMP = 1500

	_MIN_SPECTRA = 0  # must be >= 0
	_MAX_SPECTRA = 30  # must be <= df size - 1

	def __init__(self, input_filename):
		"""
		Parameters:
		------------
		@param input_filename (string)
			The csv file containing the spectral data with one column for timestamp
			and the remaining columns for historgram bins labeled b0, b1, etc
		"""
		self.df = read_csv(input_filename)
		self.num_spectra = len(self.df.columns) - 1
		self.choose_spectra(self._MIN_SPECTRA, self._MAX_SPECTRA)
		print "size:", self.num_spectra

	def trim_timestamp(self, xs, ys, min, max):
		"""
		Parameters:
		------------
		@param xs (list of ints)
			The x-values representing time
		
		@param ys (list of ints)
			The y-values for one speectra column representing photon counts 
			at the corresponding time value in xs
		
		@param min (int)
			The first (lowest) timestamp to include in the visualization
			
		@param max (int)
			The last (highest) timestamp to include in the visualization
		"""
		xs = xs[min:max]
		ys = ys[min:max]
		return xs, ys

	def choose_spectra(self, start, fin):
		"""
		Parameters:
		------------
		@param start (int)
			The first (lowest) spectra to include in the visualization
			
		@param fin (int)
			The last (highest) spectra to include in the visualization
		"""
		droplist = [x for x in range(0, start)] + [y for y in range(fin, self.num_spectra)]
		labels = ['b' + str(z) for z in droplist]
		print(labels)
		self.df = self.df.drop(labels, axis='columns')  # update df
		self.num_spectra = len(self.df.columns) - 1  # update number of spectra being used

	def plot(self, output_filename):
		"""
		Parameters:
		------------
		@param output_filename (string)
			The png file that will contain the image of the visualization when it is complete
		"""
		fig, axs = plt.subplots(self.num_spectra, sharex=True, sharey=True)
		fig.suptitle('Spectra')

		for i in range(self.num_spectra):
			xs = np.array(self.df['timestamp'])
			ys = np.array(self.df[self.df.columns[1::1]])[:,i]
			xs, ys = self.trim_timestamp(xs, ys, self._MIN_TIMESTAMP, self._MAX_TIMESTAMP)
			axs[i].plot(xs, ys)
		
		
		for ax in axs:
			ax.label_outer()
			ax.get_yaxis().set_visible(False)


		plt.xlabel('Timestamp')    
		plt.show()
		plt.savefig(output_filename)

	
if __name__ == "__main__":
	input_filename = 'nu80002092008A01_x2_bary_binned10.csv'
	output_filename = 'spectra.png'
	
	viz = Viz(input_filename)
	viz.plot(output_filename)

	print "Plot was saved to", output_filename
