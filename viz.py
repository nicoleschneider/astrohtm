import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np

class Viz(object):
	"""
	This class represents the visualization of the spectra in a dataset. 
	It takes spectra and timestamp information from a csv file and plots
	the spectra at a specified time interval. The spectra plotted are 
	specified from _MIN_SPECTRA to _MAX_SPECTRA.
	"""
	
	def __init__(self, input_filename, min_time, max_time):
		"""
		Parameters:
		------------
		@param input_filename (string)
			The csv file containing the spectral data with one column for timestamp
			and the remaining columns for historgram bins labeled b0, b1, etc
			
		@param min_time (int)
			The lower bound on the timestamp value to include in the plot
			
		@param max_time (int)
			The upper bound on the timestamp value to include in the plot
		"""
		self.df = read_csv(input_filename)
		self.num_cols = len(self.df.columns) - 1
		
		self._MIN_TIMESTAMP = min_time
		self._MAX_TIMESTAMP = max_time

	def choose_spectra(self, start, fin):
		"""
		Parameters:
		------------
		@param start (int)
			The first (lowest) spectra to include in the visualization
			
		@param fin (int)
			The last (highest) spectra to include in the visualization
		"""
		droplist = [x for x in range(0, start)] + [y for y in range(fin, self.num_cols)]
		labels = ['b' + str(z) for z in droplist]
		print "Dropping the following spectra:", labels
		self.df = self.df.drop(labels, axis='columns')  # drop spectra from the dataframe
		self.num_cols = len(self.df.columns) - 1  # update number of columns being used
		
		
	def add_anomalies(self, anomaly_filename):
		"""
		Parameters:
		------------
		@param anomaly_filename (string)
			The csv file containing timestamps in float form and anomaly scores
		"""
		anomaly_df = read_csv(anomaly_filename)
		anomaly_df = anomaly_df.drop("b0", axis='columns')
		anomaly_df = anomaly_df.drop("scaled_score", axis='columns')

		self.df = self.df.join(anomaly_df.set_index('timestamp'), on='timestamp',lsuffix='_caller', rsuffix='_temp')
		print self.df
		
		first_cols = ['timestamp', 'anomaly_score']
		self.df = self.df[[c for c in first_cols if c in self.df] + [c for c in self.df if c not in first_cols]]
		print self.df
		self.num_cols = len(self.df.columns) - 1  # update number of columns being used

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
		
	def plot(self, output_filename):
		"""
		Parameters:
		------------
		@param output_filename (string)
			The png file that will contain the image of the visualization when it is complete
		"""
		fig, axs = plt.subplots(self.num_cols, sharex=True, sharey=True)
		fig.suptitle('Spectra')

		for i in range(self.num_cols):
			xs = np.array(self.df['timestamp'])
			ys = np.array(self.df[self.df.columns[1:]])[:,i]
			xs, ys = self.trim_timestamp(xs, ys, self._MIN_TIMESTAMP, self._MAX_TIMESTAMP)
			if i == 0:  # if anomaly score
				axs[i].plot(xs, ys, 'r')  # plot as red
			else:
				axs[i].plot(xs, ys, 'b')  # plot as blue
		
		for ax in axs:
			ax.label_outer()  # remove axis info on all but outer plots
			ax.get_yaxis().set_visible(False)

		plt.xlabel('Timestamp')    
		plt.show()
		plt.savefig(output_filename)

	
if __name__ == "__main__":
	input_filename = 'nu80002092008A01_x2_bary_binned10.csv'
	output_filename = 'spectra.png'
	anomaly_filename = 'spectrum4.csv'
	
	min_time = 0  # must be >= 0
	max_time = 1500  # must be <= max time in dataset i.e. 6000 or less

	min_spectra = 0  # must be >= 0
	max_spectra = 30  # must be <= df size - 1 i.e. 30 or less
	
	viz = Viz(input_filename, min_time, max_time)
	viz.choose_spectra(min_spectra, max_spectra)
	viz.add_anomalies(anomaly_filename)
	print "size:", viz.num_cols
		
	viz.plot(output_filename)

	print "Plot was saved to", output_filename
