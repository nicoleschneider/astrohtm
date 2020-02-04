import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import sys

class Viz(object):
	"""
	This class represents the visualization of the spectra in a dataset. 
	It takes spectra and timestamp information from a csv file and plots
	the spectra at a specified time interval. The spectra plotted are 
	specified from _MIN_SPECTRA to _MAX_SPECTRA.
	"""
	
	def __init__(self, input_filename, min_time, max_time, cutoffs=[]):
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
		self.datafile = input_filename
		self.df = read_csv(input_filename)
		self.num_cols = len(self.df.columns) - 1
		
		self._MIN_TIMESTAMP = min_time
		self._MAX_TIMESTAMP = max_time
		self.cutoffs = cutoffs

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
		
		first_cols = ['timestamp', 'anomaly_score']
		self.df = self.df[[c for c in first_cols if c in self.df] + [c for c in self.df if c not in first_cols]]
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
		
		for ax in axs:
			ax.label_outer()  # remove axis info on all but outer plots
			
		anom = fig.add_subplot(self.num_cols, 1, 1, sharex=axs[0])
		anom.get_xaxis().set_visible(False)
		
		fig.suptitle(self.datafile)
		fig.subplots_adjust(hspace=0)

		for i in range(self.num_cols):
			xs = np.array(self.df['timestamp'])			
			ys = np.array(self.df[self.df.columns[1:]])[:,i]
			xs, ys = self.trim_timestamp(xs, ys, self._MIN_TIMESTAMP, self._MAX_TIMESTAMP)
		
			if i == 0:  # if anomaly scores column
				l1, = anom.plot(xs, ys, 'r')
				anom.yaxis.set_label_position("right")
			else:
				l2, = axs[i].plot(xs, ys, 'b')  # plot as blue
				axs[i].set_yticklabels([])
				axs[i].yaxis.set_label_position("right")
				axs[i].set_ylabel(self.df.columns[i+1], labelpad=15, rotation=0)
				
		anom.legend([l1, l2], ['Anomaly Score', 'Photon Count'], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
		
		fig.text(0.5, 0.04, 'Timestamp', ha='center', va='center')
		fig.text(0.06, 0.5, 'Photon Count', ha='center', va='center', rotation='vertical')

		plt.show()
		plt.savefig(output_filename)
		print "Plot was saved to", output_filename
		
		
	def make_spectrum(self, output_filename, row_num):
		"""
		Parameters:
		------------
		@param output_filename (string)
			The png file that will contain the image of the visualization when it is complete
			
		@param row_num (int)
			Row number ocrresponding to the occurrence of the anomaly that the spectrum will be made for
		"""
		
		buffer = 2  # how many spectra on either side of the anomaly point you want
		
		fig, axs = plt.subplots(2*buffer + 2)  # num plots on either side, plus center, plus anomaly score plot
		
		# plot anomalies
		xs = np.array(self.df['timestamp']).astype(int)
		ys = np.array(self.df[self.df.columns[1:]])[:,0]
		xs, ys = self.trim_timestamp(xs, ys, row_num-buffer, row_num+buffer+1)  # trim to one before one after row_num

		l1, = axs[0].plot(xs, ys, 'r')
		axs[0].yaxis.set_label_position("right")
		axs[0].set_xlabel("Timestamp (seconds)", labelpad=0, size='small')
		axs[0].set_xticks(xs)
		axs[0].tick_params(axis='x', which='major', labelsize=8)
		axs[0].set_ylim([0, 1.1])
		
		
		# plot spectra
		for i, row in enumerate(range(row_num-buffer, row_num+buffer+1)):
			data = self.df.iloc[row, 2:]  # the 2: gets rid of timestamp and anomaly score
			labels = data.axes[0]
			labels = [x[1:] for x in labels]  # strip the 'b' off labels
			
			axs[i+1].bar(labels, data)  # +1 to leave first subplot for anomaly scores
			axs[i+1].tick_params(axis='x', which='major', labelsize=8)
			axs[i+1].set_ylabel("T = "+str(xs[i]), labelpad=25, rotation=0)
			axs[i+1].yaxis.set_label_position("right")

		
		# edit overall figure
		fig.suptitle(self.datafile)
		fig.subplots_adjust(hspace=0.5)
		fig.text(0.5, 0.04, 'Channel', ha='center', va='center')
		fig.text(0.06, 0.38, 'Spectra', ha='center', va='center', rotation='vertical')
		fig.text(0.06, 0.8, 'Anomaly Score', ha='center', va='center', rotation='vertical')
		
		plt.show()
		plt.savefig(output_filename)
		print "Plot was saved to", output_filename
		
	def generate_spectra(self, threshold, start_time, stop_time):
		for i, score in enumerate(self.df['anomaly_score']):
			# if anomaly and occurs not at start or endpoint of data and is within desired time bounds:
			if (score > threshold) & (i < len(self.df)-1) & (i > 3) & (i > start_time) & (i < stop_time):
				print i
				self.make_spectrum('hist.png', i)
				
				
	
if __name__ == "__main__":

	start_time = int(sys.argv[1])
	stop_time = int(sys.argv[2])
	input_filename = sys.argv[3]
	
	#input_filename = 'nu80002092008A01_x2_bary_binned10.csv' "ni1103010157_0mpu7_cl_binned10.csv"
	output_filename = 'spectra.png'
	anomaly_filename = 'spectrum4.csv'
	
	min_time = 0  # must be >= 0
	max_time = 6000  # must be <= max time in dataset i.e. 6000 or less

	min_spectra = 0  # must be >= 0
	max_spectra = 30  # must be <= df size - 1 i.e. 30 or less
	
	viz = Viz(input_filename, min_time, max_time)
	viz.choose_spectra(min_spectra, max_spectra)
	viz.add_anomalies(anomaly_filename)
	print "size:", viz.num_cols
		
	#viz.plot(output_filename)
	viz. make_spectrum('hist.png', 2125)
	viz.generate_spectra(0.5, start_time, stop_time)
