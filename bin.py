import sys
import tqdm

import numpy as np
import pandas as pd

from astropy.table import Table

class Bin(object):

	def __init__(self, argv):
		"""
		Parameters:
		------------
		@param argv (list of strings)
			Command line arguments for:
				Lower energy bound
				Upper energy bound
				Number of seconds to aggregate by (10 sec)
				Number of bins to use in spectrum (30 bins)
				'CONST' for constant binning, otherwise any other string for normal binning
				Event filenames
		"""
		self.parse_args(argv)
		
		
		
	def parse_args(self, argv):
		"""
		Parse command line arguments
		"""
		self.lower_bound = float(argv[1])
		self.upper_bound = float(argv[2])
		self.bin_time = int(argv[3])
		self.channels = int(argv[4])
		self.CONST = True if argv[5] == 'CONST' else False
		print self.CONST
		self.event_files = argv[6:]
		
	def make_bin_edges(self, t, t0, t1):
		'''
		Find bin edges that make channel divisions so that there photon counts are equal per channel
		'''
		interval = (t['TIME'] >= t0)&(t['TIME'] < t1)
		if len(t['PI'][interval]) >= self.channels:
			qc, self.bin_edges = pd.qcut(t['PI'][interval], q=self.channels, retbins=True, duplicates='drop')
				
		print qc
		print self.bin_edges
		
		
	def run(self):
		for event_file in self.event_files:
			t = Table.read(event_file)
			print(t)

			x_label = 'RAWX'
			y_label = 'RAWY'
			
			t0 = t.meta['TSTART']
			t1 = t.meta['TSTOP']

			new_times = np.arange(t0, t1, self.bin_time)
			print new_times, "\n"
			   				   
			if self.CONST:
				self.make_bin_edges(t, t0, t1)
			else:
				self.bin_edges = ['na']

			list_of_rows = []
			energy = t['PI']*0.04 + 1.6  # valid for NuSTAR only

			self.lower_bound = np.min(t['PI'])  # since we aren't using energy
			self.upper_bound = np.max(t['PI'])  # since we aren't using energy
			
			for nt in tqdm.tqdm(new_times):
				good = (t['TIME'] >= nt)&(t['TIME'] < nt + self.bin_time)
				
				# make image
				img, _, _ = np.histogram2d(t[x_label][good], t[y_label][good],
										   range=((400, 600), (400, 600)), bins=self.channels)
				# make spectrum	   				   
				if self.CONST:
					spec, _ = np.histogram(t['PI'][good], 
						range=[self.lower_bound, self.upper_bound], bins=self.bin_edges)
				else:
					spec, _ = np.histogram(t['PI'][good], 
						range=[self.lower_bound, self.upper_bound], bins=self.channels)
					
				res = {'Time': nt, 'data': img, 'spec': spec, 'cutoffs':self.bin_edges}
				list_of_rows.append(res)
			
			result_table = Table(list_of_rows)
			print result_table, "\n"
			
			if self.CONST:
				result_table.write(
					event_file.replace('.gz', '').replace('.evt', '') 
					+ '_binned' + str(self.bin_time) + 'CONST.fits', overwrite=True
				)
			else:
				result_table.write(
					event_file.replace('.gz', '').replace('.evt', '') 
					+ '_binned' + str(self.bin_time) + '.fits', overwrite=True
				)
		

if __name__ == '__main__':

	bin = Bin(sys.argv)
	bin.run()
	
