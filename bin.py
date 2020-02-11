import tqdm
import argparse

import numpy as np
import pandas as pd

from astropy.table import Table

class Args(object):

	def __init__(self):
		"""
		Create argument parser and parse the command line args.
		"""	
		self.parser = argparse.ArgumentParser(description='Bin event data.')
		self.add_args()
		self.args = self.parser.parse_args()


	def add_args(self):
		"""
		Add command line arguments to the parser.
		"""	
		self.parser.add_argument('bounds', metavar='B', type=float, nargs=2,
                    help='the lower and upper energy bounds for the data')
		self.parser.add_argument('bin_time', metavar='T', type=int,
                    help='the number of seconds to aggregate the data by')
		self.parser.add_argument('channels', metavar='C', type=int,
                    help='the number of channels for the histogram')
		self.parser.add_argument('event_files', metavar='E', type=str, nargs='+',
                    help='the .evt file to be binned')
		self.parser.add_argument('--constant', dest='binning_function', action='store_const',
                    const=Bin.do_constant_binning, default=Bin.do_binning,
                    help='use constant number of phtotons per bin (default: time-based binning)')


		
			
class Bin(object):

	def __init__(self):
		"""
		Create bin object
		"""
		
	def do_binning(self, args):  # ORGINIAL NON CONSTANT BINNING
		'''
		Bin the data according to the number of channels specified in args.
		'''
		print "doing normal binning"
		
		for event_file in args.event_files:
			t = Table.read(event_file)
			print(t)

			x_label = 'RAWX'
			y_label = 'RAWY'
			
			t0 = t.meta['TSTART']
			t1 = t.meta['TSTOP']
			
			self.lower_bound = np.min(t['PI'])  # since we aren't using energy, else would be self.lower_bound = args.lower_bound
			self.upper_bound = np.max(t['PI'])  # since we aren't using energy, else would be self.upper_bound = args.upper_bound
			self.channels = args.channels
			self.bin_time = args.bin_time

			new_times = np.arange(t0, t1, args.bin_time)
			print new_times, "\n"
			   				   

			self.bin_edges = ['na']  # FIX ME 

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
				spec, _ = np.histogram(t['PI'][good], 
					range=[self.lower_bound, self.upper_bound], bins=self.channels)
					
				res = {'Time': nt, 'data': img, 'spec': spec, 'cutoffs':self.bin_edges}
				list_of_rows.append(res)
			
			result_table = Table(list_of_rows)
			print result_table, "\n"
			

			result_table.write(
				event_file.replace('.gz', '').replace('.evt', '') 
				+ '_binned' + str(self.bin_time) + '.fits', overwrite=True
			)

		
	def make_bin_edges(self, t, t0, t1):
		'''
		Find bin edges that make channel divisions so that there photon counts are equal per channel
		'''
		interval = (t['TIME'] >= t0)&(t['TIME'] < t1)
		if len(t['PI'][interval]) >= self.channels:
			qc, self.bin_edges = pd.qcut(t['PI'][interval], q=self.channels, retbins=True, duplicates='drop')
				
		print qc
		print "bin edges are:", self.bin_edges
		
		
	def do_constant_binning(self, args):  # CONSTANT BINNING
		'''
		Bin the data so that the bin edges make the photons roughly evenly distributed through the histogram
		'''
		print "doing constant binning"
		
		for event_file in args.event_files:
			t = Table.read(event_file)
			print(t)

			x_label = 'RAWX'
			y_label = 'RAWY'
			
			t0 = t.meta['TSTART']
			t1 = t.meta['TSTOP']
			
			self.lower_bound = np.min(t['PI'])  # since we aren't using energy, else would be self.lower_bound = args.lower_bound
			self.upper_bound = np.max(t['PI'])  # since we aren't using energy, else would be self.upper_bound = args.upper_bound
			self.channels = args.channels
			self.bin_time = args.bin_time

			new_times = np.arange(t0, t1, self.bin_time)
			print new_times, "\n"
			   				   
			self.make_bin_edges(t, t0, t1)

			list_of_rows = []
			energy = t['PI']*0.04 + 1.6  # valid for NuSTAR only
			
			for nt in tqdm.tqdm(new_times):
				good = (t['TIME'] >= nt)&(t['TIME'] < nt + self.bin_time)
				
				# make image
				img, _, _ = np.histogram2d(t[x_label][good], t[y_label][good],
										   range=((400, 600), (400, 600)), bins=self.channels)
				# make spectrum	   				   
				spec, _ = np.histogram(t['PI'][good], 
					range=[self.lower_bound, self.upper_bound], bins=self.bin_edges)
				
				res = {'Time': nt, 'data': img, 'spec': spec, 'cutoffs':self.bin_edges}
				list_of_rows.append(res)
			
			result_table = Table(list_of_rows)
			print result_table, "\n"
			
			result_table.write(
				event_file.replace('.gz', '').replace('.evt', '') 
				+ '_binned' + str(self.bin_time) + 'CONST.fits', overwrite=True
			)
			
		

if __name__ == '__main__':
	args = Args()
	ARGS = args.args
	
	bin = Bin()
	ARGS.binning_function(bin, ARGS)

