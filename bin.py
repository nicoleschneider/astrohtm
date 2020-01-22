import sys
from astropy.table import Table
import tqdm
import numpy as np

class Bin(object):

	def __init__(self, argv):
		"""
		Parameters:
		------------
		@param argv (list of strings)
			Command line arguments
		"""
		self.parse_args(argv)
		self.NUM_BINS = 30
		
		
	def parse_args(self, argv):
		"""
		Parse command line arguments
		"""
		self.lower_bound = float(argv[1])
		self.upper_bound = float(argv[2])
		self.bin_time = int(argv[3])
		self.event_files = argv[4:]
		
		
	def run(self):
		for event_file in self.event_files:
			t = Table.read(event_file)

			x_label = 'RAWX'
			y_label = 'RAWY'
			
			t0 = t.meta['TSTART']
			t1 = t.meta['TSTOP']
			new_times = np.arange(t0, t1, self.bin_time)

			list_of_rows = []
			energy = t['PI']*0.04 + 1.6
			for nt in tqdm.tqdm(new_times):
				good = (t['TIME'] >= nt)&(t['TIME'] < nt + self.bin_time)
				img, _, _ = np.histogram2d(t[x_label][good], t[y_label][good],
										   range=((400, 600), (400, 600)), bins=self.NUM_BINS)
				spec, _ = np.histogram(energy[good], 
					range=[self.lower_bound, self.upper_bound], bins=self.NUM_BINS)
				res = {'Time': nt, 'data': img, 'spec': spec}
				list_of_rows.append(res)
			
			result_table = Table(list_of_rows)
			result_table.write(
				event_file.replace('.gz', '').replace('.evt', '') 
				+ '_binned' + str(self.bin_time) + '.fits', overwrite=True
			)
		

if __name__ == '__main__':

	bin = Bin(sys.argv)
	bin.run()
	
