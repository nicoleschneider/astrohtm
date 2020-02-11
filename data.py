import csv
import tqdm
import numpy as np

import astropy
from astropy.io import fits

class Data(object):
	"""
	This class handles the data and reads it into a table so it can be easily used
	by the anomaly detection algorithm.
	"""
	
	def __init__(self, source_file, headers):
		"""
		Parameters:
		------------
		@param source_file (string)
			The filename where the data is stored (.fits file expected)
		
		@param headers (int array)
			List of the headers (timestamp nad spectrum labels like b0, b1, etc)
			to be considered.
		"""
		self.headers = headers
	
		self.hdu_list = fits.open(source_file)
		print self.hdu_list.info()
		
		self.table = astropy.table.Table.read(self.hdu_list[1])
		self.data_size = len(self.table)
		print "LENGTH IS:", self.data_size
		
		self.timestamps = self.table.field('Time')
		self.images = self.table.field('data') # we ignore for now
		self.spectrum = self.table.field('spec')
		self.spectrum_size = len(self.headers)-1
		self.cutoffs = self.table.field('cutoffs')

		
	
	def set_input_stats(self):	
		self._INPUT_MIN = [0]*self.spectrum_size  # minimum metric value of the input data
		self._INPUT_MAX = [0]*self.spectrum_size  # maximum metric value of the input data
		
		for i in range(self.spectrum_size):
			self._INPUT_MAX[i] = np.max(self.spectrum[:,i])
			self._INPUT_MIN[i] = np.min(self.spectrum[:,i])

		
	def select_cols(self, min_variance, model_params):
		"""
		Eliminate all data columns that fail to meet 
		the minimum variance requirement
		Parameters:
		------------
		@param min_var (int)
			The minimum variance a spectrum column will have else it is dropped
		
		@param model_params (dictionary)
			The dictionary of parameters for the HTM model to used
		"""
		for i, element in enumerate(self.headers[1:]):    
			self.spectrum[:,i] =  self.spectrum[:,i] - np.mean(self.spectrum[:,i]) #/( np.std(self.spectrum[:,i]) )
		#  _INPUT_MAX[i] =  _INPUT_MAX[i] - np.mean(self.spectrum[:,i]) #/( np.std(self.dspectrum[:,i]) )
			self.spectrum[:,i] = map(lambda x: max(x,0), self.spectrum[:,i])
			
			if np.var(self.spectrum[:,i]) < min_variance:
				self.headers.remove(element)
				model_params["modelParams"]["sensorParams"]["encoders"].pop(element)
				print "SELECT COL JUST REMOVED", element, "with variance", np.var(self.spectrum[:,i])
			  
		 # _INPUT_MAX = map(lambda x: max(x,0), _INPUT_MAX)	  
		print "ENDED UP USING", len(self.headers), "columns total"
		
		if len(self.headers) <= 1:
			print "NO CHANNELS WERE KEPT, ABORTING TEST"
			exit(0)
		
		
	def replace_bad_intervals(self):
		"""
		Pull power spectrum noise from csv file to fill in bad time intervals
		"""
		df = pd.read_csv("psd1.csv")
		saved_column = df['final signal'].values
		saved_column = np.append(saved_column, [0])
		self.spectrum[:,0] = saved_column

		print self.spectrum
		
	def generate_record(self, index, header_list = []):
		"""
		Parameters:
		------------
		@param index (int)
			The row number to generate a record from
		
		@param header_list (int array)
			List of the headers (timestamp nad spectrum labels like b0, b1, etc)
			to be written to the csv file. Default uses all headers in the data
			object
		"""
		if header_list == []:
			header_list = self.headers
			
		record = []
		for x, label in enumerate(header_list[1:]):
			col_number = int(label[1:])
			record.append(self.spectrum[:,col_number][index]) 
	
		record = np.insert(record, 0, self.timestamps[index]-self.timestamps[0]) # insert timestamp value to front of record
		return record
		
		
	def write_data_to_csv(self, output_file, header_list=[]):
		"""
		Write the original spectral data to a csv file
		Parameters:
		------------
		@param output_file (string)
			The filename to whioh the output will be written (should be .csv)
		
		@param header_list (int array)
			List of the headers (timestamp nad spectrum labels like b0, b1, etc)
			to be written to the csv file. Default uses all headers in the data
			object
		"""

		if header_list == []:
			header_list = self.headers
		
		csvWriter = csv.writer(open(output_file,"wb"))
		csvWriter.writerow(header_list)
		print "Headers to be written are:", self.headers
		
		for i in tqdm.tqdm(range(0, self.data_size, 1), desc='% Complete'):
			record = self.generate_record(i, header_list)
			modelInput = dict(zip(header_list, record))
			
			for label in header_list:
				modelInput[label] = float(modelInput[label])
				
			csvWriter.writerow([modelInput[x] for x in header_list])

		print "Original data has been written to", output_file
