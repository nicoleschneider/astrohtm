# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
A client to process an astronomy dataset and create a HTM anomaly detection model for it.
"""
from __future__ import division

import csv
import datetime
import logging
import tqdm
import optuna
import copy
import numpy as np
import pandas as pd

from pkg_resources import resource_filename
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter
from nupic.algorithms import anomaly_likelihood as AL
from viz import Viz

import model_params
import astropy
from astropy.io import fits
from astropy.io import ascii


class Output(object):
	"""
	This class handles the output of data and results to csv file
	format for future visualization.
	"""

	def __init__(self, output_path):
		"""
		Parameters:
		------------
		@param output_path (string)
			The filename to whioh the output will be written
		"""
		
		self.fid = open(output_path, "wb")
		self.csvWriter = csv.writer(self.fid)
		
	def write(self, entry_array):
		self.csvWriter.writerow(entry_array)
		
	def close(self):
		self.fid.close()

class Data(object):
	"""
	This class handles the data and reads it into a table so it can be easily used
	by the anomaly detection algorithm.
	"""
	
	headers = ["timestamp", "b0"]
	
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
		self.table = astropy.table.Table.read(self.hdu_list[1])
		self.data_size = len(self.table)
		print("LENGTH IS: ", self.data_size)
		
		self.timestamps = self.table.field(0)
		self.images = self.table.field(1) # we ignore for now
		self.spectrum = self.table.field(2)
		print(self.spectrum)
		
	
	def set_input_stats(self):	
		self._INPUT_MIN = [0]*30  # minimum metric value of the input data
		self._INPUT_MAX = [0]*30  # maximum metric value of the input data
		
		for i in range(len(self.headers)-1):
			self._INPUT_MAX[i] = np.max(self.spectrum[:,i])
			self._INPUT_MIN[i] = np.min(self.spectrum[:,i])

		
		
	def select_cols(self, min_variance, model_params):
		"""
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

			print "Variance is:", np.var(self.spectrum[:,i])
			
			if np.var(self.spectrum[:,i]) < min_variance:
				self.headers.remove(element)
				model_params["modelParams"]["sensorParams"]["encoders"].pop(element)
				print("___________ SELECT COL JUsT REMOVED___________", element)
			  
		 # _INPUT_MAX = map(lambda x: max(x,0), _INPUT_MAX)	  
		print("ENDED UP USING ", len(self.headers), " columns total")
		
		
	def replace_bad_intervals(self):
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
		print(self.headers)
		
		for i in tqdm.tqdm(range(0, self.data_size, 1), desc='% Complete'):
			record = self.generate_record(i, header_list)
			modelInput = dict(zip(header_list, record))
			
			for label in header_list:
				modelInput[label] = float(modelInput[label])
				
			csvWriter.writerow([modelInput[x] for x in header_list])

		print("Original data has been written to",output_file)
		
		

class AstroHTM(object):
	"""
	This class represents the hierarchical temporal memory algorithm to be 
	applied to astronomy data.
	"""
	_LOGGER = logging.getLogger(__name__)
	
	##_SOURCE_FILE = './srcB_3to40_cl_barycorr_binned_multiD.fits'
	_SOURCE_FILE = 'nu80002092008A01_x2_bary_binned10.fits'
	#_SOURCE_FILE = 'ni1103010157_0mpu7_cl_binned10.fits'
	
	_MIN_VARIANCE = 0
	_ANOMALY_SCALE_FACTOR = 300

	anomaly_count = 0
	encoder_resolution_set = False


	
	def __init__(self, min_var, headers, model_params, output_path, select_cols=False, threshold = 0.5):
		"""
		Parameters:
		------------
		@param min_var (int)
			The minimum variance a spectrum column will have else it is dropped
		
		@param model_params (dictionary)
			The dictionary of parameters for the HTM model to used
			
		@param output_path (string)
			The filename to whioh the output will be written (.csv extected)
			
		@param select_cols (boolean)
			True if columns should be removed for having low variance, false otherwise
			
		@param threshold (float from 0 to 1)
			Determines how high an anomaly score must be in order to register as an anomaly
		"""
		
		self._MIN_VARIANCE = min_var		
		self._SELECT_COLS = select_cols
		self._ANOMALY_THRESHOLD = threshold
		self.data = Data(self._SOURCE_FILE, headers)
		print(len(headers), " headers given originally")
		#self.model = self.createModel()
		self.model_params = copy.deepcopy(model_params)
		self._OUTPUT_PATH = output_path
		self.data.set_input_stats()

	def get_anomaly_count(self):
		return self.anomaly_count

	def _setRandomEncoderResolution(self, minResolution=0.001):
		"""
		Given model params, figure out the correct resolution for the
		RandomDistributed encoders. Modifies params in place.
		"""
		fields = self.data.headers[1:]

		for i, field in enumerate(fields):
			encoder = self.model_params["modelParams"]["sensorParams"]["encoders"][field]

			if encoder["type"] == "RandomDistributedScalarEncoder" and "numBuckets" in encoder:
				rangePadding = abs(self.data._INPUT_MAX[i] - self.data._INPUT_MIN[i]) * 0.2
				minValue = self.data._INPUT_MIN[i] - rangePadding
				maxValue = self.data._INPUT_MAX[i] + rangePadding
				resolution = max(minResolution, (maxValue - minValue) / encoder.pop("numBuckets") )
				encoder["resolution"] = resolution
				print "RESOLUTION________________", resolution

			self.model_params['modelParams']['sensorParams']['encoders'][field] = encoder
			
		self.encoder_resolution_set = True
		
		for i in self.model_params['modelParams']['sensorParams']['encoders'].keys():
			if i not in self.data.headers:
				self.model_params['modelParams']['sensorParams']['encoders'].pop(i)
				print(i, " was removed in reolution")


	def createModel(self):
		self._setRandomEncoderResolution()
		return ModelFactory.create(self.model_params)

  
	def setup_data(self):
		#print("SPECTRUM BEFORE PROCESSING: ", self.data.spectrum)
		
		if self._SELECT_COLS:
			print("SELECTING COLS FROM HERE ____ _______ ______________________________")
			self.data.select_cols(self._MIN_VARIANCE, self.model_params)
			self._SELECT_COLS = False
			
		#self.data.replace_bad_intervals()
		
		#print("SPECTRUM AFTER PROCESSING: ", self.data.spectrum)

  
	def setup_output(self):
		self.output = Output(self._OUTPUT_PATH)
		self.output.write(["timestamp", "b0", "scaled_score", "anomaly_score"])

  
	def generate_model_input(self, index):
		"""
		Parameters:
		------------
		@param index (int)
			The row index to create the input from
		"""
		record = self.data.generate_record(index)
		self.modelInput = dict(zip(self.data.headers, record))
	
		for b in self.data.headers:
			self.modelInput[b] = float(self.modelInput[b])
	
		self.modelInput["float"] = self.modelInput["timestamp"]
		self.modelInput["timestamp"] = datetime.datetime.fromtimestamp(self.modelInput["timestamp"])
		

	def run_model(self):
		result = self.model.run(self.modelInput)
		anomalyScore = result.inferences['anomalyScore']
		scaledScore = anomalyScore * self._ANOMALY_SCALE_FACTOR
		return anomalyScore, scaledScore
		
	
	def output_results(self, anomalyScore, scaledScore):
		"""
		Parameters:
		------------
		@param anomalyScore (float)
			The score generated by the HTM model for a given modelInput
		
		@param scaledScore (float)
			The anomalyScore multiplied by the chosen scale factor
		"""
		if anomalyScore > self._ANOMALY_THRESHOLD:
			self.anomaly_count = self.anomaly_count + 1
			self._LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", self.modelInput["timestamp"], anomalyScore)
		self.output.write([self.modelInput["float"], self.modelInput["b0"], scaledScore, "%.3f" % anomalyScore])
	
	
	def runAstroAnomaly(self):
		print("Running with min var of: ", self._MIN_VARIANCE)
		self.setup_output()
		self.setup_data()
  
		self.model = self.createModel()
		self.model.enableInference({'predictedField': 'b0'})  # doesn't matter for anomaly detection
	  
		for i in tqdm.tqdm(range(0, self.data.data_size, 1), desc='% Complete'):
			self.generate_model_input(i)
			anomalyScore, scaledScore = self.run_model()
			self.output_results(anomalyScore, scaledScore)
    
		print("Anomaly Scores have been written to", self._OUTPUT_PATH)
		self.output.close()
				


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
				'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
				'b28', 'b29']
	
	# Build and run anomaly detector 
	anomaly_file = "spectrum4.csv"
	detector = AstroHTM(0.001, headers, model_params.MODEL_PARAMS, anomaly_file, select_cols=True)
	detector.runAstroAnomaly()
	
	# Write original spectra to csv
	spectrum_file = 'nu80002092008A01_x2_bary_binned10.csv'
	detector.data.write_data_to_csv(spectrum_file)
	
	# Visualize original spectra with anomalies
	output_filename = 'spectra.png'
	viz = Viz(spectrum_file, 0, 6000)
	viz.choose_spectra(0, 30)
	viz.add_anomalies(anomaly_file)
	viz.plot(output_filename)
	print "Plot was saved to", output_filename
