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
A simple client to create a HTM anomaly detection model for astro_test dataset.
The script prints out all records that have an abnormally high anomaly
score.
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

import model_params
import nupic_anomaly_output

import astropy
from astropy.io import fits
from astropy.io import ascii


class Output(object):
	shifter = InferenceShifter()
	fileoutput = nupic_anomaly_output.NuPICFileOutput("astronomy-data")
	
	def __init__(self, output_path):
		self.fid = open(output_path, "wb")
		self.csvWriter = csv.writer(self.fid)
		
	def close(self):
		self.fileoutput.close()
		self.fid.close()

class Data(object):
	h = 0
	
	def __init__(self, source_file, headers):
		self.headers = headers
	
		self.hdu_list = fits.open(source_file)
		self.table = astropy.table.Table.read(self.hdu_list[1])
		self.data_size = len(self.table)
		print("LENGTH IS: ", self.data_size)
		print("SELF.DATATABLE is: ", self.table)
		
		self.timestamps = self.table.field(0)
		self.images = self.table.field(1) # we ignore for now
		self.spectrum = self.table.field(2)
		
		
	def select_cols(self, min_variance):
		for i, element in enumerate(self.headers[1:]):    
			self.spectrum[:,i] =  self.spectrum[:,i] - np.mean(self.spectrum[:,i]) #/( np.std(self.spectrum[:,i]) )
		#  _INPUT_MAX[i] =  _INPUT_MAX[i] - np.mean(self.spectrum[:,i]) #/( np.std(self.dspectrum[:,i]) )
			self.spectrum[:,i] = map(lambda x: max(x,0), self.spectrum[:,i])

			if np.var(self.spectrum[:,i]) < min_variance:
				self.headers.remove(element)
				model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"].pop(element)
			  
		 # _INPUT_MAX = map(lambda x: max(x,0), _INPUT_MAX)	  
		#print(self._INPUT_MAX)
		print("ENDED UP USING ", len(self.headers), " columns total")
		#self._SELECT_COLS = True
		
	def replace_bad_intervals(self):
		df = pd.read_csv("psd1.csv")
		saved_column = df['final signal'].values
		saved_column = np.append(saved_column, [0])
		self.spectrum[:,0] = saved_column

		print self.spectrum


class AstroHTM(object):

	_LOGGER = logging.getLogger(__name__)
	_SOURCE_FILE = './srcB_3to40_cl_barycorr_binned_multiD.fits'
	#_SOURCE_FILE = 'nu80002092008B01_x2_bary_binned10.fits'
	#_SOURCE_FILE = 'ni1103010157_0mpu7_cl_binned10.fits'
	_OUTPUT_PATH = "spectrum5.csv"
  
	_MIN_VARIANCE = 0
	_ANOMALY_THRESHOLD = 0.5
	_ANOMALY_SCALE_FACTOR = 300
	_SELECT_COLS = True

	anomaly_count = 0
	encoder_resolution_set = False

	# minimum metric value of test_data.flc
	_INPUT_MIN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0]  # changed to min flux value

	# maximum metric value of test_data.flc
	_INPUT_MAX = [439, 140, 41, 17, 12, 9, 9, 8, 11, 9, 10, 7, 6, 6, 7, 5, 7, 5, 7, 6, 6, 5, 5, 6, 5, 6, 4, 5, 5, 5] # changed to max flux value
	_INPUT_MAX = [300]*30

	def __init__(self, min_var):
		self._MIN_VARIANCE = min_var	

		headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
			'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
			'b28', 'b29']
			
		self.data = Data(self._SOURCE_FILE, headers)
			
		#self.model = self.createModel()
		self.encoders = model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"]

	def get_anomaly_count(self):
		return self.anomaly_count

	def _setRandomEncoderResolution(self, minResolution=0.001):
		"""
		Given model params, figure out the correct resolution for the
		RandomDistributed encoders. Modifies params in place.
		"""
		fields = self.data.headers[1:]
		
		print(fields)
		print("GONNA SET RESOLUTIoN___________________________________________")

		for i, field in enumerate(fields):
			encoder = (copy.deepcopy(self.encoders[field]))

			if encoder["type"] == "RandomDistributedScalarEncoder":
				rangePadding = abs(self._INPUT_MAX[i] - self._INPUT_MIN[i]) * 0.2
				minValue = self._INPUT_MIN[i] - rangePadding
				maxValue = self._INPUT_MAX[i] + rangePadding
				resolution = max(minResolution, (maxValue - minValue) / encoder.pop("numBuckets") )
				encoder["resolution"] = resolution

			self.encoders[field] = encoder
			
		self.encoder_resolution_set = True

	def createModel(self):
		if not self.encoder_resolution_set:
			self._setRandomEncoderResolution()
		return ModelFactory.create(model_params.MODEL_PARAMS)

  
	def setup_data(self):
		print("SPECTRUM BEFORE PROCESSING: ", self.data.spectrum)
		
		if self._SELECT_COLS:
			self.data.select_cols(self._MIN_VARIANCE)
			self._SELECT_COLS = True
			
		#self.data.replace_bad_intervals()
		
		print("SPECTRUM AFTER PROCESSING: ", self.data.spectrum)

  
	def setup_output(self):
		self.output = Output(self._OUTPUT_PATH)
		self.output.csvWriter.writerow(["timestamp", "b0", "scaled_score", "anomaly_score"])

	def generate_record(self, bs, index):
		record = []
		for x, label in enumerate(bs):
			col_number = int(label[1:])
			record.append(self.data.spectrum[:,col_number][index]) 
	
		record = np.insert(record, 0, self.data.timestamps[index]-self.data.timestamps[0]) # insert timestamp value to front of record
		return record
  
	def generate_model_input(self, index):
		bs = self.data.headers[1:]
		record = self.generate_record(bs, index)
		self.modelInput = dict(zip(self.data.headers, record))
	
		for b in bs:
			self.modelInput[b] = float(self.modelInput[b])
	
		floattime = float(self.modelInput['timestamp'])
		self.modelInput["timestamp"] = datetime.datetime.fromtimestamp(floattime)


	def run_model(self):
		result = self.model.run(self.modelInput)
		anomalyScore = result.inferences['anomalyScore']
		scaledScore = anomalyScore * self._ANOMALY_SCALE_FACTOR
		return anomalyScore, scaledScore
	
	def output_results(self, anomalyScore, scaledScore):
		self.output.fileoutput.write(self.modelInput['timestamp'], self.modelInput['b0'], 0, anomalyScore)
		
		if anomalyScore > self._ANOMALY_THRESHOLD:
			self.anomaly_count = self.anomaly_count + 1
			self._LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", self.modelInput["timestamp"], anomalyScore)
		self.output.csvWriter.writerow([self.modelInput["timestamp"], self.modelInput["b0"], scaledScore, "%.3f" % anomalyScore])

	def close_output(self):
		print("Anomaly Scores have been written to", self._OUTPUT_PATH)
		self.output.close()
	
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
    
		self.close_output()
		
	def write_data_to_csv(self, output_file):
		csvWriter = csv.writer(open(output_file,"wb"))
		headers = ['timestamp', 'value1', 'value2', 'value3', 'value4', 'value5']
		csvWriter.writerow(headers)

		col0 = self.data.table.field(0)  # time
		col1 = self.data.table.field(2)[:,0]  # value1
		col2 = self.data.table.field(2)[:,1]  # value2
		col3 = self.data.table.field(2)[:,2]  # value3
		col4 = self.data.table.field(2)[:,3]  # value4
		col5 = self.data.table.field(2)[:,4]  # value5
		print(col0)
		print(col1)
		
		for i in tqdm.tqdm(range(0, self.data.data_size, 1), desc='% Complete'):
			record = [col0[i], col1[i], col2[i], col3[i], col4[i], col5[i]]
			modelInput = dict(zip(headers, record))
			modelInput["value1"] = float(modelInput["value1"])
			modelInput["value2"] = float(modelInput["value2"])
			modelInput["value3"] = float(modelInput["value3"])
			modelInput["value4"] = float(modelInput["value4"])
			modelInput["value5"] = float(modelInput["value5"])
			floattime = float(modelInput['timestamp'])

			csvWriter.writerow([floattime, modelInput["value1"], modelInput["value2"], modelInput["value3"], modelInput["value4"], modelInput["value5"]])

		print("Original data has been written to",output_file)
		


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	#astro_test.write_data_to_csv('new_dataQPO10.csv')
