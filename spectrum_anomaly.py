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


class AstroTest(object):

	_LOGGER = logging.getLogger(__name__)
	_SOURCE_FILE = './srcB_3to40_cl_barycorr_binned_multiD.fits'
	_OUTPUT_PATH = "spectrum5.csv"
  
	_MIN_VARIANCE = 0
	_ANOMALY_THRESHOLD = 0.5
	_ANOMALY_SCALE_FACTOR = 300
	_SELECT_COLS = True

	anomaly_count = 0

	# minimum metric value of test_data.flc
	_INPUT_MIN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0]  # changed to min flux value

	# maximum metric value of test_data.flc
	_INPUT_MAX = [439, 140, 41, 17, 12, 9, 9, 8, 11, 9, 10, 7, 6, 6, 7, 5, 7, 5, 7, 6, 6, 5, 5, 6, 5, 6, 4, 5, 5, 5] # changed to max flux value
	_INPUT_MAX = [300]*30

	def __init__(self, min_var):
		self._MIN_VARIANCE = min_var

		self.hdu_list = fits.open(self._SOURCE_FILE)
		self.data = astropy.table.Table.read(self.hdu_list[1])
		self.data_size = len(self.data)
		self.headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
			'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
			'b28', 'b29']
			
		#self.model = self.createModel()

	def get_anomaly_count(self):
		return self.anomaly_count

	def _setRandomEncoderResolution(self, minResolution=0.001):
		"""
		Given model params, figure out the correct resolution for the
		RandomDistributed encoders. Modifies params in place.
		"""
		fields = self.headers[1:]
		print(fields)

		for i, field in enumerate(fields):
			encoder = (model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"][field])

		if encoder["type"] == "RandomDistributedScalarEncoder":
			rangePadding = abs(self._INPUT_MAX[i] - self._INPUT_MIN[i]) * 0.2
			minValue = self._INPUT_MIN[i] - rangePadding
			maxValue = self._INPUT_MAX[i] + rangePadding
			resolution = max(minResolution, (maxValue - minValue) / encoder.pop("numBuckets") )
			encoder["resolution"] = resolution

	def createModel(self):
		self._setRandomEncoderResolution()
		return ModelFactory.create(model_params.MODEL_PARAMS)

	def select_cols(self, col):
		if self._SELECT_COLS:

			for i, element in enumerate(self.headers[1:]):    
				col[:,i] =  col[:,i] - np.mean(col[:,i]) #/( np.std(col[:,i]) )
			#  _INPUT_MAX[i] =  _INPUT_MAX[i] - np.mean(col[:,i]) #/( np.std(col[:,i]) )
				col[:,i] = map(lambda x: max(x,0), col[:,i])

				if np.var(col[:,i]) < self._MIN_VARIANCE:
					self.headers.remove(element)
					model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"].pop(element)
			  
		 # _INPUT_MAX = map(lambda x: max(x,0), _INPUT_MAX)	  
			print(self._INPUT_MAX)
			print("ENDED UP USING ", len(self.headers), " columns total")
		  
		return col
		
		
	def replace_bad_intervals(self, col):
		df = pd.read_csv("psd1.csv")
		saved_column = df['final signal'].values
		saved_column = np.append(saved_column, [0])
		col[:,0] = saved_column

		print col
		return col
  
	def preprocess(self, col):
		col = self.select_cols(col)
		col = self.replace_bad_intervals(col)
		return col
  
	def extract_cols_from_data(self):
		col0 = self.data.field(0)
		# self.data.field(1) is the image which we will ignore for now
		col = self.data.field(2)
		col = self.preprocess(col)
		return col0, col
  
	def setup_output(self):
		shifter = InferenceShifter()
		output = nupic_anomaly_output.NuPICFileOutput("astronomy-data")
		f = open(self._OUTPUT_PATH,"wb")
		csvWriter = csv.writer(f)
		csvWriter.writerow(["timestamp", "b0", "b1", "scaled_score", "anomaly_score"])
		return f, output, csvWriter

	def generate_record(self, bs, col0, col, index):
		record = []
		for x, label in enumerate(bs):
			col_number = int(label[1:])
			record.append(col[:,col_number][index]) 
	
		record = np.insert(record, 0, col0[index]-col0[0]) # insert timestamp value to front of record
		return record
  
	def generate_model_input(self, col0, col, index):
		bs = self.headers[1:]
		record = self.generate_record(bs, col0, col, index)
		modelInput = dict(zip(self.headers, record))
	
		for b in bs:
			modelInput[b] = float(modelInput[b])
	
		floattime = float(modelInput['timestamp'])
		modelInput["timestamp"] = datetime.datetime.fromtimestamp(floattime)
		return modelInput

	def run_model(self, modelInput):
		result = self.model.run(modelInput)
		anomalyScore = result.inferences['anomalyScore']
		scaledScore = anomalyScore * self._ANOMALY_SCALE_FACTOR
		return anomalyScore, scaledScore
	
	def output_results(self, output, csvWriter, modelInput, anomalyScore, scaledScore):
		output.write(modelInput['timestamp'], modelInput['b0'], 0, anomalyScore)
	  
		if anomalyScore > self._ANOMALY_THRESHOLD:
			self.anomaly_count = self.anomaly_count + 1
			self._LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", modelInput["timestamp"], anomalyScore)
		csvWriter.writerow([modelInput["timestamp"], modelInput["b0"], scaledScore, "%.3f" % anomalyScore])

	def close_output(self, f, output):
		print("Anomaly Scores have been written to", self._OUTPUT_PATH)
		output.close()
		f.close()
	
	def runAstroAnomaly(self):
		print("Running with min var of: ", self._MIN_VARIANCE)
		f, output, csvWriter = self.setup_output()
		col0, col = self.extract_cols_from_data()
  
		self.model = self.createModel()
		self.model.enableInference({'predictedField': 'b0'})  # doesn't matter for anomaly detection
	  
		for i in tqdm.tqdm(range(0, self.data_size, 1), desc='% Complete'):
			modelInput = self.generate_model_input(col0, col, i)
			anomalyScore, scaledScore = self.run_model(modelInput)
			self.output_results(output, csvWriter, modelInput, anomalyScore, scaledScore)
    
		self.close_output(f, output)
		
  


	def objective(self, trial):
		self._MIN_VARIANCE = trial.suggest_int('_MIN_VARIANCE', 239, 241)
		self.runAstroAnomaly()
		return -1 * self.get_anomaly_count()

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	
	astro_test = AstroTest(233)
	print(astro_test._MIN_VARIANCE)
	
	study = optuna.create_study()
	study.optimize(astro_test.objective, n_trials=1)

	
