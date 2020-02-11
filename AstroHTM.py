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

import datetime
import logging
import tqdm
import copy
import sys
import numpy as np

#from pkg_resources import resource_filename
from nupic.frameworks.opf.model_factory import ModelFactory
#from nupic.data.inference_shifter import InferenceShifter
#from nupic.algorithms import anomaly_likelihood as AL

from viz import Viz
from output import Output
from data import Data
import model_params



class AstroHTM(object):
	"""
	This class represents the hierarchical temporal memory algorithm to be 
	applied to astronomy data.
	"""
	_LOGGER = logging.getLogger(__name__)
	
	#_SOURCE_FILE = './srcB_3to40_cl_barycorr_binned_multiD.fits'
	#_SOURCE_FILE = 'nu80002092008A01_x2_bary_binned10.fits'
	#_SOURCE_FILE = 'ni1103010157_0mpu7_cl_binned10.fits'
	
	_ANOMALY_SCALE_FACTOR = 300
	anomaly_count = 0
	encoder_resolution_set = False


	
	def __init__(self, source_file, min_var, headers, model_params, output_path, select_cols=False, threshold = 0.5):
		"""
		Parameters:
		------------
		@param source_file
			The fits file name containing the data to run anomaly detection on.
			Do not include '.fits' at the end of the filename, it is implied.
			
		@param min_var (int)
			The minimum variance a spectrum column will have else it is dropped
		
		@param model_params (dictionary)
			The dictionary of parameters for the HTM model to used
			
		@param output_path (string)
			The filename to whioh the output will be written (.csv extected)
			
		@param select_cols (boolean)
			True if columns should be removed for having low variance, false otherwise
			Default value is False
			
		@param threshold (float from 0 to 1)
			Determines how high an anomaly score must be in order to register as an anomaly
			Default value is 0.5
		"""
		self._SOURCE_FILE = source_file
		self._MIN_VARIANCE = min_var		
		self._SELECT_COLS = select_cols
		self._ANOMALY_THRESHOLD = threshold
		self.data = Data(self._SOURCE_FILE, headers)
		print len(headers), "headers given originally"
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
				#print "RESOLUTION:", resolution

			self.model_params['modelParams']['sensorParams']['encoders'][field] = encoder
			
		self.encoder_resolution_set = True
		
		for i in self.model_params['modelParams']['sensorParams']['encoders'].keys():
			if i not in self.data.headers:
				self.model_params['modelParams']['sensorParams']['encoders'].pop(i)


	def createModel(self):
		self._setRandomEncoderResolution()
		return ModelFactory.create(self.model_params)

  
	def setup_data(self):
		#print("SPECTRUM BEFORE PROCESSING: ", self.data.spectrum)
		
		if self._SELECT_COLS:
			self.data.select_cols(self._MIN_VARIANCE, self.model_params)
			self._SELECT_COLS = False
			
		#self.data.replace_bad_intervals()
		
		#print("SPECTRUM AFTER PROCESSING: ", self.data.spectrum)

  
	def setup_output(self):
		"""
		Create Output object and write header line to csv file
		"""
		self.output = Output(self._OUTPUT_PATH)
		self.output.write(["timestamp", str(self.data.headers[1]), "scaled_score", "anomaly_score"])

  
	def generate_model_input(self, index):
		"""
		Generate the index-th input point for the model to analyze 
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
		Output one line of results corresponding to one datapoint's analysis by the model
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
		self.output.write([self.modelInput["float"], self.modelInput[self.data.headers[1]], scaledScore, "%.3f" % anomalyScore])
	
	
	def runAstroAnomaly(self):
		"""
		Process data, setup Output, and build and run anomaly detection model
		"""
		print "...Running with min var of: ", self._MIN_VARIANCE
		self.setup_output()
		self.setup_data()
  
		self.model = self.createModel()
		self.model.enableInference({'predictedField': self.data.headers[1]})  # doesn't matter for anomaly detection
	  
		for i in tqdm.tqdm(range(0, self.data.data_size, 1), desc='% Complete'):
			self.generate_model_input(i)
			anomalyScore, scaledScore = self.run_model()
			self.output_results(anomalyScore, scaledScore)
    
		print "Anomaly Scores have been written to", self._OUTPUT_PATH
		print self.data.headers
		self.output.close()
				


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
				'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
				'b28', 'b29','b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39' ]
	
	# Parse command line arguments
	args = sys.argv[1:]
	VARIANCE_CUTOFF = float(args[0])
	TIME_MIN        = int(args[1])
	TIME_MAX        = int(args[2])
	SOURCE_FILE     = args[3]
	PLOT_FILE       = args[4]
	NUM_CHANNELS    = int(args[5])
	
	
	# Build and run anomaly detector 
	anomaly_file = "spectrum4.csv"
	detector = AstroHTM(SOURCE_FILE + '.fits', VARIANCE_CUTOFF, headers, model_params.MODEL_PARAMS, anomaly_file, select_cols=True)
	detector.runAstroAnomaly()
	
	# Write original spectra to csv
	spectrum_file = SOURCE_FILE + '.csv'
	detector.data.write_data_to_csv(spectrum_file)
	
	# Visualize original spectra with anomalies
	viz = Viz(spectrum_file, TIME_MIN, TIME_MAX, cutoffs=detector.data.cutoffs)
	viz.choose_spectra(0, NUM_CHANNELS)
	viz.add_anomalies(anomaly_file)
	#print viz.df
	viz.plot(PLOT_FILE)
	print "Plot was saved to", PLOT_FILE
