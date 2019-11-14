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

import csv
import datetime
import logging
import tqdm
import numpy as np

from pkg_resources import resource_filename
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter
from nupic.algorithms import anomaly_likelihood as AL

import model_params
import nupic_anomaly_output

import astropy
from astropy.io import fits
from astropy.io import ascii


_LOGGER = logging.getLogger(__name__)

FITS = fits.open('./srcB_3to40_cl_barycorr_binned_multiD.fits')
data = astropy.table.Table.read(FITS[1])
data = FITS[1].data
DATA_SIZE = len(data)
print("LENGTH: ", DATA_SIZE)

headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
        'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
        'b28', 'b29']

_OUTPUT_PATH = "spectrum3.csv"  # changed name of output file

_ANOMALY_THRESHOLD = 0.5

# minimum metric value of test_data.flc
_INPUT_MIN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0]  # changed to min flux value

# maximum metric value of test_data.flc
_INPUT_MAX = [439, 140, 41, 17, 12, 9, 9, 8, 11, 9, 10, 7, 6, 6, 7, 5, 7, 5, 7, 6, 6, 5, 5, 6, 5, 6, 4, 5, 5, 5] # changed to max flux value
_INPUT_MAX = [500]*30

MIN_VARIANCE = 2000
ANOMALY_SCALE_FACTOR = 450

def _setRandomEncoderResolution(minResolution=0.001):
  """
  Given model params, figure out the correct resolution for the
  RandomDistributed encoders. Modifies params in place.
"""

  fields = headers[1:]
  print(fields)

  for i, field in enumerate(fields):
    encoder = (
      model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"][field]
    )

    if encoder["type"] == "RandomDistributedScalarEncoder":
      rangePadding = abs(_INPUT_MAX[i] - _INPUT_MIN[i]) * 0.2
      minValue = _INPUT_MIN[i] - rangePadding
      maxValue = _INPUT_MAX[i] + rangePadding
      resolution = max(minResolution,
                       (maxValue - minValue) / encoder.pop("numBuckets")
                      )
      encoder["resolution"] = resolution


def createModel():
  _setRandomEncoderResolution()
  return ModelFactory.create(model_params.MODEL_PARAMS)

def select_cols(col):
  global headers

  for i, element in enumerate(headers[1:]):
    if np.var(col[:,i]) < MIN_VARIANCE:
      headers.remove(element)
      model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"].pop(element)

  print("ENDED UP USING ", len(headers), " columns total")
	
	
def runAstroAnomaly():
  shifter = InferenceShifter()
  output = nupic_anomaly_output.NuPICFileOutput("astronomy-data")

  csvWriter = csv.writer(open(_OUTPUT_PATH,"wb"))
  csvWriter.writerow(["timestamp", "b0", "b1", "scaled_score", "anomaly_score"])

  col0 = data.field(0)
  # data.field(1) is the image which we will ignore for now
  col = data.field(2)
  
  select_cols(col)
  
  model = createModel()
  model.enableInference({'predictedField': 'b0'})  # doesn't matter for anomaly detection
  
  bs = headers[1:]

  for i in tqdm.tqdm(range(0, DATA_SIZE, 1), desc='% Complete'):
  
    record = []
    for x, label in enumerate(bs):
      col_number = int(label[1:])
      record.append(col[:,col_number][i]) 
	  
    record = np.insert(record, 0, col0[i]-col0[0]) # insert timestamp value to front of record
    modelInput = dict(zip(headers, record))
	
    for b in bs:
      modelInput[b] = float(modelInput[b])

    floattime = float(modelInput['timestamp'])
    modelInput["timestamp"] = datetime.datetime.fromtimestamp(floattime)

    result = model.run(modelInput)
    anomalyScore = result.inferences['anomalyScore']
    scaledScore = anomalyScore * ANOMALY_SCALE_FACTOR
    #likelihood = AL.anomalyProbability(modelInput["b0"], anomalyScore, modelInput["timestamp"])

    output.write(modelInput['timestamp'], modelInput['b0'], 0, anomalyScore)
      
    if anomalyScore > _ANOMALY_THRESHOLD:
      _LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", 			result.rawInput["timestamp"], anomalyScore)
    csvWriter.writerow([floattime, modelInput["b0"], scaledScore,
	    "%.3f" % anomalyScore])

  print("Anomaly Scores have been written to",_OUTPUT_PATH)
  output.close()

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  runAstroAnomaly()
