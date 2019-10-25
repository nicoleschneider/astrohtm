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

from pkg_resources import resource_filename
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter

import model_params
import nupic_anomaly_output

from astropy.io import fits
import tqdm

_LOGGER = logging.getLogger(__name__)

#_INPUT_DATA_FILE = resource_filename("nupic.datafiles", "extra/nycTaxi/nycTaxi.csv")

#####################
im = fits.open('./test_data.flc')
print(im.info)

data = im[1].data
print(data)

headers = ['timestamp', 'value']
#####################

_OUTPUT_PATH = "astro_anomaly_scores12.csv"  # changed name of output file

_ANOMALY_THRESHOLD = 0.5

# minimum metric value of test_data.flc
_INPUT_MIN = 0  # changed to min flux value

# maximum metric value of test_data.flc
_INPUT_MAX = 1 # changed to max flux value


def _setRandomEncoderResolution(minResolution=0.001):
  """
  Given model params, figure out the correct resolution for the
  RandomDistributed encoder. Modifies params in place.
  """
  encoder = (
    model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"]["value"]
  )

  if encoder["type"] == "RandomDistributedScalarEncoder":
    rangePadding = abs(_INPUT_MAX - _INPUT_MIN) * 0.2
    minValue = _INPUT_MIN - rangePadding
    maxValue = _INPUT_MAX + rangePadding
    resolution = max(minResolution,
                     (maxValue - minValue) / encoder.pop("numBuckets")
                    )
    encoder["resolution"] = resolution


def createModel():
  _setRandomEncoderResolution()
  model = ModelFactory.create(model_params.MODEL_PARAMS)
  model.enableInference({'predictedField': 'value'})
  return model


def runAstroAnomaly():
  model = createModel()
  model.enableInference({'predictedField': 'value'})

  shifter = InferenceShifter()
  output = nupic_anomaly_output.NuPICPlotOutput("astronomy-data")

  total = len(data)
  #with open (_INPUT_DATA_FILE) as fin:
    #reader = data
  csvWriter = csv.writer(open(_OUTPUT_PATH,"wb"))
  csvWriter.writerow(["timestamp", "value", "anomaly_score"])

  col0 = data.field(0)
  col1 = data.field(2)
  for i in tqdm.tqdm(range(0, 15000, 1), desc='% Complete'):
    record = [col0[i], col1[i]]
    modelInput = dict(zip(headers, record))
    modelInput["value"] = float(modelInput["value"])
    modelInput["timestamp"] = datetime.datetime.fromtimestamp(float(modelInput["timestamp"])) 

    result = model.run(modelInput)
    result = shifter.shift(result)
    prediction = 0
    anomalyScore = result.inferences['anomalyScore']
    output.write(modelInput['timestamp'], modelInput['value'], prediction, anomalyScore)
      
    if anomalyScore > _ANOMALY_THRESHOLD:
      _LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", 			result.rawInput["timestamp"], anomalyScore)
      csvWriter.writerow([modelInput["timestamp"], modelInput["value"], 
	    "%.3f" % anomalyScore])

  print("Anomalies have been written to",_OUTPUT_PATH)
  output.close()

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  runAstroAnomaly()
