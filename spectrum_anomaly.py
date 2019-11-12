

import csv
import datetime
import logging
import tqdm
import numpy as np

from pkg_resources import resource_filename
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter

import model_params
import nupic_anomaly_output

import astropy
from astropy.io import fits
from astropy.io import ascii


_LOGGER = logging.getLogger(__name__)

##############################################
im = fits.open('./srcA_3to40_cl_barycorr_binned_multiD.fits')
print("im.info: ", im.info)


data = astropy.table.Table.read(im[1])
data = im[1].data

#print('0: ', data[0][0])
#print('1: ', data[0][1])
#print('2: ', data[0][2])



print("LENGTH: ", len(data))

#print('df0: ', data.field(0))
#print('df1: ', data.field(2)[:,0])
#print('df2: ', data.field(2)[:,1])


#for i in range(len(data)):
#    if data[i][2] != 100:
#        data[i][2] = 0.0

#print("Filling missing values completed.............")



headers = ['timestamp', 'b0', 'b1']
##############################################

_OUTPUT_PATH = "spectrum1.csv"  # changed name of output file

_ANOMALY_THRESHOLD = 0.5

# minimum metric value of test_data.flc
_INPUT_MIN = 0  # changed to min flux value

# maximum metric value of test_data.flc
_INPUT_MAX = 22#636 # changed to max flux value


def _setRandomEncoderResolution(minResolution=0.001):
  """
  Given model params, figure out the correct resolution for the
  RandomDistributed encoders. Modifies params in place.
  """
  fields = ["b0", "b1"]

  for field in fields:
    encoder = (
      model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"][field]
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
  return ModelFactory.create(model_params.MODEL_PARAMS)


def runAstroAnomaly():
  model = createModel()
  model.enableInference({'predictedField': 'b1'})
  total = len(data)

  shifter = InferenceShifter()
  output = nupic_anomaly_output.NuPICFileOutput("astronomy-data")

  #with open (_INPUT_DATA_FILE) as fin:
    #reader = data
  csvWriter = csv.writer(open(_OUTPUT_PATH,"wb"))
  csvWriter.writerow(["timestamp", "b0", "b1", "anomaly_score"])

  col0 = data.field(0)
  # data.field(1) is the image which we will ignore for now
  col1 = data.field(2)[:,0]
  col2 = data.field(2)[:,1]
  col3 = data.field(2)[:,2]
  for i in tqdm.tqdm(range(0, total, 1), desc='% Complete'):
    subheaders = ["timestamp", "b0", "b1"]
    subrecord = [col0[i], col1[i], col2[i]]
    record = dict(zip(subheaders, subrecord))
    modelInput = record

    #modelInput["b0"] = float(modelInput["b0"])
    #modelInput["b1"] = float(modelInput["b1"])
    #modelInput["b2"] = float(modelInput["b2"])
    modelInput["b0"] = float(modelInput["b0"])
    modelInput["b1"] = float(modelInput["b1"])

    floattime = float(modelInput['timestamp'])
    modelInput["timestamp"] = datetime.datetime.fromtimestamp(floattime)
    result = model.run(modelInput)
    prediction = 0
    anomalyScore = result.inferences['anomalyScore']
    output.write(modelInput['timestamp'], modelInput['b0'], modelInput['b1'], anomalyScore)


    if anomalyScore > _ANOMALY_THRESHOLD:
      _LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.",                      result.rawInput["timestamp"], anomalyScore)
    csvWriter.writerow([floattime, modelInput["b0"], modelInput["b1"],
            "%.3f" % anomalyScore])

  print("Anomaly Scores have been written to",_OUTPUT_PATH)
  output.close()

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  runAstroAnomaly()


