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

import model_params
import nupic_anomaly_output

import astropy
from astropy.io import fits
from astropy.io import ascii


_LOGGER = logging.getLogger(__name__)

##############################################
im = fits.open('./test_data.flc')
print("im.info: ", im.info)

data = astropy.table.Table.read(im[1])
data = im[1].data

print("data: ", data)
print("LENGTH: ", len(data))

for i in range(len(data)):
    if data[i][2] != 100:
        data[i][2] = 0.0

print("Filling missing values completed.............")
print(data)


headers = ['timestamp', 'value']
##############################################

_OUTPUT_PATH = "test_data.csv"  # changed name of output file



def runOutput():
  total = len(data)

  csvWriter = csv.writer(open(_OUTPUT_PATH,"wb"))
  csvWriter.writerow(["timestamp", "value"])

  col0 = data.field(0)  # time
  col1 = data.field(2)  # value
  for i in tqdm.tqdm(range(0, total, 1), desc='% Complete'):
    record = [col0[i], col1[i]]
    modelInput = dict(zip(headers, record))
    modelInput["value"] = float(modelInput["value"])
    floattime = float(modelInput['timestamp'])

    csvWriter.writerow([floattime, modelInput["value"]])

  print("Data has been written to",_OUTPUT_PATH)

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  runOutput()
