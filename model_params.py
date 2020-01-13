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

MODEL_PARAMS = {
  "inferenceArgs": {
    "predictionSteps": [1],
    "predictedField": "b0",
    "inputPredictedField": "auto"
  },
  "aggregationInfo": {
      "seconds": 0,
      "fields": [],
      "months": 0,
      "days": 0,
      "years": 0,
      "hours": 0,
      "microseconds": 0,
      "weeks": 0,
      "minutes": 0,
      "milliseconds": 0
    },
  "model": "HTMPrediction",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "inferenceType": "TemporalAnomaly",
    "sensorParams": {
      "encoders": {
        "timestamp_timeOfDay": {
          "type": "DateEncoder",
          "timeOfDay": [
            21,
            9.49
          ],
          "fieldname": "timestamp",
          "name": "timestamp"
        },
        "timestamp_dayOfWeek": None,
        "timestamp_weekend": None,
        "b0": {
          "name": "b0",
          "fieldname": "b0",
          "seed": 42,
          "numBuckets": 30,
          "type": "RandomDistributedScalarEncoder",
        },
		"b1": {
          "name": "b1",
          "fieldname": "b1",
          "seed": 42,
          "numBuckets": 37,
          "type": "RandomDistributedScalarEncoder",
        },
		"b1": {
          "name": "b1",
          "fieldname": "b1",
          "seed": 42,
          "numBuckets": 150,
          "type": "RandomDistributedScalarEncoder",
        },
	    "b2": {
          "name": "b2",
          "fieldname": "b2",
          "seed": 42,
          "numBuckets": 171,
          "type": "RandomDistributedScalarEncoder",
        },
		"b3": {
          "name": "b3",
          "fieldname": "b3",
          "seed": 42,
          "numBuckets": 171,
          "type": "RandomDistributedScalarEncoder",
        },
		"b4": {
          "name": "b4",
          "fieldname": "b4",
          "seed": 42,
          "numBuckets": 171,
          "type": "RandomDistributedScalarEncoder",
        },
		"b5": {
          "name": "b5",
          "fieldname": "b5",
          "seed": 42,
          "numBuckets": 171,
          "type": "RandomDistributedScalarEncoder",
        },
		"b6": {
          "name": "b6",
          "fieldname": "b6",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b7": {
          "name": "b7",
          "fieldname": "b7",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b8": {
          "name": "b8",
          "fieldname": "b8",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b9": {
          "name": "b9",
          "fieldname": "b9",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b10": {
          "name": "b10",
          "fieldname": "b10",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b11": {
          "name": "b11",
          "fieldname": "b11",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
	    "b12": {
          "name": "b12",
          "fieldname": "b12",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b13": {
          "name": "b13",
          "fieldname": "b13",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b14": {
          "name": "b14",
          "fieldname": "b14",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b15": {
          "name": "b15",
          "fieldname": "b15",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b16": {
          "name": "b16",
          "fieldname": "b16",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b17": {
          "name": "b17",
          "fieldname": "b17",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b18": {
          "name": "b18",
          "fieldname": "b18",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b19": {
          "name": "b19",
          "fieldname": "b19",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b20": {
          "name": "b20",
          "fieldname": "b20",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b21": {
          "name": "b21",
          "fieldname": "b21",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
	    "b22": {
          "name": "b22",
          "fieldname": "b22",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b23": {
          "name": "b23",
          "fieldname": "b23",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b24": {
          "name": "b24",
          "fieldname": "b24",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b25": {
          "name": "b25",
          "fieldname": "b25",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b26": {
          "name": "b26",
          "fieldname": "b26",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b27": {
          "name": "b27",
          "fieldname": "b27",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b28": {
          "name": "b28",
          "fieldname": "b28",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
		"b29": {
          "name": "b29",
          "fieldname": "b29",
          "seed": 42,
          "numBuckets": 121,
          "type": "RandomDistributedScalarEncoder",
        },
      },
      "sensorAutoReset": None,
      "verbosity": 0
    },
    "spEnable": True,
    "spParams": {
      "spatialImp": "cpp",
      "potentialPct": 0.8,
      "columnCount": 2048,
      "globalInhibition": 1,
      "inputWidth": 0,
      "boostStrength": 0.0,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "spVerbosity": 0,
      "spatialImp": "cpp",
      "synPermActiveInc": 0.003,
      "synPermConnected": 0.2,
      "synPermInactiveDec": 0.0005
    },
    "trainSPNetOnlyIfRequested": False,
    "tmEnable": True,
    "tmParams": {
      "activationThreshold": 13,
      "cellsPerColumn": 32,
      "columnCount": 2048,
      "globalDecay": 0.0,
      "initialPerm": 0.21,
      "inputWidth": 2048,
      "maxAge": 0,
      "maxSegmentsPerCell": 128,
      "maxSynapsesPerSegment": 32,
      "minThreshold": 10,
      "newSynapseCount": 20,
      "outputType": "normal",
      "pamLength": 3,
      "permanenceDec": 0.1,
      "permanenceInc": 0.1,
      "seed": 1960,
      "temporalImp": "cpp",
      "verbosity": 0
    },
    "clEnable": False,
    "clParams": {
      "alpha": 0.035828933612157998,
      "regionName": "SDRClassifierRegion",
      "steps": "1",
      "verbosity": 0
    },
    "anomalyParams": {
      "anomalyCacheRecords": None,
      "autoDetectThreshold": None,
      "autoDetectWaitRecords": 5030
    }
  }
}
