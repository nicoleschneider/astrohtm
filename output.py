import csv

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
