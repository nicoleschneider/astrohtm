import unittest
import copy

from astro_anomaly import AstroHTM
import model_params


class Tester(unittest.TestCase):
	def test_init(self):
		"""
		Test that AstroHTM object was set up correctly.
		"""
		self.headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 
			'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 
			'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
			'b28', 'b29']
		obj = AstroHTM(250, copy.deepcopy(self.headers), model_params.MODEL_PARAMS, "spectrum5.csv", select_cols=True)
		self.assertEqual(obj._MIN_VARIANCE, 250)
		self.assertEqual(obj.data.headers, self.headers)
		obj.runAstroAnomaly()
		self.assertNotEqual(obj.data.headers, self.headers)
		
		
	def test_zero_data(self):
		"""
		Test that anomaly_count is updated correctly.
		"""
		#headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
		#		'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
		#		'b28', 'b29']
		headers=['timestamp', 'b0', 'b1']
		obj = AstroHTM(250, headers, model_params.MODEL_PARAMS, "spectrum6.csv")
		
		print("THIS IS ORIGINAL ____________________ ", obj.data.spectrum[1])
		obj.data.spectrum[1] = [0*x for x in obj.data.spectrum[1]]
		obj.data.spectrum[2] = [0*x for x in obj.data.spectrum[2]]
		print("THIS IS AFTER ____________________ ", obj.data.spectrum[1])
		print(obj.data.spectrum)
		
		for x in obj.data.spectrum[1]:
			self.assertEqual(x, 0)
			
		for x in obj.data.spectrum[2]:
			self.assertEqual(x, 0)
			
		obj.runAstroAnomaly()
		anomaly_count = obj.get_anomaly_count()
		self.assertEqual(anomaly_count, 27)
		
        
		
    # def test_get_anomaly_count(self):
        # """
        # Test that anomaly_count is updated correctly.
        # """
		# # assert anomaly_count >= 0
        # pass
		
    # def test_setRandomEncoderResolution(self):
        # """
        # Test that RandomEncoder resolutions are set properly.
        # """
        # pass
		
    # def test_createModel(self):
        # """
        # Test that the mdoel is created properly.
        # """
        # pass
		
    # def test_setup_data(self):
        # """
        # Test that the data is extracted properly.
        # """
		# # new self.data.spectrum has fewer or same columns than old
        # pass
		
		
    # def test_setup_output(self):
        # """
        # Test that 
        # """
        # pass
		
    # def test_generate_record(self):
        # """
        # Test that 
        # """
        # pass
		
    # def test_generate_model_input(self):
        # """
        # Test that 
        # """
        # pass
		
    # def test_run_model(self):
        # """
        # Test that 
        # """
		# # anomaly score is between 0 and 1
		# # scaled score is between 0 and scale factor
        # pass
		
    # def test_output_results(self):
        # """
        # Test that 
        # """
        # pass
			
    # def test_runAstroAnomaly(self):
        # """
        # Test that 
        # """
        # pass
		
		
if __name__ == '__main__':
    unittest.main()
	
