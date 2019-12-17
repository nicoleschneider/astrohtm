import unittest

from astro_anomaly import AstroHTM


class Tester(unittest.TestCase):
    def test_init(self):
        """
        Test that AstroHTM object was set up correctly.
        """
        obj = AstroHTM(250)
        self.assertEqual(obj._MIN_VARIANCE, 250)
		
    def test_get_anomaly_count(self):
        """
        Test that anomaly_count is updated correctly.
        """
        pass
		
    def test_setRandomEncoderResolution(self):
        """
        Test that RandomEncoder resolutions are set properly.
        """
        pass
		
    def test_createModel(self):
        """
        Test that the mdoel is created properly.
        """
        pass
		
    def test_select_cols(self):
        """
        Test that the right columns are eliminated.
        """
        pass
		
    def test_replace_bad_intervals(self):
        """
        Test that blank intervals are filled with power spectrum noise.
        """
        pass
		
    def test_preprocess(self):
        """
        Test that 
        """
        pass
		
    def test_extract_cols_from_data(self):
        """
        Test that 
        """
        pass
		
    def test_setup_output(self):
        """
        Test that 
        """
        pass
		
    def test_generate_record(self):
        """
        Test that 
        """
        pass
		
    def test_generate_model_input(self):
        """
        Test that 
        """
        pass
		
    def test_run_model(self):
        """
        Test that 
        """
        pass
		
    def test_output_results(self):
        """
        Test that 
        """
        pass
		
    def test_close_output(self):
        """
        Test that 
        """
        pass
		
    def test_runAstroAnomaly(self):
        """
        Test that 
        """
        pass
		
    def test_write_data_to_csv(self):
        """
        Test that 
        """
        pass
		
if __name__ == '__main__':
    unittest.main()
	
