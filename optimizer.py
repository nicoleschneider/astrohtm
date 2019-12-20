from astro_anomaly import AstroHTM
import optuna
import model_params
	
	
def objective(trial):
	headers = ['timestamp', 'b0', 'b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
				'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 
				'b28', 'b29']
	astro_test = AstroHTM(249, headers, model_params.MODEL_PARAMS, "spectrum3.csv")
	print(astro_test._MIN_VARIANCE)
	
	astro_test._MIN_VARIANCE = trial.suggest_int('_MIN_VARIANCE', 249, 249)
	astro_test.runAstroAnomaly()
	anomaly_count = astro_test.get_anomaly_count()
	del astro_test
	return -1 * anomaly_count
		
		

	
study = optuna.create_study()
study.optimize(objective, n_trials=1)


# try different integers for value of model_params.MODEL_PARAMS["modelParams"]["sensorParams"]["encoders"]["b0"]["numBuckets"]
