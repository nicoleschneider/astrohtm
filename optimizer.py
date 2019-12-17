from astro_anomaly import AstroHTM
import optuna
	
	
def objective(trial):
	astro_test = AstroHTM(249)
	print(astro_test._MIN_VARIANCE)
	
	astro_test._MIN_VARIANCE = trial.suggest_int('_MIN_VARIANCE', 249, 249)
	astro_test.runAstroAnomaly()
	anomaly_count = astro_test.get_anomaly_count()
	del astro_test
	return -1 * anomaly_count
		
		

	
study = optuna.create_study()
study.optimize(objective, n_trials=1)
