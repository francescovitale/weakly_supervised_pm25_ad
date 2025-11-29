:: Options:
:: cal_methods=[linear_regression random_forest xg_boost]
:: pd_variant=[im, ilp, hm]
:: n_clusters=<integer>
:: n_simulation_traces=<integer>

set cal_methods=xgboost

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%x in (%cal_methods%) do (

	mkdir Results\%%x
	
	for /D %%p IN ("Output\*") DO (
		del /s /f /q %%p\*.*
		for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
		rmdir "%%p" /s /q
	)
	
	python calibration_single_models.py %%x
	
	xcopy Output\ Results\%%x /E

)
