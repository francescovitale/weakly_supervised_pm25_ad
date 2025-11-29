:: Options:
:: clustering_mode=[auto, manual]
:: variance_threshold=<float>
:: min_samples_fraction=[0,1]
:: noise_factor=[0,1]
:: eps=<float>
:: min_samples=<integer>

set clustering_mode=auto
set variance_threshold=0.99
set min_samples_fraction=0.5
set noise_factor=0.9

for /D %%p IN ("Output\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

python weakly_supervised_labeling.py %clustering_mode% %variance_threshold% %min_samples_fraction% %noise_factor%

