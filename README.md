# Execution instructions and project description

The calibration, forecasting and anomaly detection experiments can be replicated by running, respectively, the Jupyter notebooks "calibration_experiment.ipynb" and "anomaly_detection_experiment.ipynb" in any Jupyter-compliant runtime environment (e.g., Google Colab). Make sure the relative/absolute indexing of the AD and C folders is correct, as these contain the data being used within the C/Input and AD/Input subdirectories.  

The results of the calibration experiment will be available after executing the calibration_experiment.ipynb notebook in the C/Output folder. Specifically, for each time frame, the uncalibrated and calibrated versions of the F+A and PM_{2.5,1..4} data related to each day of November will be saved as .csv files.

Similarly, the results of the anomaly detection experiment will be available after executing the anomaly_detection_experiment.ipynb notebook in the AD/Output folder. This folder contains two subdirectories: Models and MSE. Models contains the weights of the time frame-wise LSTM networks, whereas MSE contains the results obtained when comparing the forecasted version of the normal test and anomalous PM$_{2.5}$ for each data type.
